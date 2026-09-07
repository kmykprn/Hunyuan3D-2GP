# Cloud Run 移植のための事前計測

## これは何か

`roomplanner-web`（3D部屋模様替えアプリ）の家具生成を、外部APIではなく
**自前で動かす**ための準備。このリポジトリを GCP の Cloud Run（GPU）に載せる前に、
ローカルのGPUで生成時間とVRAM使用量を測る。

### なぜ測るのか

Cloud Run のGPUは**秒課金**なので、`生成時間 × 単価` がそのまま費用になる。
現時点の月額見積もりは「L4での生成時間」を推定値で置いており、そこが
2倍外れると月額も2倍動く。**見積もりの不確かさがこの一点に集約されている。**

あわせて、以下も同時に確定させる。

- CUDA拡張（`custom_rasterizer` / `differentiable_renderer`）のビルドが通るか
- L4の24GBに収まる profile はどれか
- モデルの重みが何GBあるか（＝コンテナイメージの大きさとコールドスタート時間）

## 背景：なぜ自前で動かすのか

外部APIは値上げ・仕様変更・廃止がこちらの都合と無関係に起きるため、
長期的な管理コストが高い。実際に以下が起きている。

- RunPod のエンドポイントが知らぬ間に削除されていた
- Tripo は V2 API を 2026年11月に停止する
- 旧実装は Synexa という外部APIに依存していたが、コードを読むまで気づけなかった

## 手順

### 1. セットアップ

README の指定バージョンに**厳密に**従うこと。ここを外すとCUDA拡張のビルドが通らない。

```bash
conda create -n hy3d python==3.10.9
conda activate hy3d

pip install torch==2.5.1 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/test/cu124
pip install -r requirements.txt

# CUDA拡張のビルド（ここが山場）
cd hy3dgen/texgen/custom_rasterizer       && python3 setup.py install && cd ../../..
cd hy3dgen/texgen/differentiable_renderer && python3 setup.py install && cd ../../..
```

### 2. 計測

```bash
bash measure.sh
```

profile 3 / 2 / 1 × 形状のみ / テクスチャ付き の**6パターン**を実行し、
所要時間とVRAM最大値を `measure_result.txt` に記録する。
初回はモデルの重みをダウンロードするため時間がかかる。

別の画像で測りたい場合は引数で渡す。

```bash
bash measure.sh path/to/image.png
```

### 3. 記録する数字

`measure_result.txt` の内容をそのまま残す。特に重要なのは以下。

| 項目 | 用途 |
|---|---|
| GPU名とVRAM | L4への換算係数を出す |
| profileごとの所要時間 | 秒課金なので直接コストになる |
| VRAM最大値 | L4の24GBに収まる profile を決める |
| HuggingFaceキャッシュのサイズ | イメージサイズとコールドスタートの見積もり |

## profile の意味

```
1 = HighRAM_HighVRAM     ← 最速
2 = HighRAM_LowVRAM
3 = LowRAM_HighVRAM      ← 既定（VRAM 9GB想定）
4 = LowRAM_LowVRAM       （6GB以上）
5 = VerylowRAM_LowVRAM
```

mmgp は「VRAMが足りない環境で動かす」ための仕組みで、重みをCPUとGPUの間で
退避させる代わりに時間がかかる。手元のPCなら「遅くても動く」のが正義だが、
**Cloud Run は秒課金なので遅い＝高い**。

L4 の 24GB は丸ごと確保されて課金されるので、使い切ったほうが得。
既定の 3 は VRAM 9GB 前提なので、**もっと速い側（1〜2）に振れるはず**。
それがどれだけ速くなるかを測るのがこの計測の主目的。

## ビルドで詰まったときによくある原因

- `nvcc` が見つからない → CUDA Toolkit の導入が必要
- torch の CUDA バージョンと、システムの CUDA バージョンが不一致
- `ninja` が入っていない
- C++ コンパイラのバージョンが古い

エラーログは捨てずに残すこと（`measure_*.log` に出力される）。

## 計測後にやること

1. 実測値で月額見積もりを引き直す
2. Dockerfile を書く（`nvidia/cuda:12.4.x-devel` ベース + 拡張のビルド + 重みの取り込み）
3. Cloud Run（GPU L4）にデプロイし、コールドスタートを含む実費用を測る
4. API層を作る（Cloud Run CPU + Cloud Tasks + Firestore + Cloud Storage）

### 現時点の見積もり（実測前）

ユーザー100人・月500生成の想定。

| 項目 | 月額 |
|---|---|
| Cloud Run GPU（L4） | $35 |
| Artifact Registry（20GBイメージ） | $2 |
| Cloud Storage + 下り通信 | $0.5 |
| Cloud Run CPU / Firestore / Cloud Tasks | $0（無料枠） |
| **合計** | **約$38** |

L4 の単価は GPU 単体ではなく、必須の 4vCPU / 16GiB を含めて
**$0.000323/秒（＝$1.16/時）**。1生成120秒として $0.039、
コールドスタート込みで $0.07 前後を見込んでいる。
