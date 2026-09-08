# 作業依頼: マルチステージビルドでイメージを10GB未満にする

このドキュメントは**別のセッションに実機で実行してもらう**ための作業指示。
GPUのある環境（RTX 4070 / CUDA 12.4 / NVIDIA Container Toolkit 導入済み）で動かす。

書いた側（このPRを作ったセッション）にはGPUが無く、**ビルドも実行も一度も試していない**。
以下は机上の設計であり、そのまま通る保証はない。**通らなかったという結果も成果**なので、
詰まった箇所と実際のエラーをPRに書き戻してほしい。

## 何のための作業か

`roomplanner-web` の家具生成を、外部APIではなく自前のGPUコンテナで動かす計画の一部
（背景は `CLOUDRUN_MEASURE.md`）。

現在のイメージは **14.6GB**（モデルの重み20GBは既に外に出してある）。
Cloud Run のイメージストリーミングは **10GB未満のイメージでのみ有効**で、
超えると取り込みのオーバーヘッドがボトルネックになる。
つまり今のままでは、重みを外に出した効果が薄れる。

14.6GB の大半は CUDA の **devel** イメージ（ツールキット一式）と torch。
`nvcc` が要るのは CUDA拡張をビルドする一瞬だけなので、
**ビルドを devel で行い、成果物だけ runtime イメージへ移す**のがこの作業。

## ゴール

| | 現状 | 目標 |
|---|---|---|
| イメージサイズ（重みを除く） | 14.6GB | **10GB未満** |
| テクスチャ付き生成 | 成功 | 成功のまま |
| 形状生成 / 面数削減 / テクスチャ生成の各時間 | 14.6 / 7.0 / 45.6 秒 | 変わらないこと |

**サイズだけ達成して生成が壊れているのは失敗。** 逆に、10GBを切れなくても
「どこまで削れて、残りの内訳が何か」が分かれば前進する。

## 前提の確認

作業前にこれが通ること。

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
ls ~/.cache/huggingface/hub   # 重み約20GB がダウンロード済みであること
```

重みが無ければ、先にホスト側で一度 `bash measure.sh` を回すのが早い。

## 手順

### 0. 現状を測っておく（比較の基準になる）

```bash
docker build -t hunyuan3d:base -f Dockerfile .
docker images hunyuan3d:base
docker history hunyuan3d:base --human --format '{{.Size}}\t{{.CreatedBy}}' | head -30
```

`docker history` の出力を残しておくこと。**どの層が重いのかが、この作業の一番の情報**。
14.6GB の内訳が「devel イメージ 6.5GB / torch 4GB / その他」なのか、
別の配分なのかで、この後にやることが変わる。

### 1. マルチステージ版をビルドする

`Dockerfile.multistage` がこのPRに入っている。元の `Dockerfile` との差は2点だけ。

1. **ステージを分けた** — `devel` でビルドし、`/opt/venv` だけを `runtime` へコピーする
2. **torchaudio を入れるのをやめた** — リポジトリ全体で import している箇所が無い
   （`grep -rn torchaudio --include=*.py .` が空）

```bash
docker build -t hunyuan3d:multistage -f Dockerfile.multistage .
docker images | grep hunyuan3d
```

**ここで失敗する可能性が高い。** 想定される失敗と対処は下の「詰まりそうなところ」に書いた。

### 2. 動くことを確認する（サイズより先にこちらを見る）

```bash
docker run --rm --gpus all \
  -v "$HOME/.cache/huggingface:/models" \
  -v "$(pwd)/output:/app/output" \
  hunyuan3d:multistage \
  python minimal_demo_mmgp.py \
    --input-image assets/example_images/052.png \
    --output ./output --texture --profile 3
```

確認すること:

- `output/textured_mesh.glb` が出ること
- **`white_mesh.glb` ではないこと** — 過去に、テクスチャ生成が静かに失敗して
  形状だけのモデルを出しながら「成功」と報告していたことがある
- ログに出る各工程の時間が、上のゴール表の値と大きく変わっていないこと

### 3. サイズを比べる

```bash
docker images | grep hunyuan3d
docker history hunyuan3d:multistage --human --format '{{.Size}}\t{{.CreatedBy}}' | head -30
```

### 4. 10GBを切れなかった場合の追加の削り代

**ステージ分割だけでは 10GB を切らない可能性がある。**
devel → runtime で落ちるのは概ね 4GB 程度で、14.6GB からだと 10.6GB 前後に着地する
計算になる。足りなければ以下を順に足す。**一度に1つずつ変えて、毎回サイズを測ること。**

| 削るもの | 根拠 | やり方 |
|---|---|---|
| `gradio` / `gradio_litmodel3d` | `gradio` を import しているのは `gradio_app.py` だけ。`gradio_litmodel3d` はどこからも import されていない。この先使うのは `api_server.py`（fastapi + uvicorn）なので不要 | `requirements.txt` から2行消す。**`gradio_app.py` は動かなくなる**ので、消すなら同時にその旨を README に書く |
| `assets/report` `assets/images` | READMEに貼る画像で、実行には使わない。計 51MB | `.dockerignore` に足す。効果は小さいので優先度は低い |
| pip の `nvidia-*` パッケージ | torch の cu124 wheel は CUDA のライブラリ一式を site-packages に同梱する。runtime イメージが持つものと重複している | **手を出さないほうがよい**。torch はバンドル版を前提にロードするので、消すと `undefined symbol` で壊れる |

さらに踏み込むなら、runtime ではなく `nvidia/cuda:12.4.1-base-ubuntu22.04` を土台にする手もある
（torch は自前の CUDA ライブラリを持っているので、理屈のうえでは動く）。
ただし `custom_rasterizer` が何にリンクしているか次第なので、**上の手を尽くしてから**試すこと。

### 5. 通ったら差し替える

マルチステージ版で生成が成功し、サイズが縮んだら:

- `Dockerfile.multistage` の内容で `Dockerfile` を置き換え、`Dockerfile.multistage` は消す
  （2つ残すと、どちらが正なのか分からなくなる）
- `DOCKER.md` の「⚠️ イメージサイズ 14.6GB」の節と「次の段階」の0番を、実測値で書き換える
- `CLOUDRUN_MEASURE.md` の「実測から分かったこと ④」と月額見積もりの
  Artifact Registry の行（現在 14.6GB / $1.5）を書き換える
- `MULTISTAGE_TASK.md`（このファイル）は消す

## 詰まりそうなところ

既知の落とし穴は `DOCKER.md` の表にまとまっている（4件とも実機で踏んで潰したもの）。
そちらを先に読むこと。以下は**マルチステージ化で新たに出そうなもの**。

| 症状 | 見立てと対処 |
|---|---|
| builder で `ensurepip is not available` | `python3.10-venv` の入れ忘れ。Dockerfile.multistage には入れてある |
| runtime で `python: command not found` / venv が動かない | `/opt/venv/bin/python` は `/usr/bin/python3.10` へのシンボリックリンク。runtime 側に同じパスで python3.10 が要る。入れてあるが、パスがずれていないか `docker run --rm hunyuan3d:multistage python -V` で確認 |
| 実行時に `libXXX.so.N: cannot open shared object file` | **これが本命の失敗**。devel には入っていて runtime には無いライブラリを、拡張がリンクしている。`ldd /opt/venv/lib/python3.10/site-packages/custom_rasterizer_kernel*.so` を両方のステージで実行して差分を見る。足りないものを runtime 側の apt に足す |
| 実行時に `undefined symbol` | 拡張をビルドした torch と実行時の torch が別物になっている。venv を丸ごとコピーしていれば起きないはずなので、起きたらコピー範囲を疑う |
| `nvcc: not found`（実行時） | 実行時に nvcc を要求する処理が残っている。その場合はマルチステージ化そのものを見直す必要があるので、**どの処理が要求しているかを特定して報告してほしい** |
| サイズがほとんど変わらない | `docker images` が見ているのは最終ステージだけのはず。中間イメージを数えていないか確認する。それでも変わらないなら `docker history` の内訳を報告してほしい |

## 報告してほしいこと

このPRにコメントで書き戻す。

1. **ビルドが通ったか** — 通らなければ、失敗したコマンドとエラーの全文
2. **`docker images` の結果** — before / after のサイズ
3. **`docker history` の上位10層** — before / after
4. **生成が成功したか** — `textured_mesh.glb` が出たか、各工程の所要時間
5. **10GBを切れたか** — 切れなければ、何を追加で削って、どこで止まったか
6. **Dockerfile を直したなら、その内容と理由**

ここに書いた設計が間違っていた場合は、**遠慮なくそう書いてほしい**。
この指示は実機で確認していない机上の案なので、実機の結果のほうが常に正しい。
