# Docker での実行

Cloud Run(GPU) への移植に向けた第1段階。**まずローカルのGPUでコンテナが動くことを確認する。**

Cloud Run にいきなり上げると、ビルド失敗の原因がコンテナなのか Cloud Run の設定なのか
切り分けられない。ローカルで通してから上げる。

## 前提

- NVIDIA ドライバと Docker、[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) が入っていること
- `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` が通ること
- 重みをダウンロード済みであること（`~/.cache/huggingface` に約20GB）
  - まだなら、ホスト側で一度 `bash measure.sh` を回すのが手っ取り早い

## 1. ビルド

```bash
docker build -t hunyuan3d:local .
```

初回は torch と CUDA拡張のビルドで時間がかかる。

**`TORCH_CUDA_ARCH_LIST` に注意。** ビルド時にGPUは見えないので、Dockerfile で
対象アーキテクチャを明示している。既定は `8.9`（Ada Lovelace = RTX 4070 / L4）。
別世代のGPUを使うなら Dockerfile を変える。

| GPU | 値 |
|---|---|
| RTX 40シリーズ / L4 | 8.9 |
| RTX 30シリーズ / A10G | 8.6 |
| A100 | 8.0 |
| H100 | 9.0 |

## 2. 実行

重みはイメージに入っていないので、ホストのキャッシュをマウントして渡す。

```bash
docker run --rm --gpus all \
  -v "$HOME/.cache/huggingface:/models" \
  -v "$(pwd)/output:/app/output" \
  hunyuan3d:local \
  python minimal_demo_mmgp.py \
    --input-image assets/example_images/052.png \
    --output ./output --texture --profile 3
```

`output/` に `textured_mesh.glb` が出れば成功。

### ホストでの実行と同じ結果になるか確かめる

`CLOUDRUN_MEASURE.md` の計測をコンテナ内で回せば、ホストとの差が分かる。

```bash
docker run --rm --gpus all \
  -v "$HOME/.cache/huggingface:/models" \
  -v "$(pwd)/output:/app/output" \
  -v "$(pwd)/measure_result.txt:/app/measure_result.txt" \
  hunyuan3d:local \
  bash measure.sh
```

コンテナ化による速度低下はほとんど無いはずだが、確認しておくと
Cloud Run での見積もりの土台が固くなる。

## 3. 詰まりやすいところ

| 症状 | 原因と対処 |
|---|---|
| `nvcc: not found` | ベースイメージが `runtime` になっている。`devel` を使う |
| `No CUDA runtime is found` （ビルド中） | `TORCH_CUDA_ARCH_LIST` の指定漏れ。GPUを検出しようとして失敗している |
| `undefined symbol` （実行時） | 拡張をビルドした torch と、実行時の torch が別物。イメージを作り直す |
| `libGL.so.1 が無い` | `libgl1` の入れ忘れ。Dockerfile に入れてある |
| 拡張のビルドで `ModuleNotFoundError: torch` | `--no-build-isolation` が抜けている。拡張は setup.py の中で torch を import するため必須 |
| 拡張のビルドが `setup.py install is deprecated` で止まる | Dockerfile では `pip install .` を使っている。もし `setup.py install` に戻すなら `pip install "setuptools<80"` で固定する |
| モデルのダウンロードが始まる | マウント先が違う。`HF_HOME=/models` なので `~/.cache/huggingface` を `/models` に渡す |

## 重みをイメージに入れていない理由

重みは約20GB（形状 7.2GB ＋ テクスチャ 12.7GB）ある。

Cloud Run のイメージストリーミングは **10GB 未満のモデルでは有効**だが、
それを超えると取り込みとストリーミングのオーバーヘッドがボトルネックになる。
20GB はその領域なので、イメージから出して実行時に与える。

副次的な利点として、**重みとコードのライフサイクルが分離される**。
コードを1行直すたびに20GBのイメージを焼き直す必要がなくなり、開発中の反復が速い。

## 次の段階（Cloud Run 向け）

このイメージは CLI を動かすところまで。Cloud Run に載せるには以下が要る。

1. **HTTPサーバ化** — `api_server.py` が既に `POST /send` → `GET /status/{uid}` を
   実装しているので、これを起点にする
2. **起動時に GCS から重みを落とす** — `gcloud storage cp --recursive` の並列ダウンロードが
   大きな重みでは最速。Cloud Storage FUSE は初回ダウンロードを並列化しないため遅い
3. **Cloud Run の設定** — L4 GPU、最低 4vCPU / 16GiB、同時実行数 1、スケールtoゼロ

順序としては、ローカルで 1 まで作って動作確認 → Cloud Run にデプロイして
コールドスタートを含む実測、が安全。
