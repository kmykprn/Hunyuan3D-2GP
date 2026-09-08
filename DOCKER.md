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
| **diso** のビルドで `ModuleNotFoundError: No module named 'torch'` | diso は setup.py で torch を import するのに `build-system.requires` に宣言していない。torch を先に入れても分離環境からは見えないので、diso だけ分けて `--no-build-isolation` で入れる |
| `fatal error: cuda_runtime.h: No such file or directory` | `CUDA_HOME` が空。devel イメージは `nvcc` を PATH に置くが `CUDA_HOME` は設定しないため、明示が必要 |
| diso が CPU 版でビルドされ、`cuda_runtime.h` が見つからない | ビルド中はGPUが見えず `torch.cuda.is_available()` が False になり、CUDA版ではなくCPU版が選ばれる。`FORCE_CUDA=1` で回避する |
| **面数削減で `Unknown format for load: ply`** | `libopengl0` の欠落。`libgl1` が入れる `libGL.so.1` とは別物で、無いと pymeshlab のプラグインが全滅し対応形式がひとつも登録されない。**モデルのロードと形状生成は成功して見えるため原因が遠い** |
| 拡張のビルドが `setup.py install is deprecated` で止まる | Dockerfile では `pip install .` を使っている。もし `setup.py install` に戻すなら `pip install "setuptools<80"` で固定する |
| モデルのダウンロードが始まる | マウント先が違う。`HF_HOME=/models` なので `~/.cache/huggingface` を `/models` に渡す |
| builder で `ensurepip is not available` | `python3.10-venv` の入れ忘れ。venv を作るのに必要 |
| runtime で共有ライブラリが足りない | 実測では起きなかった。torch が `nvidia-*-cu12` を venv 内に同梱するため、CUDAランタイムは venv ごと運ばれる。もし起きたら両ステージで `ldd` を取って差分を見る |

## 重みをイメージに入れていない理由

重みは約20GB（形状 7.2GB ＋ テクスチャ 12.7GB）ある。

Cloud Run の公式ガイドは、**モデルの重みが 10GB 未満ならイメージに焼き込んでよい**が、
それを超えるものは Cloud Storage から `gcloud storage cp` で並列ダウンロードするのが
最速だとしている。20GB はその領域なので、イメージから出して実行時に与える。

> Google recommends downloading ML models from Cloud Storage (略), though you might
> alternatively store models inside container images if they're smaller than 10 GB.
> — [Best practices: AI inference on Cloud Run services with GPUs](https://docs.cloud.google.com/run/docs/configuring/services/gpu-best-practices)

**ただし、これはまだ検証していない前提**。イメージに焼き込む案（`COPY` するだけで
GCSバケットも権限も起動スクリプトも不要）のほうが、実装が小さいぶん先に試す価値がある。
どちらが速いかは Cloud Run 上で実測して決める。

## 実測（2026-09-08）

RTX 4070 12GB / profile 3 / `assets/example_images/052.png`

| 工程 | ホスト | コンテナ |
|---|---|---|
| 形状生成 | 14.5 秒 | 14.6 秒 |
| 面数削減 | 7.1 秒 | 7.0 秒 |
| テクスチャ生成 | 46.4 秒 | 45.6 秒 |
| 全体 | 98.5 秒 | 125.0 秒 |

各工程の所要時間はホストとほぼ一致しており、**コンテナ化による速度低下はない**。
全体の差（約26秒）はイメージの起動と重みの読み込み。

### イメージサイズ 9.47GB（重みを含まない）

マルチステージ化により **14.6GB → 9.47GB（-35%）**。生成結果と各工程の所要時間は
単一ステージ版と変わらない（交互に2回ずつ実行して確認）。

| 層 | before | after |
|---|---|---|
| CUDA ベース | 4.98GB（devel） | **2.05GB（runtime）** |
| torch 系 | 5.42GB | 6.87GB（venv に統合、torchaudio 除去済み） |
| requirements + diso | 1.45GB | （venv に含む） |
| apt | 298MB | 230MB |

**注意: これはコールドスタートのための最適化ではない。**
Cloud Run のイメージストリーミングは「起動に必要なブロックだけ」を取るため、
イメージが大きいこと自体は起動時間にほとんど効かない
（公式ブログは「15GB の CUDA イメージでも小さなアプリ並みに起動する」としている）。
削って得られるのは、Artifact Registry の保管費（$0.10/GB/月）、ビルドと push の速さ、
runtime にコンパイラを置かないことによる攻撃面の縮小。

**10GB という数字は「重みをイメージに焼き込んでよいか」の基準であって、
イメージ全体のサイズの基準ではない。** 上の「重みをイメージに入れていない理由」を参照。

### コンテナが root で動くこと

`USER` を切っていないため、バインドマウントした `output/` や `gradio_cache/` の
中身が root 所有になり、ホスト側から消せなくなる。当面はコンテナ経由で消すか、
`docker run --user "$(id -u):$(id -g)"` を付ける。
Cloud Run では実害がないので、`USER` を切るのは実機で確認できるときにまとめて行う。

## 次の段階（Cloud Run 向け）

このイメージは CLI を動かすところまで。Cloud Run に載せるには以下が要る。

1. **HTTPサーバ化** — `api_server.py` が既に `POST /send` → `GET /status/{uid}` を
   実装しているので、これを起点にする。ただし mmgp を通していない点を含め、
   そのままでは Cloud Run に載らない（`CLOUDRUN_MEASURE.md` の「次にやること」参照）
2. **重みの供給方法を決める** — イメージに焼き込む案と GCS から落とす案があり、
   どちらが速いかは未検証。実装が小さいのは焼き込む案
3. **Cloud Run の設定** — L4 GPU、最低 4vCPU / 16GiB、同時実行数 1、スケールtoゼロ。
   **L4 は東京(asia-northeast1)では使えない**。アジアは asia-southeast1（シンガポール）
   または asia-south1（ムンバイ）

順序としては、1 を待たずに **今のイメージをそのまま Cloud Run Job としてデプロイする**のが
最短。CLI が `CMD` になっているのでそのまま動き、コールドスタートと L4 での生成時間という
一番大きな2つの未知数が、HTTPサーバを書く前に確定する。
