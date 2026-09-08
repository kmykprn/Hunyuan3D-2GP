# Hunyuan3D-2GP を Cloud Run(GPU) で動かすためのイメージ。
#
# 設計の要点:
#   - CUDA拡張のビルドに nvcc が要るため devel イメージを使う（runtime では nvcc が無い）
#   - モデルの重み 20GB はイメージに入れない。10GB を超えると Cloud Run の
#     イメージストリーミングがかえって遅くなるため、実行時に外から与える
#   - torch を requirements.txt より先に固定版で入れる。requirements.txt の
#     torch はバージョン未指定なので、後から入れると別のCUDA版が来て拡張が壊れる

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Ubuntu 22.04 の標準 Python は 3.10 系で、README が指定する 3.10.9 と同系列。
#
# libgl1 / libglib2.0-0 は opencv が必要とする。
# libopengl0（libOpenGL.so.0）は pymeshlab のプラグインが必要とする。
# libgl1 が入れる libGL.so.1 とは別物で、欠けていると libio_base.so の
# ロードに失敗し、対応形式がひとつも登録されないまま
# `Unknown format for load: ply` として表面化する。
# 面数削減の処理で初めて踏むため、モデルのロードまでは成功して見える
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3.10-dev python3-pip \
      build-essential git \
      libgl1 libopengl0 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python \
    && python -m pip install --upgrade pip

WORKDIR /app

# README と同じ取得元・同じバージョンにそろえる。
# ローカルで動作確認できている組み合わせから離れないことを優先する
RUN pip install torch==2.5.1 torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/test/cu124

# --- CUDA拡張のビルド設定 ---
# diso と後段の custom_rasterizer は、どちらもここでの設定に従ってコンパイルされる。
#
# nvidia/cuda の devel イメージは nvcc を PATH に置くが CUDA_HOME は空のままで、
# setup.py 側はこれを見て include パスを組み立てるため、指定しないと
# `fatal error: cuda_runtime.h: No such file or directory` で落ちる。
ENV CUDA_HOME=/usr/local/cuda
# diso は `torch.cuda.is_available() and CUDA_HOME is not None` で CUDA版か
# CPU版かを決める。docker build 中はGPUが見えず前者が False になるため、
# 放っておくと CppExtension でビルドされ、ソースが include する
# cuda_runtime.h が見つからず落ちる。FORCE_CUDA はそのための脱出口。
ENV FORCE_CUDA=1
# ビルド時にGPUは見えないため、対象アーキテクチャを明示しないと
# setup.py が実機を検出しようとして失敗する。
# 8.9 = Ada Lovelace。RTX 4070（ローカル）と L4（Cloud Run）の両方が該当する。
# 別世代のGPUで動かすときはここを変える（例: A100 は 8.0、H100 は 9.0）
ENV TORCH_CUDA_ARCH_LIST="8.9"

# diso と CUDA拡張は --no-build-isolation でビルドするため、
# ビルド用のツールがイメージ側に存在している必要がある。
# requirements.txt の依存の副作用に任せると、上流の変更で静かに壊れる
RUN pip install setuptools wheel

COPY requirements.txt .

# diso は setup.py の中で torch を import するのに、それを
# build-system.requires に宣言していない。pip は PEP 517 の分離環境で
# wheel を作るので、torch を先に入れてあってもその環境からは見えず、
# 必ず ModuleNotFoundError: No module named 'torch' で落ちる。
# diso だけ分けて --no-build-isolation で入れる
RUN grep -v '^diso$' requirements.txt > /tmp/req.txt \
 && pip install -r /tmp/req.txt \
 && pip install --no-build-isolation diso

# 拡張のソースだけ先に入れてビルドする。
# こうしておくと、アプリのコードを直してもこの重い層が再利用される
COPY hy3dgen/texgen/custom_rasterizer       hy3dgen/texgen/custom_rasterizer
COPY hy3dgen/texgen/differentiable_renderer hy3dgen/texgen/differentiable_renderer

# README は `python setup.py install` と書いているが、setuptools 80 以降で
# このコマンドは削除されている。新規に入る setuptools では失敗しうるため
# 同等の `pip install .` を使う。
# --no-build-isolation が必須: これらの拡張は setup.py の中で torch を import する。
# 分離環境でビルドすると torch が無く、必ず失敗する。
RUN pip install --no-build-isolation ./hy3dgen/texgen/custom_rasterizer \
 && pip install --no-build-isolation ./hy3dgen/texgen/differentiable_renderer

COPY . .

# 重みの置き場所。ローカルでは既存の HuggingFace キャッシュを、
# Cloud Run では GCS から落としたディレクトリをここにマウントする。
# HF_HOME 配下の hub/ をそのまま使うので、~/.cache/huggingface を丸ごと渡せばよい
ENV HF_HOME=/models

# 既定は使い方の表示。実際の実行はコマンドを上書きして行う（DOCKER.md 参照）
CMD ["python", "minimal_demo_mmgp.py"]
