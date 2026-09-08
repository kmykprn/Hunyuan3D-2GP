# Hunyuan3D-2GP を Cloud Run(GPU) で動かすためのイメージ（マルチステージ）。
#
# 構成:
#   builder（devel）で CUDA拡張をビルドし、出来上がった venv だけを
#   runtime へ移す。nvcc が要るのはビルドの一瞬だけなので、
#   CUDAツールキット一式を最終イメージに持ち込まない。
#
# 実測: 14.6GB → 9.47GB（-35%）。生成結果・各工程の所要時間は単一ステージ版と同じ。
#   内訳の変化は CUDAベース層 4.98GB(devel) → 2.05GB(runtime) が主。
#
# なぜ小さくするのか（※コールドスタートのためではない）:
#   - Artifact Registry の保管費が減る（$0.10/GB/月）
#   - ビルドと push が速くなる
#   - runtime にコンパイラを置かないので、攻撃面が小さい
#
#   Cloud Run のイメージストリーミングは「起動に必要なブロックだけ」を取るため、
#   イメージが大きいこと自体は起動時間にほとんど効かない。
#   公式ガイドが挙げる 10GB の閾値は「モデルの重みをイメージに焼き込んでよいか」
#   の基準であって、イメージ全体のサイズの基準ではない。
#   https://docs.cloud.google.com/run/docs/configuring/services/gpu-best-practices
#
# モデルの重み 20GB はこのイメージに入っていない。実行時に外から与える（DOCKER.md 参照）。

# ══════════════════════════════════════════════════════
# builder ── nvcc が要る作業をここで全部やる
# ══════════════════════════════════════════════════════
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# python3.10-venv は Ubuntu が ensurepip を別パッケージに切り出しているため必要。
# 無いと `python -m venv` が「ensurepip is not available」で失敗する
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3.10-dev python3.10-venv python3-pip \
      build-essential git \
    && rm -rf /var/lib/apt/lists/*

# 成果物を1本のディレクトリにまとめる。こうしないと pip の入れた物が
# /usr/lib/python3/dist-packages と /usr/local/lib/... に散らばり、
# 次のステージへ「必要な物だけ」を移せない
RUN python3.10 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

RUN pip install --upgrade pip setuptools wheel

WORKDIR /app

# torchaudio は入れない。リポジトリ全体で import している箇所がひとつも無く、
# CUDA カーネルを抱えたぶんだけイメージが太る。
RUN pip install torch==2.5.1 torchvision \
      --index-url https://download.pytorch.org/whl/test/cu124

# --- CUDA拡張のビルド設定 ---
# devel イメージは nvcc を PATH に置くが CUDA_HOME は空のままで、
# 指定しないと cuda_runtime.h が見つからず落ちる
ENV CUDA_HOME=/usr/local/cuda
# ビルド中はGPUが見えず torch.cuda.is_available() が False になるため、
# 放っておくと diso が CPU版としてビルドされて失敗する
ENV FORCE_CUDA=1
# 8.9 = Ada Lovelace。RTX 4070（ローカル）と L4（Cloud Run）の両方が該当する
ENV TORCH_CUDA_ARCH_LIST="8.9"

COPY requirements.txt .

# diso は setup.py の中で torch を import するのに build-system.requires に
# 宣言していないため、PEP 517 の分離環境からは torch が見えない。
# diso だけ分けて --no-build-isolation で入れる
RUN grep -v '^diso$' requirements.txt > /tmp/req.txt \
 && pip install -r /tmp/req.txt \
 && pip install --no-build-isolation diso

# 拡張のソースだけ先に入れてビルドする。
# こうしておくと、アプリのコードを直してもこの重い層が再利用される
COPY hy3dgen/texgen/custom_rasterizer       hy3dgen/texgen/custom_rasterizer
COPY hy3dgen/texgen/differentiable_renderer hy3dgen/texgen/differentiable_renderer

# --no-build-isolation が必須（拡張は setup.py の中で torch を import する）
RUN pip install --no-build-isolation ./hy3dgen/texgen/custom_rasterizer \
 && pip install --no-build-isolation ./hy3dgen/texgen/differentiable_renderer

# ══════════════════════════════════════════════════════
# runtime ── nvcc もコンパイラも持たない、実行だけの層
# ══════════════════════════════════════════════════════
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# python3.10 は venv の実体（/opt/venv/bin/python は /usr/bin/python3.10 への
# シンボリックリンク）なので、runtime 側にも同じパスで必要。
# python3.10-dev / build-essential / pip はビルド専用なので入れない。
#
# libgl1 / libglib2.0-0 は opencv、libopengl0 は pymeshlab のプラグインが必要とする。
# libopengl0 が欠けると、面数削減の段で `Unknown format for load: ply` として
# 表面化する（形状生成までは成功して見えるので原因が遠い）
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 \
      libgl1 libopengl0 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# builder の成果物はこの1本だけ。CUDAツールキットもコンパイラも持ってこない
COPY --from=builder /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

WORKDIR /app
COPY . .

# 重みの置き場所。ローカルではホストの HuggingFace キャッシュを、
# Cloud Run では GCS から落としたディレクトリをここにマウントする
ENV HF_HOME=/models

CMD ["python", "minimal_demo_mmgp.py"]
