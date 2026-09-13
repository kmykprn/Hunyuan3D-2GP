# 切り抜きサービスのイメージ。CPU だけで動く。
#
# モデルの重み（224MB）はここで焼き込む。起動のたびに取りに行くと、
# コールドスタートが GitHub の配信速度に左右される。
# md5 は rembg が配布に付けている値で、差し替えられたイメージが
# 通らないようにするための検証
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# BiRefNet-general-lite（MIT）。rembg の配布物をそのまま使う
ARG MODEL_URL=https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx
ARG MODEL_MD5=4fab47adc4ff364be1713e97b7e66334
RUN mkdir -p /models \
    && curl -fsSL "$MODEL_URL" -o /models/birefnet-general-lite.onnx \
    && echo "$MODEL_MD5  /models/birefnet-general-lite.onnx" | md5sum -c -

COPY main.py birefnet.py .

# Cloud Run は $PORT で待ち受けることを要求する
ENV PORT=8080
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT}
