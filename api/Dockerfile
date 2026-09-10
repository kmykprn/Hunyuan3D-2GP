# API層のイメージ。GPU も CUDA も要らないので、生成側の
# Dockerfile（9.47GB）とは別に軽量なものを用意する。
#
# ここが重いとスケールアウトのたびにコールドスタートが伸びる。
# 生成そのものは Cloud Run Job 側でやるので、こちらは薄く保つ。
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY main.py .

# Cloud Run は $PORT で待ち受けることを要求する。
# 既定の 8080 を入れておき、環境変数があればそちらに従う
ENV PORT=8080
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT}
