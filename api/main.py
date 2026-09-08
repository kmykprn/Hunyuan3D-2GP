# roomplanner-web から叩く API。仕様は SPEC.md。
#
# 段階1では認証コードを書かない。Cloud Run の IAM で閉じておけば
# 自分以外は叩けないので、認証の問題と配線の問題を混ぜずに確認できる。
# uid は固定値で、段階3で Firebase の匿名認証に差し替える。
#
# 状態は GCS のファイルだけで持つ。書き込みは1ジョブ数回、読みは
# ポーリングだけなので Firestore は要らない（SPEC.md 参照）。

import datetime
import json
import os
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from google.cloud import storage

# --- 設定 ---------------------------------------------------------------

OUTPUTS_BUCKET = os.environ["OUTPUTS_BUCKET"]

# 段階3で Firebase のトークンから取る。それまでは固定値
STAGE1_FIXED_UID = "local-test"

MAX_IMAGE_BYTES = 5 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg"}

# 署名付きURLの有効期限。長くするほど漏れたときの影響が伸びる
SIGNED_URL_TTL = datetime.timedelta(hours=1)

app = FastAPI()
_storage = storage.Client()


def _now() -> str:
    """UTC の ISO8601。status.json のタイムスタンプはこの形で統一する。"""
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _bucket():
    return _storage.bucket(OUTPUTS_BUCKET)


def _status_blob(job_id: str):
    return _bucket().blob(f"jobs/{job_id}/status.json")


def _signed_model_url(job_id: str) -> str:
    """成果物の署名付きURLを発行する。

    Cloud Run のサービスアカウントは秘密鍵を持たないため、
    generate_signed_url をそのまま呼ぶと
    「you need a private key to sign credentials」で失敗する。

    IAM の SignBlob API に署名を代行させることで回避する。そのために
    サービスアカウント自身に roles/iam.serviceAccountTokenCreator が要る
    （infra/api.tf で自分自身に付けている）。
    """
    credentials = _storage._credentials
    # メタデータサーバー由来の認証情報は service_account_email を
    # 遅延で解決するため、先に一度リフレッシュしておく
    if not getattr(credentials, "service_account_email", None):
        import google.auth.transport.requests

        credentials.refresh(google.auth.transport.requests.Request())

    return _bucket().blob(f"jobs/{job_id}/model.glb").generate_signed_url(
        version="v4",
        expiration=SIGNED_URL_TTL,
        method="GET",
        service_account_email=credentials.service_account_email,
        access_token=credentials.token,
    )


# パスが /health なのは、Cloud Run では /healthz が使えないため。
# Google Frontend が /healthz を横取りし、アプリに到達する前に
# HTML の404を返す（アプリ側で登録しても届かない）。
# SPEC.md は /healthz と書いているが、この環境では実現できない。
@app.get("/health")
def health():
    return {"ok": True}


@app.post("/jobs", status_code=202)
async def create_job(image: UploadFile = File(...)):
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"画像は PNG か JPEG のみ。受け取ったのは {image.content_type}",
        )

    data = await image.read()
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"画像は {MAX_IMAGE_BYTES // (1024 * 1024)}MB まで。受け取ったのは {len(data)} バイト",
        )
    if not data:
        raise HTTPException(status_code=400, detail="画像が空")

    job_id = f"job_{uuid.uuid4().hex[:16]}"
    uid = STAGE1_FIXED_UID

    _bucket().blob(f"jobs/{job_id}/input.png").upload_from_string(
        data, content_type=image.content_type
    )

    now = _now()
    _status_blob(job_id).upload_from_string(
        json.dumps(
            {
                "jobId": job_id,
                "uid": uid,
                "state": "queued",
                "createdAt": now,
                "updatedAt": now,
                "executionName": None,
                "error": None,
            },
            ensure_ascii=False,
        ),
        content_type="application/json",
    )

    # 段階2でここからジョブを起動する
    return {"jobId": job_id}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    blob = _status_blob(job_id)
    if not blob.exists():
        raise HTTPException(status_code=404, detail="そのジョブは存在しない")

    status = json.loads(blob.download_as_text())

    # 他人のジョブは存在自体を隠す（403 ではなく 404）
    if status.get("uid") != STAGE1_FIXED_UID:
        raise HTTPException(status_code=404, detail="そのジョブは存在しない")

    # 段階2で、updatedAt が古いジョブの execution 状態を照会して
    # 死んでいれば failed にする処理をここに足す

    body = {
        "state": status["state"],
        "createdAt": status["createdAt"],
    }
    if status["state"] == "succeeded":
        body["modelUrl"] = _signed_model_url(job_id)
    if status.get("error"):
        body["error"] = status["error"]
    return body
