# roomplanner-web から叩く API。仕様は SPEC.md。
#
# 段階1では認証コードを書かない。Cloud Run の IAM で閉じておけば
# 自分以外は叩けないので、認証の問題と配線の問題を混ぜずに確認できる。
# uid は固定値で、段階3で Firebase の匿名認証に差し替える。
#
# 状態は GCS のファイルだけで持つ。書き込みは1ジョブ数回、読みは
# ポーリングだけなので Firestore は要らない（SPEC.md 参照）。

import datetime
import io
import json
import os
import uuid

import firebase_admin
import google.auth
import google.auth.transport.requests
import pillow_heif
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from firebase_admin import auth as fb_auth
from google.cloud import storage
from PIL import Image, ImageOps, UnidentifiedImageError

# iPhone が既定で送る HEIC/HEIF を Pillow で開けるようにする。
# import しただけでは有効にならず、この登録が要る
pillow_heif.register_heif_opener()

# --- 設定 ---------------------------------------------------------------

OUTPUTS_BUCKET = os.environ["OUTPUTS_BUCKET"]
PROJECT_ID = os.environ["PROJECT_ID"]
REGION = os.environ["REGION"]
JOB_NAME = os.environ["JOB_NAME"]

# この時間を過ぎても状態が動かないジョブは、execution を見に行って
# 死んでいれば failed にする。OOM の signal 9 ではジョブ自身が
# 何も書けないため、これが無いと永遠に running のままになる
STALE_AFTER = datetime.timedelta(minutes=15)

# 限定公開中は許可リストに載っている uid だけを通す。
# 製品版では false にする。変わるのはこのフラグと Cloud Run の入口設定だけで、
# コードの作り直しは発生しない
ENFORCE_ALLOWLIST = os.environ.get("ENFORCE_ALLOWLIST", "true").lower() == "true"

DAILY_LIMIT = int(os.environ.get("DAILY_LIMIT", "10"))

# 許可リストは GCS に置く。uid を足すたびに再デプロイしないため。
# 毎リクエスト読むと待ち時間が延びるので短時間だけ覚えておく
ALLOWLIST_PATH = "config/allowed_uids.json"
ALLOWLIST_TTL = datetime.timedelta(seconds=60)
_allowlist_cache: tuple[datetime.datetime, set] | None = None

MAX_IMAGE_BYTES = 5 * 1024 * 1024

# Content-Type ではなく、実際にデコードできた形式で判定する。
# 拡張子や Content-Type は詐称できるうえ、実際に「PNG を名乗る壊れたファイル」が
# 検証をすり抜けて GPU を起動させ、10分後に失敗した実績がある（約39円の無駄）
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "HEIF", "HEIC", "MPO"}

# 生成パイプラインは長辺1024pxを前提にしている。クライアント側でも縮小するが、
# 携帯から直接上げられた場合に備えてサーバー側でも縮める
MAX_EDGE = 1024

# 署名付きURLの有効期限。長くするほど漏れたときの影響が伸びる
SIGNED_URL_TTL = datetime.timedelta(hours=1)

app = FastAPI()
_storage = storage.Client()

# ADC で初期化する。Cloud Run 上ではサービスアカウントの権限がそのまま使われる
firebase_admin.initialize_app()

# Cloud Run の管理APIを叩くための認証済みセッション。
# ジョブの起動と、死活の照会に使う
_credentials, _ = google.auth.default(
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)
_session = google.auth.transport.requests.AuthorizedSession(_credentials)

_JOB_BASE = (
    f"https://run.googleapis.com/v2/projects/{PROJECT_ID}"
    f"/locations/{REGION}/jobs/{JOB_NAME}"
)


def _start_job(job_id: str) -> str:
    """ジョブを起動し、execution 名を返す。

    どの入力を処理するかは環境変数の上書きで伝える。ジョブ側は
    JOB_ID からマウント済みの /jobs/{jobId}/ を見る。
    """
    res = _session.post(
        f"{_JOB_BASE}:run",
        json={
            "overrides": {
                "containerOverrides": [
                    {"env": [{"name": "JOB_ID", "value": job_id}]}
                ]
            }
        },
        timeout=30,
    )
    res.raise_for_status()
    # 返るのは長時間実行オペレーション。実際の execution 名はその中にある
    body = res.json()
    return body.get("metadata", {}).get("name") or body.get("name", "")


def _execution_is_dead(execution_name: str) -> bool:
    """execution が既に終わっているか。照会できないときは False（触らない）。"""
    if not execution_name:
        return False
    res = _session.get(f"https://run.googleapis.com/v2/{execution_name}", timeout=30)
    if res.status_code != 200:
        return False
    return bool(res.json().get("completionTime"))


def _now() -> str:
    """UTC の ISO8601。status.json のタイムスタンプはこの形で統一する。"""
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _bucket():
    return _storage.bucket(OUTPUTS_BUCKET)


def _status_blob(job_id: str):
    return _bucket().blob(f"jobs/{job_id}/status.json")


def _write_status(job_id: str, status: dict) -> None:
    status["updatedAt"] = _now()
    _status_blob(job_id).upload_from_string(
        json.dumps(status, ensure_ascii=False), content_type="application/json"
    )


def _reconcile_if_stale(status: dict) -> dict:
    """しばらく動きの無いジョブについて、実際に死んでいないか確かめる。

    ジョブ側は失敗時に自分で failed を書くが、OOM の signal 9 では
    何も書けずに死ぬ。その取りこぼしをここで拾う。
    """
    if status["state"] not in ("queued", "running"):
        return status

    updated = datetime.datetime.fromisoformat(status["updatedAt"])
    if datetime.datetime.now(datetime.timezone.utc) - updated < STALE_AFTER:
        return status

    if not _execution_is_dead(status.get("executionName")):
        return status

    status["state"] = "failed"
    status["error"] = "ジョブが状態を残さずに終了した（OOM や強制終了の可能性）"
    _write_status(status["jobId"], status)
    return status


def _signed_model_url(job_id: str) -> str:
    """成果物の署名付きURLを発行する。

    Cloud Run のサービスアカウントは秘密鍵を持たないため、
    generate_signed_url をそのまま呼ぶと
    「you need a private key to sign credentials」で失敗する。

    IAM の SignBlob API に署名を代行させることで回避する。そのために
    サービスアカウント自身に roles/iam.serviceAccountTokenCreator が要る
    （infra/api.tf で自分自身に付けている）。
    """
    # storage.Client() が持つ認証情報ではなく _credentials を使う。
    # 前者はストレージ用のスコープしか持たず、SignBlob を呼ぶと
    # 403 ACCESS_TOKEN_SCOPE_INSUFFICIENT になる。
    # _credentials は cloud-platform スコープで取得してある
    credentials = _credentials
    if not getattr(credentials, "service_account_email", None) or not credentials.token:
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
def _uid_from_token(authorization: str | None) -> str:
    """Authorization ヘッダの Firebase ID トークンを検証して uid を返す。"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization ヘッダが無い")
    try:
        decoded = fb_auth.verify_id_token(authorization[len("Bearer "):])
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"トークンが無効: {e}")
    return decoded["uid"]


def _allowed_uids() -> set:
    global _allowlist_cache
    now = datetime.datetime.now(datetime.timezone.utc)
    if _allowlist_cache and now - _allowlist_cache[0] < ALLOWLIST_TTL:
        return _allowlist_cache[1]
    blob = _bucket().blob(ALLOWLIST_PATH)
    uids = set(json.loads(blob.download_as_text())) if blob.exists() else set()
    _allowlist_cache = (now, uids)
    return uids


def _check_allowed(uid: str) -> None:
    if ENFORCE_ALLOWLIST and uid not in _allowed_uids():
        raise HTTPException(status_code=403, detail="このアカウントはまだ利用できない")


def _active_blob(uid: str):
    return _bucket().blob(f"quota/{uid}/active.json")


def _check_no_active_job(uid: str) -> None:
    """同じ利用者の並行実行を防ぐ。

    許すとGPUが同時に立ち上がり、事故的に高くつく。
    """
    blob = _active_blob(uid)
    if not blob.exists():
        return
    job_id = json.loads(blob.download_as_text()).get("jobId")
    st = _status_blob(job_id) if job_id else None
    if st and st.exists():
        state = json.loads(st.download_as_text())["state"]
        if state in ("queued", "running"):
            raise HTTPException(status_code=409, detail=f"生成中のジョブがある: {job_id}")
    # 終わっているので解放する
    blob.delete()


def _consume_quota(uid: str) -> None:
    """当日の実行回数を1増やす。上限に達していれば 429。

    generation の指定で競合を検出する。同じ利用者が同時に投げても
    二重に数え落とさない
    """
    today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    blob = _bucket().blob(f"quota/{uid}/{today}.json")
    if blob.exists():
        blob.reload()
        count = json.loads(blob.download_as_text()).get("count", 0)
        generation = blob.generation
    else:
        count, generation = 0, 0
    if count >= DAILY_LIMIT:
        raise HTTPException(
            status_code=429, detail=f"本日の上限（{DAILY_LIMIT}回）に達した"
        )
    try:
        blob.upload_from_string(
            json.dumps({"count": count + 1}),
            content_type="application/json",
            if_generation_match=generation,
        )
    except Exception:
        raise HTTPException(status_code=409, detail="同時に処理されたので、やり直してほしい")


@app.get("/health")
def health():
    return {"ok": True}


def _normalize_image(data: bytes) -> bytes:
    """受け取った画像を検証し、PNG に正規化して返す。

    ここで弾けなかった画像は GPU を10分間動かしたうえで失敗するので、
    受け付けの時点で確実に判定する。

    あわせて以下を吸収する。
      - iPhone の HEIC、Android の WebP
      - 写真の EXIF 回転（無視すると横倒しのまま生成される）
      - 携帯から直接上げられた大きすぎる画像
    """
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400, detail="画像として読み取れない。ファイルが壊れている可能性がある"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"画像の読み取りに失敗: {e}")

    if img.format not in ALLOWED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"対応していない形式: {img.format}。JPEG / PNG / WebP / HEIC が使える",
        )

    # 写真は EXIF に回転情報を持つ。これを反映しないと横倒しで生成される
    img = ImageOps.exif_transpose(img)

    # 透過を持つ画像はそのまま活かす（背景除去済みの素材が来ることがある）。
    # 持たないものは RGB に寄せる
    img = img.convert("RGBA" if img.mode in ("RGBA", "LA", "P") else "RGB")

    if max(img.size) > MAX_EDGE:
        img.thumbnail((MAX_EDGE, MAX_EDGE), Image.LANCZOS)

    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


@app.post("/jobs", status_code=202)
async def create_job(
    image: UploadFile = File(...), authorization: str | None = Header(default=None)
):
    uid = _uid_from_token(authorization)
    _check_allowed(uid)
    _check_no_active_job(uid)

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="画像が空")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"画像は {MAX_IMAGE_BYTES // (1024 * 1024)}MB まで。"
                   f"受け取ったのは {len(data) / (1024 * 1024):.1f}MB",
        )

    # GPU を起動する前にここで確実に弾く
    png = _normalize_image(data)

    # 画像が正当だと分かってから枠を消費する。壊れた画像で枠を失わせない
    _consume_quota(uid)

    job_id = f"job_{uuid.uuid4().hex[:16]}"

    _bucket().blob(f"jobs/{job_id}/input.png").upload_from_string(
        png, content_type="image/png"
    )

    now = _now()
    status = {
        "jobId": job_id,
        "uid": uid,
        "state": "queued",
        "createdAt": now,
        "updatedAt": now,
        "executionName": None,
        "error": None,
    }
    _write_status(job_id, status)

    try:
        execution_name = _start_job(job_id)
    except Exception as e:
        # 起動できなかったことを状態に残す。残さないと queued のまま放置される
        _write_status(job_id, {**status, "state": "failed", "error": f"ジョブの起動に失敗: {e}"})
        raise HTTPException(status_code=502, detail=f"ジョブの起動に失敗: {e}")

    _write_status(job_id, {**status, "executionName": execution_name})
    _active_blob(uid).upload_from_string(
        json.dumps({"jobId": job_id}), content_type="application/json"
    )
    return {"jobId": job_id}


@app.get("/jobs/{job_id}")
def get_job(job_id: str, authorization: str | None = Header(default=None)):
    uid = _uid_from_token(authorization)
    blob = _status_blob(job_id)
    if not blob.exists():
        raise HTTPException(status_code=404, detail="そのジョブは存在しない")

    status = json.loads(blob.download_as_text())

    # 他人のジョブは存在自体を隠す（403 ではなく 404）
    if status.get("uid") != uid:
        raise HTTPException(status_code=404, detail="そのジョブは存在しない")

    status = _reconcile_if_stale(status)

    body = {
        "state": status["state"],
        "createdAt": status["createdAt"],
    }
    if status["state"] == "succeeded":
        body["modelUrl"] = _signed_model_url(job_id)
    if status.get("error"):
        body["error"] = status["error"]
    return body
