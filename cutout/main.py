# 写真から家具だけを切り抜く API。roomplanner-web から叩く。仕様は SPEC.md。
#
# 3D 生成（api/）とは別のサービスにしてある。あちらは GPU ジョブを起動する
# 薄い受付で、こちらは CPU で数秒の推論を自分で行う。メモリの要件が
# 512Mi と 12Gi で桁違いなので、同じコンテナに同居させると受付側まで
# 太くなる。3D をやめることになっても、こちらは単独で残せる。
#
# 状態は GCS のファイル。1 日の回数（quota/）と、預かった切り抜き（jobs/）。
#
# 切り抜きは「預けて、あとで取りに行く」形（/cutout-jobs）。受け付けたら入力を GCS に
# 置き、Cloud Tasks に「この 1 件を処理せよ」を積んで、すぐ受付番号を返す。処理は
# Cloud Tasks がこのサービス自身の /run を叩いて行う（別のリクエストなので CPU が付く）。
# 画面は受付番号で状態を見に来る。iPhone は PWA を裏に回すと数秒で通信を切るので、
# 1 本の接続で 30〜50 秒待たせる形（旧 /cutouts、消した）は、URL をコピーしに行く間に切れていた。

import asyncio
import collections
import datetime
import io
import json
import os
import threading
import time
import uuid

import firebase_admin
import pillow_heif
from fastapi import FastAPI, File, Header, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from firebase_admin import auth as fb_auth
from google.api_core.exceptions import NotFound, PreconditionFailed
from google.auth.transport import requests as google_requests
from google.cloud import storage, tasks_v2
from google.oauth2 import id_token as google_id_token
from PIL import Image, ImageOps, UnidentifiedImageError

from birefnet import crop_to_content, load_session, predict_mask, to_png

# iPhone が既定で送る HEIC/HEIF を Pillow で開けるようにする
pillow_heif.register_heif_opener()

# --- 設定 ---------------------------------------------------------------

# 1 日の回数を数えるファイルの置き場。生成物バケットとは分ける。
# このサービスが 3D 生成の状態やジョブの入力に触れる必要は無い
STATE_BUCKET = os.environ["STATE_BUCKET"]

# 1 人あたり 1 日の上限。1 枚 0.1 円程度なので、これは費用の歯止めというより
# 「叩き放題にしない」ため。将来ここが無料枠になり、超えたら課金する土台
DAILY_LIMIT = int(os.environ.get("DAILY_LIMIT", "50"))

ALLOWED_ORIGINS = [
    o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()
]

# 預かった切り抜きを処理する順番待ち（Cloud Tasks の queue、projects/…/queues/…）と、
# その処理がこのサービス自身を叩くときの URL・名乗るサービスアカウント。
# Terraform（infra/cutout.tf）が入れる。空なら /cutout-jobs は 503 で閉じる
TASK_QUEUE = os.environ.get("TASK_QUEUE", "")
SELF_URL = os.environ.get("SELF_URL", "").rstrip("/")
TASK_SERVICE_ACCOUNT = os.environ.get("TASK_SERVICE_ACCOUNT", "")

# 預かった入力の JPEG 品質。長辺 1024 に縮めてから置くので 200KB 前後
INPUT_JPEG_QUALITY = 92

# モデルの重み。Dockerfile がイメージに焼く（md5 で検証済み）。
# BiRefNet-general-lite（MIT）。前処理・後処理は birefnet.py
MODEL_PATH = os.environ.get("MODEL_PATH", "/models/birefnet-general-lite.onnx")

MAX_IMAGE_BYTES = 5 * 1024 * 1024

# 推論に使うスレッド数。Terraform が CPU 数と同じ値を入れる。
#
# os.cpu_count() に頼らない。Cloud Run の第 2 世代コンテナではホストの CPU 数
# （数十）が返り、4 vCPU の枠に対して過剰なスレッドを立てて逆に遅くなる
# （実測: 手元 5 秒の推論が 33 秒になった）
INFERENCE_THREADS = int(os.environ.get("INFERENCE_THREADS", "0")) or (os.cpu_count() or 1)

# Content-Type ではなく、実際にデコードできた形式で判定する（api/main.py と同じ理由）
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "HEIF", "HEIC", "MPO"}

# 返す切り抜きの長辺。モデルの入力と同じにしておく。
# これより大きく返しても、輪郭の精度は 1024 で頭打ちになる
MAX_EDGE = 1024

# 推論の直近の所要秒数。応答に「見込み」として載せ、画面の円の進み方に使ってもらう。
# 起動直後で実測が無いときの既定は、4 vCPU での実測に合わせてある
RECENT_INFERENCE_SECONDS: collections.deque[float] = collections.deque(maxlen=5)
DEFAULT_INFERENCE_SECONDS = 8.0

# 推論は同時に 1 つ（1 件で 6.4GB 使う。2 つ走るとメモリが足りない）。
# サービスの同時リクエスト数は 8 にしてあり、状態確認の GET は推論の合間に同じ台が返す。
# /run が重なったら 429 を返し、Cloud Tasks に少し後で再試行させる
_inference_slot = threading.Lock()

app = FastAPI()

# 許可するオリジンは列挙する。ワイルドカードにしない（api/main.py と同じ方針）
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
    allow_credentials=False,
    max_age=3600,
)

firebase_admin.initialize_app()
_storage = storage.Client()
_tasks = tasks_v2.CloudTasksClient() if TASK_QUEUE else None


_session = load_session(MODEL_PATH, INFERENCE_THREADS)


# --- 認証と回数 ---------------------------------------------------------


def _identity_from_token(authorization: str | None) -> tuple[str, str]:
    """Authorization ヘッダの Firebase ID トークンを検証して、uid とメールを返す。

    **Google に紐づいたアカウントだけを通す。** 匿名アカウントにはメールが無いので 401。
    Web の API キーは公開値で、匿名アカウントは curl だけで作れる。それを通すと
    アプリを触っていない人でも叩けてしまう。将来の課金の単位もこのメールになる
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization ヘッダが無い")
    try:
        decoded = fb_auth.verify_id_token(authorization[len("Bearer "):])
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"トークンが無効: {e}")
    email = decoded.get("email") if decoded.get("email_verified") else None
    if not email:
        raise HTTPException(status_code=401, detail="Google ログインが必要")
    return decoded["uid"], email.lower()


def _quota_blob(uid: str):
    """当日ぶんのカウンタ。日付が変わると別のファイルになり、自然に 0 から始まる"""
    today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    return _storage.bucket(STATE_BUCKET).blob(f"quota/{uid}/{today}.json")


def _read_quota(blob) -> tuple[int, int]:
    """当日の回数と generation を返す。まだ無ければ (0, 0)。

    generation は条件付き書き込みに渡す。0 は「まだ無いときだけ書く」の意味になる
    """
    if not blob.exists():
        return 0, 0
    blob.reload()
    try:
        return int(json.loads(blob.download_as_text()).get("count", 0)), blob.generation
    except (ValueError, TypeError):
        # 壊れていたら数え直す。多く数える側に倒れるので安全側
        return 0, blob.generation


def _consume_quota(uid: str) -> None:
    """当日の回数を 1 増やす。上限に達していれば 429。

    generation の指定で競合を検出する。同じ利用者が同時に投げても二重に数え落とさない
    """
    blob = _quota_blob(uid)
    count, generation = _read_quota(blob)
    if count >= DAILY_LIMIT:
        raise HTTPException(status_code=429, detail=f"本日の上限（{DAILY_LIMIT}回）に達した")
    try:
        blob.upload_from_string(
            json.dumps({"count": count + 1}),
            content_type="application/json",
            if_generation_match=generation,
        )
    except PreconditionFailed:
        raise HTTPException(status_code=409, detail="同時に処理されたので、やり直してほしい")


def _restore_quota(uid: str) -> None:
    """こちら都合で失敗した 1 回を回数に数えない。戻せなければ諦める（多く数える側に倒す）"""
    blob = _quota_blob(uid)
    count, generation = _read_quota(blob)
    if count <= 0:
        return
    try:
        blob.upload_from_string(
            json.dumps({"count": count - 1}),
            content_type="application/json",
            if_generation_match=generation,
        )
    except PreconditionFailed:
        pass


# --- 画像 ---------------------------------------------------------------


def _decode_image(data: bytes) -> Image.Image:
    """受け取った画像を検証して、向きを直した RGB にして返す。"""
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

    # 写真は EXIF に回転情報を持つ。これを反映しないと横倒しのまま切り抜かれる
    img = ImageOps.exif_transpose(img).convert("RGB")
    if max(img.size) > MAX_EDGE:
        img.thumbnail((MAX_EDGE, MAX_EDGE), Image.LANCZOS)
    return img


def _expected_inference_seconds() -> float:
    """次の推論にかかりそうな秒数。直近の平均。"""
    if not RECENT_INFERENCE_SECONDS:
        return DEFAULT_INFERENCE_SECONDS
    return sum(RECENT_INFERENCE_SECONDS) / len(RECENT_INFERENCE_SECONDS)


def _timed_predict_mask(img: Image.Image) -> Image.Image:
    """推論して、かかった秒数を覚える。"""
    started = time.perf_counter()
    mask = predict_mask(_session, img)
    RECENT_INFERENCE_SECONDS.append(time.perf_counter() - started)
    return mask


def _finish(img: Image.Image, mask: Image.Image) -> bytes:
    """マスクをアルファに入れ、余白を切り落として PNG にする。"""
    rgba = img.convert("RGBA")
    rgba.putalpha(mask)
    return to_png(crop_to_content(rgba))


# --- 入口 ---------------------------------------------------------------


# パスが /health なのは、Cloud Run では /healthz が使えないため（api/main.py 参照）
@app.get("/health")
def health():
    return {"ok": True}


# --- 預ける形（/cutout-jobs） ------------------------------------------------
#
# jobs/{uid}/{id}/input.jpg   受け付けた写真（長辺 1024 に縮め、向きを直した JPEG）
# jobs/{uid}/{id}/status.json {"phase": queued|running|done|failed, ...}
# jobs/{uid}/{id}/result.png  切り抜き（done のとき）
#
# uid をパスに含めるのは、状態を見に来た人が自分の分しか読めないようにするため
# （トークンの uid から組み立てる）。古いものはバケットのライフサイクルで消える（1 日）。


def _job_blob(uid: str, job_id: str, name: str):
    return _storage.bucket(STATE_BUCKET).blob(f"jobs/{uid}/{job_id}/{name}")


def _read_status(uid: str, job_id: str) -> dict | None:
    blob = _job_blob(uid, job_id, "status.json")
    try:
        return json.loads(blob.download_as_text())
    except NotFound:
        return None


def _write_status(uid: str, job_id: str, status: dict) -> None:
    _job_blob(uid, job_id, "status.json").upload_from_string(
        json.dumps(status), content_type="application/json"
    )


def _now() -> float:
    return time.time()


def _enqueue_run(uid: str, job_id: str) -> None:
    """Cloud Tasks に「この 1 件を処理せよ」を積む。

    処理はこのサービス自身の /run を、サービスアカウントの OIDC トークン付きで叩く。
    /run 側はそのトークンを検証して、Cloud Tasks 以外からの呼び出しを 403 で弾く
    """
    _tasks.create_task(
        parent=TASK_QUEUE,
        task={
            "http_request": {
                "http_method": tasks_v2.HttpMethod.POST,
                "url": f"{SELF_URL}/cutout-jobs/{uid}/{job_id}/run",
                "oidc_token": {
                    "service_account_email": TASK_SERVICE_ACCOUNT,
                    "audience": SELF_URL,
                },
            }
        },
    )


def _verify_task_token(authorization: str | None) -> None:
    """/run を叩いてきたのが Cloud Tasks（自分のサービスアカウント）か確かめる。"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=403, detail="内部の入口")
    try:
        claims = google_id_token.verify_oauth2_token(
            authorization[len("Bearer "):], google_requests.Request(), audience=SELF_URL
        )
    except Exception as e:
        raise HTTPException(status_code=403, detail=f"トークンが無効: {e}")
    if not claims.get("email_verified") or claims.get("email") != TASK_SERVICE_ACCOUNT:
        raise HTTPException(status_code=403, detail="内部の入口")


def _public_status(status: dict) -> dict:
    """画面に返す形。経過秒はこちらの時計で出す（端末の時計とずれても困らないように）。"""
    now = _now()
    phase = status["phase"]
    if phase == "running":
        elapsed = now - status.get("startedAt", now)
    elif phase == "queued":
        elapsed = now - status.get("createdAt", now)
    else:
        elapsed = 0.0
    return {
        "phase": phase,
        "expectedSeconds": status.get("expectedSeconds"),
        "elapsed": round(max(elapsed, 0.0), 1),
        "error": status.get("error"),
    }


@app.post("/cutout-jobs", status_code=202)
async def create_cutout_job(
    image: UploadFile = File(...),
    authorization: str | None = Header(default=None),
):
    """写真を預けて受付番号をもらう。切り抜きはあとで（/cutout-jobs/{id} を見に来る）。

    認証・画像の検証・回数は /cutouts と同じ。受け付けたら入力を GCS に置き、
    Cloud Tasks に処理を積んで 202 で返る。積めなければ回数を戻して 503
    """
    uid, _email = _identity_from_token(authorization)
    if not (_tasks and SELF_URL and TASK_SERVICE_ACCOUNT):
        raise HTTPException(status_code=503, detail="預かる仕組みが設定されていない")

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="画像が空")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"画像は {MAX_IMAGE_BYTES // (1024 * 1024)}MB まで。"
                   f"受け取ったのは {len(data) / (1024 * 1024):.1f}MB",
        )
    img = _decode_image(data)
    _consume_quota(uid)

    job_id = uuid.uuid4().hex
    expected = round(_expected_inference_seconds(), 1)
    try:
        # HEIC でも向きが違っても、処理側は JPEG を開くだけで済むようにしておく
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=INPUT_JPEG_QUALITY)
        _job_blob(uid, job_id, "input.jpg").upload_from_string(buf.getvalue(), content_type="image/jpeg")
        _write_status(uid, job_id, {"phase": "queued", "createdAt": _now(), "expectedSeconds": expected})
        _enqueue_run(uid, job_id)
    except Exception as e:
        _restore_quota(uid)
        raise HTTPException(status_code=503, detail=f"預かれなかった: {e}")
    return {"id": job_id, "expectedSeconds": expected}


@app.post("/cutout-jobs/{uid}/{job_id}/run")
async def run_cutout_job(
    uid: str,
    job_id: str,
    authorization: str | None = Header(default=None),
):
    """預かった 1 件を処理する。Cloud Tasks だけが叩く（OIDC トークンで確かめる）。

    Cloud Tasks は 2xx 以外を再試行する。推論そのものの失敗はやり直しても同じなので
    status を failed にして 200 で返す。処理中に落ちた（インスタンスが消えた等）なら
    再試行で来るので、done でなければやり直す
    """
    _verify_task_token(authorization)
    status = _read_status(uid, job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="預かりが無い")
    if status["phase"] in ("done", "failed"):
        return {"phase": status["phase"]}

    # 推論の枠が空いていなければ待たずに 429。Cloud Tasks が数秒後に持ってくる
    if not _inference_slot.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="別の切り抜きを処理中")
    try:
        started = _now()
        _write_status(uid, job_id, {**status, "phase": "running", "startedAt": started})
        try:
            img = Image.open(io.BytesIO(_job_blob(uid, job_id, "input.jpg").download_as_bytes()))
            img.load()
            img = img.convert("RGB")
            mask = await asyncio.to_thread(_timed_predict_mask, img)
            png = await asyncio.to_thread(_finish, img, mask)
            _job_blob(uid, job_id, "result.png").upload_from_string(png, content_type="image/png")
            _write_status(uid, job_id, {**status, "phase": "done", "startedAt": started, "finishedAt": _now()})
            return {"phase": "done"}
        except Exception as e:
            # こちら都合の失敗は回数に数えない
            _restore_quota(uid)
            _write_status(uid, job_id, {**status, "phase": "failed", "startedAt": started, "error": f"切り抜きに失敗: {e}"})
            return {"phase": "failed"}
    finally:
        _inference_slot.release()


# 状態確認で待ってもらえる上限（秒）。サービスの打ち切り 60 秒より十分短く
MAX_WAIT_SECONDS = 25.0
# 待っている間に状態を読み直す間隔（秒）。GCS の小さな読み取りなので費用は無視できる
WAIT_POLL_SECONDS = 1.0


@app.get("/cutout-jobs/{job_id}")
async def get_cutout_job(
    job_id: str,
    wait: float = 0,
    after: str | None = None,
    authorization: str | None = Header(default=None),
):
    """預けた 1 件の状態。自分の分しか見えない（uid はトークンから）。

    wait を付けると、工程が after から変わる（または done / failed になる）まで最大
    wait 秒（上限 25）サーバーで待ってから返す。画面が 2 秒ごとに叩く代わりに、
    「起動待ち → 推論中 → 完成」の変わり目ごとに 1 回で済む。
    待っている途中で画面が裏に回って接続が切れても、状態は GCS にあるので、
    戻ってきてもう一度叩けばよい
    """
    uid, _email = _identity_from_token(authorization)
    deadline = time.monotonic() + min(max(wait, 0.0), MAX_WAIT_SECONDS)
    while True:
        status = _read_status(uid, job_id)
        if status is None:
            raise HTTPException(status_code=404, detail="預かりが無い")
        phase = status["phase"]
        settled = phase in ("done", "failed")
        changed = after is not None and phase != after
        if settled or changed or time.monotonic() >= deadline:
            return _public_status(status)
        await asyncio.sleep(min(WAIT_POLL_SECONDS, max(deadline - time.monotonic(), 0.0)))


@app.get("/cutout-jobs/{job_id}/result")
def get_cutout_result(job_id: str, authorization: str | None = Header(default=None)):
    """できあがった切り抜き（透過 PNG）。done になるまでは 404。"""
    uid, _email = _identity_from_token(authorization)
    status = _read_status(uid, job_id)
    if status is None or status["phase"] != "done":
        raise HTTPException(status_code=404, detail="まだできていない")
    return Response(content=_job_blob(uid, job_id, "result.png").download_as_bytes(), media_type="image/png")
