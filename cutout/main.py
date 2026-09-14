# 写真から家具だけを切り抜く API。roomplanner-web から叩く。仕様は SPEC.md。
#
# 3D 生成（api/）とは別のサービスにしてある。あちらは GPU ジョブを起動する
# 薄い受付で、こちらは CPU で数秒の推論を自分で行う。メモリの要件が
# 512Mi と 12Gi で桁違いなので、同じコンテナに同居させると受付側まで
# 太くなる。3D をやめることになっても、こちらは単独で残せる。
#
# 状態は 1 日の回数だけ。GCS のファイルで数える（api/main.py と同じ作り）。

import asyncio
import base64
import collections
import datetime
import io
import json
import os
import time

import firebase_admin
import pillow_heif
from fastapi import FastAPI, File, Header, HTTPException, Response, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from firebase_admin import auth as fb_auth
from google.api_core.exceptions import PreconditionFailed
from google.cloud import storage
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

# 流している途中で黙る時間の上限。推論は 10〜20 秒黙るが、応答のヘッダを送ったあとに
# 無通信が続くと、途中の経路（iPhone では HTTP/3）が応答を閉じてしまうことがあった。
# その間も同じ工程の行を繰り返し流して、接続が生きていることを示す
HEARTBEAT_SECONDS = float(os.environ.get("HEARTBEAT_SECONDS", "2"))

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


def _event(phase: str, **fields) -> bytes:
    """応答の 1 行。NDJSON（1 行 1 JSON）。"""
    return (json.dumps({"phase": phase, **fields}) + "\n").encode()


async def _heartbeat_until(task: asyncio.Future, event):
    """task が終わるまで、HEARTBEAT_SECONDS ごとに event(経過秒) の行を流す。

    行を流すのは接続を黙らせないため。画面側は同じ工程の行が繰り返し来ても
    円の基準時刻を動かさない（経過秒は行の中に入れて渡す）。結果は呼び出し側が task から取る
    """
    started = time.perf_counter()
    while True:
        finished, _ = await asyncio.wait({task}, timeout=HEARTBEAT_SECONDS)
        if finished:
            return
        yield event(round(time.perf_counter() - started, 1))


# --- 入口 ---------------------------------------------------------------


# パスが /health なのは、Cloud Run では /healthz が使えないため（api/main.py 参照）
@app.get("/health")
def health():
    return {"ok": True}


@app.post("/cutouts")
async def create_cutout(
    image: UploadFile = File(...),
    authorization: str | None = Header(default=None),
    accept: str | None = Header(default=None),
):
    """切り抜く。Accept が application/x-ndjson なら工程を 1 行ずつ流し、そうでなければ PNG を返す（SPEC.md）。

    2 通りあるのは切り替えの順序のため。配信済みの画面は PNG を待っているので、
    先にサーバーを流す形だけにすると、画面を更新するまでの間ずっと失敗する。
    画面が NDJSON を読めるようになったら PNG の形は消してよい。

    推論そのものは途中経過を出せない（onnxruntime の中で止まる）ので、流せるのは
    「受け付けた → 推論中（見込み秒数つき）→ 仕上げ → 完成」の 4 つ。
    それでも、1 行目が届くまでが「起動待ち」だと画面側で区別できる。
    コールドスタートで 15 秒黙っている間、画面が嘘の円を出さずに済む。

    認証・画像の検証・回数は、ストリームを始める前に普通の HTTP エラーで返す。
    始めたあとの失敗は最後の行（failed）で伝える。ヘッダはもう送ってしまっているため
    """
    uid, _email = _identity_from_token(authorization)

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="画像が空")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"画像は {MAX_IMAGE_BYTES // (1024 * 1024)}MB まで。"
                   f"受け取ったのは {len(data) / (1024 * 1024):.1f}MB",
        )
    # 数える前に弾く。読めない画像で回数を減らさない
    img = _decode_image(data)

    # 推論の前に数える。後で数えると、同時に投げられた分が全部通ってしまう
    _consume_quota(uid)

    if "application/x-ndjson" not in (accept or ""):
        try:
            mask = await asyncio.to_thread(_timed_predict_mask, img)
            png = await asyncio.to_thread(_finish, img, mask)
        except Exception:
            # こちら都合の失敗は回数に数えない
            _restore_quota(uid)
            raise
        return Response(content=png, media_type="image/png")

    async def stream():
        try:
            yield _event("received")
            expected = round(_expected_inference_seconds(), 1)
            yield _event("cutting", expectedSeconds=expected, elapsed=0)
            # 推論は数秒ブロックする。イベントループを止めないよう別スレッドで走らせ、
            # その間は同じ行を繰り返し流す（黙ると接続を閉じられることがある）
            predicting = asyncio.ensure_future(asyncio.to_thread(_timed_predict_mask, img))
            async for line in _heartbeat_until(
                predicting, lambda s: _event("cutting", expectedSeconds=expected, elapsed=s)
            ):
                yield line
            mask = predicting.result()
            yield _event("finishing", elapsed=0)
            finishing = asyncio.ensure_future(asyncio.to_thread(_finish, img, mask))
            async for line in _heartbeat_until(finishing, lambda s: _event("finishing", elapsed=s)):
                yield line
            png = finishing.result()
            yield _event("done", png=base64.b64encode(png).decode())
        except Exception as e:
            # こちら都合の失敗は回数に数えない
            _restore_quota(uid)
            yield _event("failed", error=f"切り抜きに失敗: {e}")

    return StreamingResponse(stream(), media_type="application/x-ndjson")
