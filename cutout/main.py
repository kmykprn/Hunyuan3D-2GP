# 写真から家具だけを切り抜く API。roomplanner-web から叩く。仕様は SPEC.md。
#
# 3D 生成（api/）とは別のサービスにしてある。あちらは GPU ジョブを起動する
# 薄い受付で、こちらは CPU で数秒の推論を自分で行う。メモリの要件が
# 512Mi と 12Gi で桁違いなので、同じコンテナに同居させると受付側まで
# 太くなる。3D をやめることになっても、こちらは単独で残せる。
#
# 状態は 1 日の回数だけ。GCS のファイルで数える（api/main.py と同じ作り）。

import datetime
import io
import json
import os

import firebase_admin
import numpy as np
import onnxruntime as ort
import pillow_heif
from fastapi import FastAPI, File, Header, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from firebase_admin import auth as fb_auth
from google.api_core.exceptions import PreconditionFailed
from google.cloud import storage
from PIL import Image, ImageOps, UnidentifiedImageError

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
# BiRefNet-general-lite（MIT）。入力は 1024×1024 固定
MODEL_PATH = os.environ.get("MODEL_PATH", "/models/birefnet-general-lite.onnx")
MODEL_INPUT_SIZE = 1024
# ImageNet の平均と分散。学習時と同じ値でないと精度が落ちる
MODEL_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
MODEL_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

MAX_IMAGE_BYTES = 5 * 1024 * 1024

# Content-Type ではなく、実際にデコードできた形式で判定する（api/main.py と同じ理由）
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "HEIF", "HEIC", "MPO"}

# 返す切り抜きの長辺。モデルの入力と同じにしておく。
# これより大きく返しても、輪郭の精度は 1024 で頭打ちになる
MAX_EDGE = 1024

# 切り抜きを内容の周りで切り詰めるときの余白（辺に対する比）。
# ぴったりに切ると、輪郭のぼかしが端で切れて縁が硬く見える
CROP_MARGIN = 0.02

# 「中身がある」とみなす不透明度。これ未満しかない行や列は切り落とす
CROP_ALPHA_THRESHOLD = 8

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


def _load_session() -> ort.InferenceSession:
    """モデルを読む。起動時に 1 度だけ。

    **メモリアリーナは切る。** 入れたままだと推論のたびにアリーナが伸び、
    3 回目には 11GB を超える（実測）。切れば 6.4GB で安定する。
    コンテナのメモリ上限（12Gi）はこの数字から決めてある
    """
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = False
    # 同時実行は 1（Cloud Run の設定）なので、CPU は全部この 1 件に使う
    options.intra_op_num_threads = os.cpu_count() or 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(MODEL_PATH, options, providers=["CPUExecutionProvider"])


_session = _load_session()


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


def _normalize(img: Image.Image) -> np.ndarray:
    """モデルの入力にする。1024×1024 に伸ばし、ImageNet の平均と分散で正規化する。

    縦横比は保たない（モデルが正方形固定）。出力側で元の大きさに戻すので歪みは残らない
    """
    resized = img.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.LANCZOS)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = (array - MODEL_MEAN) / MODEL_STD
    return array.transpose(2, 0, 1)[None]


def _predict_mask(img: Image.Image) -> Image.Image:
    """不透明度のマスク（元画像と同じ大きさ、L モード）を返す。"""
    input_name = _session.get_inputs()[0].name
    logits = _session.run(None, {input_name: _normalize(img)})[0][0, 0]
    # sigmoid で 0〜1 にしたあと、最小と最大で引き伸ばす。
    # 引き伸ばさないと、輪郭がはっきりした写真でも背景がうっすら残る
    pred = 1.0 / (1.0 + np.exp(-logits))
    low, high = pred.min(), pred.max()
    if high > low:
        pred = (pred - low) / (high - low)
    mask = Image.fromarray((pred * 255).astype(np.uint8), mode="L")
    return mask.resize(img.size, Image.LANCZOS)


def _crop_to_content(rgba: Image.Image) -> Image.Image:
    """透明な余白を切り落とす。少しだけ余白を残す。

    アプリ側はこの画像を板に貼るので、余白が多いと板の大きさと物の大きさが
    ずれ、ドラッグで掴める範囲が物からはみ出す。何も残らなければそのまま返す
    """
    alpha = np.asarray(rgba.getchannel("A"))
    rows = np.where(alpha.max(axis=1) >= CROP_ALPHA_THRESHOLD)[0]
    cols = np.where(alpha.max(axis=0) >= CROP_ALPHA_THRESHOLD)[0]
    if len(rows) == 0 or len(cols) == 0:
        return rgba
    top, bottom, left, right = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    margin_y = int((bottom - top) * CROP_MARGIN)
    margin_x = int((right - left) * CROP_MARGIN)
    return rgba.crop((
        max(0, left - margin_x),
        max(0, top - margin_y),
        min(rgba.width, right + margin_x),
        min(rgba.height, bottom + margin_y),
    ))


def _cut_out(img: Image.Image) -> bytes:
    """切り抜いた透過 PNG を返す。"""
    rgba = img.convert("RGBA")
    rgba.putalpha(_predict_mask(img))
    out = io.BytesIO()
    _crop_to_content(rgba).save(out, format="PNG")
    return out.getvalue()


# --- 入口 ---------------------------------------------------------------


# パスが /health なのは、Cloud Run では /healthz が使えないため（api/main.py 参照）
@app.get("/health")
def health():
    return {"ok": True}


@app.post("/cutouts")
async def create_cutout(
    image: UploadFile = File(...), authorization: str | None = Header(default=None)
):
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
    try:
        png = _cut_out(img)
    except Exception:
        # こちら都合の失敗は回数に数えない
        _restore_quota(uid)
        raise
    return Response(content=png, media_type="image/png")
