# BiRefNet-general-lite（MIT）で背景を抜く、純粋な画像処理の部分。
#
# 切り抜きサービス（cutout/main.py）と、8 方向の画像を作るワーカー（multiview.py）の
# 両方から使う。ここには GCS も Firebase も無い。入れると import した側が繋ぎに行く

import io
import os

import numpy as np
import onnxruntime as ort
from PIL import Image

# 入力は 1024×1024 固定
MODEL_INPUT_SIZE = 1024
# ImageNet の平均と分散。学習時と同じ値でないと精度が落ちる
MODEL_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
MODEL_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# 切り抜きを内容の周りで切り詰めるときの余白（辺に対する比）。
# ぴったりに切ると、輪郭のぼかしが端で切れて縁が硬く見える
CROP_MARGIN = 0.02
# 「中身がある」とみなす不透明度。これ未満しかない行や列は切り落とす
CROP_ALPHA_THRESHOLD = 8


def load_session(model_path: str, threads: int) -> ort.InferenceSession:
    """モデルを読む。

    **メモリアリーナは切る。** 入れたままだと推論のたびにアリーナが伸び、
    3 回目には 11GB を超える（実測）。切れば 6.4GB で安定する
    """
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = False
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    print(f"inference threads={threads} (os.cpu_count={os.cpu_count()})", flush=True)
    return ort.InferenceSession(model_path, options, providers=["CPUExecutionProvider"])


def _normalize(img: Image.Image) -> np.ndarray:
    """モデルの入力にする。1024×1024 に伸ばし、ImageNet の平均と分散で正規化する。

    縦横比は保たない（モデルが正方形固定）。出力側で元の大きさに戻すので歪みは残らない
    """
    resized = img.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.LANCZOS)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = (array - MODEL_MEAN) / MODEL_STD
    return array.transpose(2, 0, 1)[None]


def predict_mask(session: ort.InferenceSession, img: Image.Image) -> Image.Image:
    """不透明度のマスク（元画像と同じ大きさ、L モード）を返す。img は RGB"""
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: _normalize(img)})[0][0, 0]
    # sigmoid で 0〜1 にしたあと、最小と最大で引き伸ばす。
    # 引き伸ばさないと、輪郭がはっきりした写真でも背景がうっすら残る
    pred = 1.0 / (1.0 + np.exp(-logits))
    low, high = pred.min(), pred.max()
    if high > low:
        pred = (pred - low) / (high - low)
    mask = Image.fromarray((pred * 255).astype(np.uint8), mode="L")
    return mask.resize(img.size, Image.LANCZOS)


def content_box(rgba: Image.Image) -> tuple[int, int, int, int] | None:
    """中身のある範囲（left, top, right, bottom）。何も無ければ None"""
    alpha = np.asarray(rgba.getchannel("A"))
    rows = np.where(alpha.max(axis=1) >= CROP_ALPHA_THRESHOLD)[0]
    cols = np.where(alpha.max(axis=0) >= CROP_ALPHA_THRESHOLD)[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def crop_with_margin(rgba: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    """範囲に少しだけ余白を足して切り詰める"""
    left, top, right, bottom = box
    margin_x = int((right - left) * CROP_MARGIN)
    margin_y = int((bottom - top) * CROP_MARGIN)
    return rgba.crop((
        max(0, left - margin_x),
        max(0, top - margin_y),
        min(rgba.width, right + margin_x),
        min(rgba.height, bottom + margin_y),
    ))


def crop_to_content(rgba: Image.Image) -> Image.Image:
    """透明な余白を切り落とす。何も残らなければそのまま返す。

    アプリ側はこの画像を板に貼るので、余白が多いと板の大きさと物の大きさが
    ずれ、ドラッグで掴める範囲が物からはみ出す
    """
    box = content_box(rgba)
    return crop_with_margin(rgba, box) if box else rgba


def matte(session: ort.InferenceSession, img: Image.Image) -> Image.Image:
    """RGB の画像に、推定した不透明度を付けて RGBA で返す（切り詰めはしない）"""
    rgba = img.convert("RGBA")
    rgba.putalpha(predict_mask(session, img.convert("RGB")))
    return rgba


def to_png(rgba: Image.Image) -> bytes:
    out = io.BytesIO()
    rgba.save(out, format="PNG")
    return out.getvalue()
