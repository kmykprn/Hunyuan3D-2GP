# 楽天市場の商品ページの URL から、商品情報と画像を取る。
#
# スクレイピングはしない。楽天市場商品検索 API（公式）に itemCode で問い合わせる。
# 画像だけはブラウザから楽天の CDN を直接取れない（CORS）ので、ここで取って返す。
#
# ここには GCS も Firebase も無い。純粋な「URL → 商品情報」だけで、テストしやすくしてある。

import io
import os
import re
from urllib.parse import urlparse, urlencode, parse_qsl, urlunparse

import requests
from PIL import Image, UnidentifiedImageError

# 楽天市場商品検索 API。2026 年版から accessKey が要る
SEARCH_ENDPOINT = "https://openapi.rakuten.co.jp/ichibams/api/IchibaItem/Search/20260701"

# 楽天ウェブサービスのアプリ登録にある「許可するウェブサイト」に載せたサイト。
# 「一覧に無いところからの呼び出しは拒否する」とあり、出どころは Referer で見られると
# 考えられる。サーバーからの呼び出しには Referer が付かないので、自分で付ける
APP_REFERER = os.environ.get("RAKUTEN_REFERER", "https://kmykprn.github.io/roomplanner-web/")

# 商品ページの URL の形。shop と code は英数字・ハイフン・アンダースコア・ドット
ITEM_URL = re.compile(r"^https?://item\.rakuten\.co\.jp/([a-z0-9][a-z0-9\-]*)/([A-Za-z0-9][A-Za-z0-9\-_.]*)/?", re.IGNORECASE)

# 画像を取ってよいホスト。楽天の商品画像 CDN だけ（SSRF を防ぐ）
IMAGE_HOSTS = ("thumbnail.image.rakuten.co.jp", "image.rakuten.co.jp", "shop.r10s.jp", "tshop.r10s.jp")

# 画像の大きさ。API は 128px の URL を返すが、_ex を書き換えれば大きいものが取れる
IMAGE_SIZE = "1024x1024"
MAX_IMAGE_BYTES = 5 * 1024 * 1024
HTTP_TIMEOUT = 10


def parse_item_url(url: str) -> str | None:
    """商品ページの URL から API の itemCode（"shop:code"）を作る。楽天の商品ページでなければ None"""
    match = ITEM_URL.match(url.strip())
    if not match:
        return None
    shop, code = match.group(1).lower(), match.group(2)
    return f"{shop}:{code}"


def large_image_url(url: str) -> str:
    """API が返す 128px の画像 URL を、切り抜きに耐える大きさに書き換える。

    楽天の画像 CDN は `_ex=幅x高さ` で大きさを指定する。無ければ足す
    """
    parts = urlparse(url)
    query = [(k, v) for k, v in parse_qsl(parts.query) if k != "_ex"]
    query.append(("_ex", IMAGE_SIZE))
    return urlunparse(parts._replace(query=urlencode(query)))


# 寸法の書き方はまちまち。よくある形を順に試す。
#   幅120cm 奥行80cm 高さ75cm / 幅120×奥行80×高さ75cm / W120×D80×H75cm / 120×80×75cm（幅×奥行×高さ）
_NUM = r"(\d+(?:\.\d+)?)"
_LABELED = {
    "w": re.compile(r"(?:幅|横幅|W)\s*[:：]?\s*約?\s*" + _NUM + r"\s*(cm|mm|㎝)", re.IGNORECASE),
    "d": re.compile(r"(?:奥行き?|D)\s*[:：]?\s*約?\s*" + _NUM + r"\s*(cm|mm|㎝)", re.IGNORECASE),
    "h": re.compile(r"(?:高さ|H)\s*[:：]?\s*約?\s*" + _NUM + r"\s*(cm|mm|㎝)", re.IGNORECASE),
}
# 単位が末尾に 1 回だけの形: 幅120×奥行80×高さ75cm / W120×D80×H75cm / 120×80×75cm
_TRIPLE = re.compile(
    r"(?:幅|W)?\s*約?\s*" + _NUM + r"\s*(?:cm|mm|㎝)?\s*[×xX＊*]\s*"
    r"(?:奥行き?|D)?\s*約?\s*" + _NUM + r"\s*(?:cm|mm|㎝)?\s*[×xX＊*]\s*"
    r"(?:高さ|H)?\s*約?\s*" + _NUM + r"\s*(cm|mm|㎝)",
    re.IGNORECASE,
)


def _to_meters(value: str, unit: str) -> float:
    number = float(value)
    return number / 1000 if unit.lower() == "mm" else number / 100


def parse_dimensions(text: str) -> dict | None:
    """商品名や説明から幅・奥行・高さ（m）を拾う。3 つそろわなければ None。

    ありえない値（3cm 未満、5m 超）は誤検出として捨てる
    """
    if not text:
        return None
    found: dict[str, float] = {}
    for key, pattern in _LABELED.items():
        match = pattern.search(text)
        if match:
            found[key] = _to_meters(match.group(1), match.group(2))
    if len(found) < 3:
        match = _TRIPLE.search(text)
        if match:
            unit = match.group(4)
            found = {
                "w": _to_meters(match.group(1), unit),
                "d": _to_meters(match.group(2), unit),
                "h": _to_meters(match.group(3), unit),
            }
    if len(found) < 3:
        return None
    if any(not (0.03 <= v <= 5.0) for v in found.values()):
        return None
    return {k: round(found[k], 3) for k in ("w", "h", "d")}


class RakutenError(Exception):
    """楽天 API の応答が使えないとき。status は利用者に返す HTTP の状態コード"""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def lookup(item_code: str, application_id: str, access_key: str, affiliate_id: str) -> dict:
    """商品検索 API に itemCode で問い合わせ、1 件の商品情報を返す。"""
    params = {
        "format": "json",
        "itemCode": item_code,
        "applicationId": application_id,
        "accessKey": access_key,
        "hits": 1,
    }
    if affiliate_id:
        params["affiliateId"] = affiliate_id
    try:
        response = requests.get(
            SEARCH_ENDPOINT, params=params, timeout=HTTP_TIMEOUT, headers={"Referer": APP_REFERER}
        )
    except requests.RequestException as e:
        raise RakutenError(502, f"楽天に接続できない: {e}")
    if response.status_code == 429:
        raise RakutenError(429, "楽天の問い合わせが混み合っている。少し待ってから")
    if response.status_code != 200:
        raise RakutenError(502, f"楽天の応答が {response.status_code}")
    body = response.json()
    items = body.get("Items") or []
    if not items:
        raise RakutenError(404, "その商品が見つからない（販売終了か、URL が違う）")
    item = items[0]
    # 古い版は {"Item": {...}} で包んでいる。新しい版はそのまま
    if "Item" in item and isinstance(item["Item"], dict):
        item = item["Item"]
    images = item.get("mediumImageUrls") or []
    image_url = None
    if images:
        first = images[0]
        image_url = first.get("imageUrl") if isinstance(first, dict) else first
    return {
        "name": item.get("itemName", ""),
        "price": item.get("itemPrice"),
        "shop": item.get("shopName", ""),
        "url": item.get("itemUrl", ""),
        "affiliateUrl": item.get("affiliateUrl") or item.get("itemUrl", ""),
        "imageUrl": image_url,
        "size": parse_dimensions(f"{item.get('itemName', '')}\n{item.get('itemCaption', '')}"),
    }


def fetch_image(url: str) -> tuple[bytes, str]:
    """商品画像を取り、(bytes, MIME) で返す。楽天の CDN 以外は取りに行かない"""
    host = (urlparse(url).hostname or "").lower()
    if not any(host == h or host.endswith("." + h) for h in IMAGE_HOSTS):
        raise RakutenError(502, "商品画像の置き場が想定外")
    try:
        response = requests.get(large_image_url(url), timeout=HTTP_TIMEOUT, stream=True)
        response.raise_for_status()
        data = response.raw.read(MAX_IMAGE_BYTES + 1, decode_content=True)
    except requests.RequestException as e:
        raise RakutenError(502, f"商品画像を取れない: {e}")
    if len(data) > MAX_IMAGE_BYTES:
        raise RakutenError(502, "商品画像が大きすぎる")
    try:
        image = Image.open(io.BytesIO(data))
        image.verify()
        mime = Image.MIME.get(image.format or "", "")
    except (UnidentifiedImageError, Exception):
        raise RakutenError(502, "商品画像として読めない")
    if mime not in ("image/jpeg", "image/png", "image/webp"):
        raise RakutenError(502, f"商品画像の形式が想定外: {mime}")
    return data, mime
