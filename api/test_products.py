"""楽天の商品取り込みを、楽天 API と GCS と Firebase を偽物に差し替えて動かす。

test_allowlist.py と同じ作り。main.py は読み込み時に外へ繋ぎに行くので、import する前に差し替える。
"""
import base64, io, json, os, sys
from unittest import mock

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.update({
    "OUTPUTS_BUCKET": "b", "CONFIG_BUCKET": "c", "JOB_NAME": "j",
    "PROJECT_ID": "p", "REGION": "r", "ALLOWED_ORIGINS": "https://x",
    "DISPATCHER_SERVICE_ACCOUNT": "job@p.iam.gserviceaccount.com",
    "DISPATCH_AUDIENCE": "https://hunyuan3d-dispatch/p",
    "RAKUTEN_APPLICATION_ID": "app", "RAKUTEN_ACCESS_KEY": "key", "RAKUTEN_AFFILIATE_ID": "aff",
    "PRODUCT_DAILY_LIMIT": "2",
})

from fastapi.testclient import TestClient
from google.api_core.exceptions import NotFound, PreconditionFailed

import rakuten


class FakeBlob:
    def __init__(self, bucket, name): self.bucket, self.name = bucket, name
    @property
    def _rec(self): return self.bucket.store.get(self.name)
    @property
    def generation(self): return self._rec["gen"] if self._rec else None
    def exists(self): return self._rec is not None
    def reload(self):
        if not self._rec: raise NotFound(self.name)
    def download_as_text(self): return self._rec["data"]
    def upload_from_string(self, data, content_type=None, if_generation_match=None):
        cur = self._rec
        if if_generation_match is not None:
            expected = cur["gen"] if cur else 0
            if if_generation_match != expected: raise PreconditionFailed("generation mismatch")
        self.bucket.seq += 1
        self.bucket.store[self.name] = {"data": data, "gen": self.bucket.seq}

class FakeBucket:
    def __init__(self): self.store, self.seq = {}, 0
    def blob(self, name): return FakeBlob(self, name)


with mock.patch("google.cloud.storage.Client"), \
     mock.patch("firebase_admin.initialize_app"), \
     mock.patch("google.auth.default", return_value=(mock.MagicMock(), "p")), \
     mock.patch("google.auth.transport.requests.AuthorizedSession"):
    import main

BUCKET = FakeBucket()
main._bucket = lambda: BUCKET
main.fb_auth.verify_id_token = lambda token: {"uid": "u1", "email": "a@b.c", "email_verified": True}
client = TestClient(app=main.app)

results = []
def check(name, cond):
    results.append((name, cond)); print(("ok   " if cond else "FAIL ") + name)

# --- URL の解析 ---
check("商品ページの URL から shop:code", rakuten.parse_item_url("https://item.rakuten.co.jp/some-shop/abc-123/") == "some-shop:abc-123")
check("末尾の / やクエリがあってもよい", rakuten.parse_item_url("https://item.rakuten.co.jp/shop1/item_9.5?s=1#x") == "shop1:item_9.5")
check("店名は小文字にそろえる", rakuten.parse_item_url("https://item.rakuten.co.jp/ShopX/ITEM/") == "shopx:ITEM")
check("楽天以外は None", rakuten.parse_item_url("https://www.amazon.co.jp/dp/B0XXXX") is None)
check("検索ページは None", rakuten.parse_item_url("https://search.rakuten.co.jp/search/mall/sofa/") is None)

# --- 寸法の抽出 ---
m = rakuten.parse_dimensions
check("幅/奥行/高さ + cm", m("サイズ: 幅120cm 奥行80cm 高さ75cm") == {"w": 1.2, "h": 0.75, "d": 0.8})
check("約・全角コロン・㎝", m("幅：約 90㎝、奥行き：約 45㎝、高さ：約 180㎝") == {"w": 0.9, "h": 1.8, "d": 0.45})
check("W×D×H + mm", m("外寸 W1200×D800×H750mm") == {"w": 1.2, "h": 0.75, "d": 0.8})
check("単位が末尾に 1 回", m("幅120×奥行80×高さ75cm") == {"w": 1.2, "h": 0.75, "d": 0.8})
check("数字だけの 3 つ組は 幅×奥行×高さ", m("本体サイズ 120×80×75cm") == {"w": 1.2, "h": 0.75, "d": 0.8})
check("2 つしか無ければ None", m("幅120cm 高さ75cm") is None)
check("ありえない値は捨てる", m("幅1cm 奥行80cm 高さ75cm") is None)
check("寸法が無ければ None", m("北欧風 2人掛けソファ グレー") is None)

# --- 画像 URL ---
check("_ex を 1024 に書き換える", rakuten.large_image_url("https://thumbnail.image.rakuten.co.jp/@0_mall/s/cabinet/a.jpg?_ex=128x128") == "https://thumbnail.image.rakuten.co.jp/@0_mall/s/cabinet/a.jpg?_ex=1024x1024")
check("_ex が無ければ足す", rakuten.large_image_url("https://shop.r10s.jp/s/cabinet/a.jpg").endswith("a.jpg?_ex=1024x1024"))

# --- 楽天 API の応答の読み方（両方の版） ---
class FakeResponse:
    def __init__(self, status, body): self.status_code, self._body = status, body
    def json(self): return self._body
ITEM = {"itemName": "北欧ソファ 幅120cm 奥行80cm 高さ75cm", "itemPrice": 24800, "shopName": "家具店", "itemUrl": "https://item.rakuten.co.jp/shop/x/",
        "affiliateUrl": "https://hb.afl.rakuten.co.jp/…", "mediumImageUrls": [{"imageUrl": "https://thumbnail.image.rakuten.co.jp/@0_mall/shop/a.jpg?_ex=128x128"}], "itemCaption": ""}
with mock.patch.object(rakuten.requests, "get", return_value=FakeResponse(200, {"Items": [{"Item": ITEM}]})):
    p = rakuten.lookup("shop:x", "app", "key", "aff")
check("旧い版（Item で包む）を読める", p["name"].startswith("北欧") and p["price"] == 24800 and p["size"] == {"w": 1.2, "h": 0.75, "d": 0.8})
with mock.patch.object(rakuten.requests, "get", return_value=FakeResponse(200, {"Items": [{**ITEM, "mediumImageUrls": ["https://shop.r10s.jp/a.jpg"]}]})):
    p = rakuten.lookup("shop:x", "app", "key", "aff")
check("新しい版（そのまま・画像が文字列）を読める", p["imageUrl"] == "https://shop.r10s.jp/a.jpg" and p["affiliateUrl"].startswith("https://hb.afl"))
with mock.patch.object(rakuten.requests, "get", return_value=FakeResponse(200, {"Items": []})):
    try:
        rakuten.lookup("shop:x", "app", "key", "aff"); check("見つからなければ 404", False)
    except rakuten.RakutenError as e:
        check("見つからなければ 404", e.status == 404)
with mock.patch.object(rakuten.requests, "get", return_value=FakeResponse(429, {})):
    try:
        rakuten.lookup("shop:x", "app", "key", "aff"); check("混み合っていれば 429", False)
    except rakuten.RakutenError as e:
        check("混み合っていれば 429", e.status == 429)

# --- 画像の取得 ---
def png_bytes():
    buf = io.BytesIO(); Image.new("RGB", (16, 16), (200, 100, 50)).save(buf, "PNG"); return buf.getvalue()
class FakeRaw:
    def __init__(self, data): self._data = data
    def read(self, n, decode_content=True): return self._data[:n]
class FakeImageResponse:
    def __init__(self, data): self.raw = FakeRaw(data)
    def raise_for_status(self): pass
with mock.patch.object(rakuten.requests, "get", return_value=FakeImageResponse(png_bytes())) as g:
    data, mime = rakuten.fetch_image("https://thumbnail.image.rakuten.co.jp/@0_mall/shop/a.jpg?_ex=128x128")
    check("画像を取って MIME を返す", mime == "image/png" and len(data) > 0)
    check("大きいサイズで取りに行く", "_ex=1024x1024" in g.call_args[0][0])
try:
    rakuten.fetch_image("https://evil.example.com/a.jpg"); check("楽天の CDN 以外は取りに行かない", False)
except rakuten.RakutenError as e:
    check("楽天の CDN 以外は取りに行かない", e.status == 502)
with mock.patch.object(rakuten.requests, "get", return_value=FakeImageResponse(b"not an image")):
    try:
        rakuten.fetch_image("https://shop.r10s.jp/a.jpg"); check("画像でなければ 502", False)
    except rakuten.RakutenError as e:
        check("画像でなければ 502", e.status == 502)

# --- 入口 ---
def post(url, token="t"):
    return client.post("/products", json={"url": url}, headers={"Authorization": f"Bearer {token}"} if token else {})

def quota_count():
    names = [n for n in BUCKET.store if n.startswith("quota/u1/") and n.endswith("-products.json")]
    return json.loads(BUCKET.store[names[0]]["data"])["count"] if names else 0

BUCKET.store.clear()
check("トークン無しは 401", post("https://item.rakuten.co.jp/shop/x/", token=None).status_code == 401)
check("楽天以外の URL は 400", post("https://www.amazon.co.jp/dp/B0").status_code == 400)
check("400 では回数を数えない", quota_count() == 0)
with mock.patch.object(rakuten, "lookup", return_value={**{"name": "n", "price": 1, "shop": "s", "url": "u", "affiliateUrl": "a", "size": None}, "imageUrl": "https://shop.r10s.jp/a.jpg"}), \
     mock.patch.object(rakuten, "fetch_image", return_value=(png_bytes(), "image/png")):
    r = post("https://item.rakuten.co.jp/shop/x/")
    check("取り込めれば 200", r.status_code == 200)
    body = r.json()
    check("商品情報と画像（base64）が返る", body["name"] == "n" and body["affiliateUrl"] == "a" and base64.b64decode(body["imageBase64"])[:4] == b"\x89PNG" and body["imageType"] == "image/png")
    check("1 回数えた", quota_count() == 1)
    post("https://item.rakuten.co.jp/shop/x/")
    check("上限（2 回）を超えると 429", post("https://item.rakuten.co.jp/shop/x/").status_code == 429)
BUCKET.store.clear()
with mock.patch.object(rakuten, "lookup", side_effect=rakuten.RakutenError(404, "無い")):
    check("楽天に無ければ 404", post("https://item.rakuten.co.jp/shop/x/").status_code == 404)
with mock.patch.object(main, "RAKUTEN_ACCESS_KEY", ""):
    check("設定が無ければ 503", post("https://item.rakuten.co.jp/shop/x/").status_code == 503)

failed = [n for n, ok in results if not ok]
print(f"\n{len(results) - len(failed)}/{len(results)} passed")
sys.exit(1 if failed else 0)
