"""切り抜き API の、モデル以外の部分を動かす。

モデルは 224MB あるので CI では読まない。推論は「中央の四角だけ前景」を返す
偽物に差し替え、認証・回数・画像の検証・切り詰めを確かめる。
main.py は読み込み時に GCS / Firebase / モデルへ繋ぎに行くので、import する前に差し替える。
"""
import base64, io, json, os, sys
from unittest import mock

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.update({"STATE_BUCKET": "s", "ALLOWED_ORIGINS": "https://x", "DAILY_LIMIT": "2"})

from fastapi import HTTPException
from fastapi.testclient import TestClient
from google.api_core.exceptions import NotFound, PreconditionFailed


class FakeBlob:
    def __init__(self, bucket, name):
        self.bucket, self.name = bucket, name
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
            if if_generation_match != expected:
                raise PreconditionFailed("generation mismatch")
        self.bucket.seq += 1
        self.bucket.store[self.name] = {"data": data, "gen": self.bucket.seq}


class FakeBucket:
    def __init__(self): self.store, self.seq = {}, 0
    def blob(self, name): return FakeBlob(self, name)


class FakeSession:
    """1024×1024 の入力に対して、中央の 1/2 の四角だけを前景にする"""
    def get_inputs(self):
        node = mock.Mock(); node.name = "input_image"
        return [node]
    def run(self, _outputs, feed):
        x = feed["input_image"]
        assert x.shape == (1, 3, 1024, 1024) and x.dtype == np.float32
        logits = np.full((1, 1, 1024, 1024), -10.0, dtype=np.float32)
        logits[:, :, 256:768, 256:768] = 10.0
        return [logits]


with mock.patch("google.cloud.storage.Client"), \
     mock.patch("firebase_admin.initialize_app"), \
     mock.patch("onnxruntime.InferenceSession", return_value=FakeSession()):
    import main

BUCKET = FakeBucket()
main._storage = mock.Mock(bucket=lambda name: BUCKET)
client = TestClient(app=main.app)

# トークンの検証を偽物にする。値は (uid, email, email_verified)
TOKENS = {
    "google": {"uid": "u1", "email": "Friend@Example.com", "email_verified": True},
    "unverified": {"uid": "u2", "email": "x@example.com", "email_verified": False},
    "anon": {"uid": "u3"},
}
def fake_verify(token):
    if token not in TOKENS: raise ValueError("bad token")
    return TOKENS[token]
main.fb_auth.verify_id_token = fake_verify


def png_bytes(size=(800, 600), color=(200, 100, 50)):
    buf = io.BytesIO(); Image.new("RGB", size, color).save(buf, "PNG"); return buf.getvalue()

def post(token, data=png_bytes(), filename="photo.png", stream=True):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    # 画面は Accept で流す形を頼む。付けなければ PNG（配信済みの画面向け）
    if stream: headers["Accept"] = "application/x-ndjson"
    return client.post("/cutouts", files={"image": (filename, data, "image/png")}, headers=headers)

def reset():
    BUCKET.store.clear(); BUCKET.seq = 0

def events(response):
    """NDJSON の応答を行ごとの dict にする"""
    return [json.loads(line) for line in response.text.splitlines() if line]

def result_png(response):
    """最後の行（done）の PNG を取り出す"""
    return Image.open(io.BytesIO(base64.b64decode(events(response)[-1]["png"])))

def quota_count(uid="u1"):
    names = [n for n in BUCKET.store if n.startswith(f"quota/{uid}/")]
    return json.loads(BUCKET.store[names[0]]["data"])["count"] if names else 0

results = []
def check(name, cond):
    results.append((name, cond)); print(("ok   " if cond else "FAIL ") + name)


# --- 認証 ---
reset()
check("ヘッダ無しは 401", post(None).status_code == 401)
check("壊れたトークンは 401", post("bogus").status_code == 401)
check("匿名アカウントは 401", post("anon").status_code == 401)
check("メール未確認は 401", post("unverified").status_code == 401)
check("401 のときは回数を数えない", quota_count() == 0)
uid, email = main._identity_from_token("Bearer google")
check("メールは小文字にそろえる", (uid, email) == ("u1", "friend@example.com"))

# --- 切り抜き ---
reset()
r = post("google")
check("Google ログイン済みは 200 で NDJSON", r.status_code == 200 and r.headers["content-type"].startswith("application/x-ndjson"))
phases = [e["phase"] for e in events(r)]
check("工程が順に流れる", phases == ["received", "cutting", "finishing", "done"])
check("推論中の行に見込み秒数が付く", isinstance(events(r)[1]["expectedSeconds"], (int, float)))
out = result_png(r)
check("透過 PNG で返る", out.mode == "RGBA")
# 800×600 の中央 1/2（400×300）に 2% の余白を足した大きさに切り詰まる
check("内容の周りで切り詰める", (400 <= out.width <= 420) and (300 <= out.height <= 315))
alpha = np.asarray(out.getchannel("A"))
check("中央は不透明、隅は透明", alpha[out.height // 2, out.width // 2] == 255 and alpha[0, 0] < 8)
check("1 回数えた", quota_count() == 1)

# --- Accept が無ければ PNG（配信済みの画面向け） ---
reset()
r = post("google", stream=False)
check("Accept 無しは PNG で返る", r.status_code == 200 and r.headers["content-type"] == "image/png")
check("PNG でも透過で切り詰まる", Image.open(io.BytesIO(r.content)).mode == "RGBA" and 400 <= Image.open(io.BytesIO(r.content)).width <= 420)
with mock.patch.object(main, "_timed_predict_mask", side_effect=RuntimeError("boom")):
    try:
        post("google", stream=False)
        check("PNG の形で推論に失敗したら 500", False)
    except RuntimeError:
        check("PNG の形で推論に失敗したら 500", True)
check("PNG の形でも失敗は回数を戻す", quota_count() == 1)

# --- 画像の検証 ---
reset()
check("空のファイルは 400", post("google", data=b"").status_code == 400)
check("画像でないものは 400", post("google", data=b"not an image").status_code == 400)
buf = io.BytesIO(); Image.new("RGB", (10, 10)).save(buf, "BMP")
check("対応外の形式は 400", post("google", data=buf.getvalue()).status_code == 400)
check("400 のときは回数を数えない", quota_count() == 0)
big = Image.new("RGB", (3000, 2000)); buf = io.BytesIO(); big.save(buf, "JPEG", quality=95)
r = post("google", data=buf.getvalue())
out = result_png(r)
check("大きな写真は長辺 1024 に縮めてから切る", r.status_code == 200 and out.width <= 1024)

# --- 回数 ---
reset()
check("上限（2 回）までは通る", post("google").status_code == 200 and post("google").status_code == 200)
check("3 回目は 429", post("google").status_code == 429)
check("429 では回数が増えない", quota_count() == 2)

reset()
with mock.patch.object(main, "_timed_predict_mask", side_effect=RuntimeError("boom")):
    r = post("google")
check("推論の失敗は最後の行で failed", r.status_code == 200 and events(r)[-1]["phase"] == "failed")
check("こちら都合の失敗は回数を戻す", quota_count() == 0)

reset()
main.RECENT_INFERENCE_SECONDS.clear()
check("実測が無いうちは既定の見込み", main._expected_inference_seconds() == main.DEFAULT_INFERENCE_SECONDS)
post("google")
check("推論のたびに実測を覚える", len(main.RECENT_INFERENCE_SECONDS) == 1)

reset()
# 読んでから書くまでの間に別のリクエストが書いた → 409
original_read = main._read_quota
def racy_read(blob):
    count, gen = original_read(blob)
    blob.upload_from_string(json.dumps({"count": count + 1}))
    return count, gen
with mock.patch.object(main, "_read_quota", racy_read):
    try:
        main._consume_quota("u1")
        check("競合は 409", False)
    except HTTPException as e:
        check("競合は 409", e.status_code == 409)

# --- 切り詰め ---
empty = Image.new("RGBA", (50, 40), (0, 0, 0, 0))
check("何も残らなければそのまま返す", main.crop_to_content(empty).size == (50, 40))

failed = [n for n, ok in results if not ok]
print(f"\n{len(results) - len(failed)}/{len(results)} passed")
sys.exit(1 if failed else 0)
