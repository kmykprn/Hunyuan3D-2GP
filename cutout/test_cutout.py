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
    def download_as_text(self):
        if not self._rec: raise NotFound(self.name)
        return self._rec["data"]
    def download_as_bytes(self):
        if not self._rec: raise NotFound(self.name)
        data = self._rec["data"]
        return data if isinstance(data, bytes) else data.encode()
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
    "google2": {"uid": "u9", "email": "other@example.com", "email_verified": True},
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

# 推論が長いあいだは、同じ工程の行を繰り返し流す（接続を黙らせない）
import time as _time
def slow_predict(img):
    _time.sleep(0.35)
    return main.predict_mask(main._session, img)
with mock.patch.object(main, "HEARTBEAT_SECONDS", 0.1), mock.patch.object(main, "_timed_predict_mask", side_effect=slow_predict):
    reset()
    beats = events(post("google"))
cutting = [e for e in beats if e["phase"] == "cutting"]
check("推論中は行を繰り返し流す", len(cutting) >= 3)
check("繰り返す行に経過秒が増えていく", [e["elapsed"] for e in cutting] == sorted(e["elapsed"] for e in cutting) and cutting[-1]["elapsed"] > 0)
check("繰り返しても工程の順は変わらない", [p for p in (e["phase"] for e in beats) if p != "cutting"] == ["received", "finishing", "done"])

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

# --- 預ける形（/cutout-jobs） ---
class FakeTasks:
    """積まれた処理を覚えるだけ。実行はテストが /run を叩いて行う"""
    def __init__(self): self.tasks = []
    def create_task(self, parent, task): self.tasks.append((parent, task))

def job_headers(token="google"):
    return {"Authorization": f"Bearer {token}"}

def post_job(token="google", data=png_bytes()):
    return client.post("/cutout-jobs", files={"image": ("photo.png", data, "image/png")}, headers=job_headers(token))

def run_job(uid, job_id, token="task"):
    return client.post(f"/cutout-jobs/{uid}/{job_id}/run", headers=job_headers(token))

def fake_verify_task(authorization):
    if authorization != "Bearer task": raise HTTPException(status_code=403, detail="内部の入口")

reset()
check("預かる設定が無ければ 503", post_job().status_code == 503)
check("503 では回数を数えない", quota_count() == 0)

fake_tasks = FakeTasks()
with mock.patch.object(main, "_tasks", fake_tasks), \
     mock.patch.object(main, "SELF_URL", "https://cutout.example"), \
     mock.patch.object(main, "TASK_SERVICE_ACCOUNT", "cutout@example.iam"), \
     mock.patch.object(main, "_verify_task_token", fake_verify_task):
    reset()
    check("預ける入口もヘッダ無しは 401", post_job(token=None).status_code == 401)
    r = post_job()
    check("預けると 202 で受付番号と見込み秒数", r.status_code == 202 and "id" in r.json() and isinstance(r.json()["expectedSeconds"], (int, float)))
    job_id = r.json()["id"]
    check("預けた時点で 1 回数える", quota_count() == 1)
    check("Cloud Tasks に /run が積まれる", len(fake_tasks.tasks) == 1 and fake_tasks.tasks[0][1]["http_request"]["url"] == f"https://cutout.example/cutout-jobs/u1/{job_id}/run")
    check("積む処理はサービスアカウントの OIDC トークン付き", fake_tasks.tasks[0][1]["http_request"]["oidc_token"]["service_account_email"] == "cutout@example.iam")
    check("入力の JPEG が置かれる", f"jobs/u1/{job_id}/input.jpg" in BUCKET.store)
    st = client.get(f"/cutout-jobs/{job_id}", headers=job_headers()).json()
    check("預けた直後は queued", st["phase"] == "queued" and st["expectedSeconds"] is not None)
    check("他人の受付番号は 404", client.get(f"/cutout-jobs/{job_id}", headers=job_headers("google2")).status_code == 404)
    check("できる前の結果は 404", client.get(f"/cutout-jobs/{job_id}/result", headers=job_headers()).status_code == 404)
    check("/run はトークン無しでは 403", run_job("u1", job_id, token="bogus").status_code == 403)
    check("/run は無い受付番号なら 404", run_job("u1", "nope").status_code == 404)
    r = run_job("u1", job_id)
    check("/run で処理して done", r.status_code == 200 and r.json()["phase"] == "done")
    st = client.get(f"/cutout-jobs/{job_id}", headers=job_headers()).json()
    check("状態が done になる", st["phase"] == "done")
    out = Image.open(io.BytesIO(client.get(f"/cutout-jobs/{job_id}/result", headers=job_headers()).content))
    check("結果は透過 PNG で切り詰まっている", out.mode == "RGBA" and 400 <= out.width <= 420)
    check("done のあと /run が再び来ても処理しない", run_job("u1", job_id).json()["phase"] == "done")

    reset(); fake_tasks.tasks.clear()
    job_id = post_job().json()["id"]
    with mock.patch.object(main, "_timed_predict_mask", side_effect=RuntimeError("boom")):
        r = run_job("u1", job_id)
    check("推論に失敗したら failed で 200（再試行させない）", r.status_code == 200 and r.json()["phase"] == "failed")
    st = client.get(f"/cutout-jobs/{job_id}", headers=job_headers()).json()
    check("failed の理由が読める", st["phase"] == "failed" and "切り抜きに失敗" in st["error"])
    check("失敗は回数を戻す", quota_count() == 0)

    reset(); fake_tasks.tasks.clear()
    with mock.patch.object(fake_tasks, "create_task", side_effect=RuntimeError("queue down")):
        r = post_job()
    check("積めなければ 503", r.status_code == 503)
    check("積めなかった分は回数を戻す", quota_count() == 0)

# --- 切り詰め ---
empty = Image.new("RGBA", (50, 40), (0, 0, 0, 0))
check("何も残らなければそのまま返す", main.crop_to_content(empty).size == (50, 40))

failed = [n for n, ok in results if not ok]
print(f"\n{len(results) - len(failed)}/{len(results)} passed")
sys.exit(1 if failed else 0)
