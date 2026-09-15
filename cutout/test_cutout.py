"""切り抜き API の、モデル以外の部分を動かす。

モデルは 224MB あるので CI では読まない。推論は「中央の四角だけ前景」を返す
偽物に差し替え、認証・回数・画像の検証・切り詰めを確かめる。
main.py は読み込み時に GCS / Firebase / モデルへ繋ぎに行くので、import する前に差し替える。
"""
import io, json, os, sys
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

def reset():
    BUCKET.store.clear(); BUCKET.seq = 0

def quota_count(uid="u1"):
    names = [n for n in BUCKET.store if n.startswith(f"quota/{uid}/")]
    return json.loads(BUCKET.store[names[0]]["data"])["count"] if names else 0

results = []
def check(name, cond):
    results.append((name, cond)); print(("ok   " if cond else "FAIL ") + name)


# --- 預ける形（/cutout-jobs）の道具 ---
class FakeTasks:
    """積まれた処理を覚えるだけ。実行はテストが /run を叩いて行う"""
    def __init__(self): self.tasks = []
    def create_task(self, parent, task): self.tasks.append((parent, task))

def headers(token="google"):
    return {"Authorization": f"Bearer {token}"} if token else {}

def post_job(token="google", data=png_bytes(), filename="photo.png"):
    return client.post("/cutout-jobs", files={"image": (filename, data, "image/png")}, headers=headers(token))

def run_job(uid, job_id, token="task"):
    return client.post(f"/cutout-jobs/{uid}/{job_id}/run", headers=headers(token))

def status_of(job_id, token="google"):
    return client.get(f"/cutout-jobs/{job_id}", headers=headers(token))

def result_of(job_id, token="google"):
    return client.get(f"/cutout-jobs/{job_id}/result", headers=headers(token))

def cut(data=png_bytes()):
    """預けて処理して、結果の PNG を返す（一連の流れの近道）"""
    job_id = post_job(data=data).json()["id"]
    run_job("u1", job_id)
    return Image.open(io.BytesIO(result_of(job_id).content))

def fake_verify_task(authorization):
    if authorization != "Bearer task": raise HTTPException(status_code=403, detail="内部の入口")


# --- 預かる設定が無いとき ---
reset()
check("預かる設定が無ければ 503", post_job().status_code == 503)
check("503 では回数を数えない", quota_count() == 0)

# 以降は設定が入っているものとして動かす
fake_tasks = FakeTasks()
main._tasks = fake_tasks
main.SELF_URL = "https://cutout.example"
main.TASK_SERVICE_ACCOUNT = "cutout@example.iam"
main._verify_task_token = fake_verify_task

# --- 認証 ---
reset()
check("ヘッダ無しは 401", post_job(None).status_code == 401)
check("壊れたトークンは 401", post_job("bogus").status_code == 401)
check("匿名アカウントは 401", post_job("anon").status_code == 401)
check("メール未確認は 401", post_job("unverified").status_code == 401)
check("401 のときは回数を数えない", quota_count() == 0)
uid, email = main._identity_from_token("Bearer google")
check("メールは小文字にそろえる", (uid, email) == ("u1", "friend@example.com"))

# --- 預ける → 処理 → 取りに行く ---
reset()
r = post_job()
check("預けると 202 で受付番号と見込み秒数", r.status_code == 202 and "id" in r.json() and r.json()["expectedSeconds"] is not None)
job_id = r.json()["id"]
check("預けた時点で 1 回数える", quota_count() == 1)
check("Cloud Tasks に /run が積まれる", len(fake_tasks.tasks) == 1 and fake_tasks.tasks[0][1]["http_request"]["url"] == f"https://cutout.example/cutout-jobs/u1/{job_id}/run")
check("積む処理はサービスアカウントの OIDC トークン付き", fake_tasks.tasks[0][1]["http_request"]["oidc_token"]["service_account_email"] == "cutout@example.iam")
check("入力の JPEG が置かれる", f"jobs/u1/{job_id}/input.jpg" in BUCKET.store)
st = status_of(job_id).json()
check("預けた直後は queued", st["phase"] == "queued" and st["expectedSeconds"] is not None)
check("他人の受付番号は 404", status_of(job_id, "google2").status_code == 404)
check("できる前の結果は 404", result_of(job_id).status_code == 404)
check("/run はトークン無しでは 403", run_job("u1", job_id, token="bogus").status_code == 403)
check("/run は無い受付番号なら 404", run_job("u1", "nope").status_code == 404)
r = run_job("u1", job_id)
check("/run で処理して done", r.status_code == 200 and r.json()["phase"] == "done")
check("状態が done になる", status_of(job_id).json()["phase"] == "done")
out = Image.open(io.BytesIO(result_of(job_id).content))
check("透過 PNG で返る", out.mode == "RGBA")
# 800×600 の中央 1/2（400×300）に 2% の余白を足した大きさに切り詰まる
check("内容の周りで切り詰める", (400 <= out.width <= 420) and (300 <= out.height <= 315))
alpha = np.asarray(out.getchannel("A"))
check("中央は不透明、隅は透明", alpha[out.height // 2, out.width // 2] == 255 and alpha[0, 0] < 8)
check("done のあと /run が再び来ても処理しない", run_job("u1", job_id).json()["phase"] == "done")

# --- 推論の枠は 1 つ ---
reset(); fake_tasks.tasks.clear()
job_id = post_job().json()["id"]
with main._inference_slot:
    r = run_job("u1", job_id)
check("別の推論が走っている間の /run は 429（Cloud Tasks が再試行する）", r.status_code == 429)
check("429 のときは状態を動かさない", status_of(job_id).json()["phase"] == "queued")
check("枠が空けば処理できる", run_job("u1", job_id).json()["phase"] == "done")
check("処理のあと枠は空いている", main._inference_slot.acquire(blocking=False) and (main._inference_slot.release() or True))

# --- 推論の失敗 ---
reset(); fake_tasks.tasks.clear()
job_id = post_job().json()["id"]
with mock.patch.object(main, "_timed_predict_mask", side_effect=RuntimeError("boom")):
    r = run_job("u1", job_id)
check("推論に失敗したら failed で 200（再試行させない）", r.status_code == 200 and r.json()["phase"] == "failed")
st = status_of(job_id).json()
check("failed の理由が読める", st["phase"] == "failed" and "切り抜きに失敗" in st["error"])
check("失敗は回数を戻す", quota_count() == 0)
check("失敗のあとも枠は空いている", main._inference_slot.acquire(blocking=False) and (main._inference_slot.release() or True))

reset(); fake_tasks.tasks.clear()
with mock.patch.object(fake_tasks, "create_task", side_effect=RuntimeError("queue down")):
    r = post_job()
check("積めなければ 503", r.status_code == 503)
check("積めなかった分は回数を戻す", quota_count() == 0)

# --- 画像の検証 ---
reset()
check("空のファイルは 400", post_job(data=b"").status_code == 400)
check("画像でないものは 400", post_job(data=b"not an image").status_code == 400)
buf = io.BytesIO(); Image.new("RGB", (10, 10)).save(buf, "BMP")
check("対応外の形式は 400", post_job(data=buf.getvalue()).status_code == 400)
check("400 のときは回数を数えない", quota_count() == 0)
big = Image.new("RGB", (3000, 2000)); buf = io.BytesIO(); big.save(buf, "JPEG", quality=95)
check("大きな写真は長辺 1024 に縮めてから切る", cut(buf.getvalue()).width <= 1024)

# --- 回数 ---
reset()
check("上限（2 回）までは通る", post_job().status_code == 202 and post_job().status_code == 202)
check("3 回目は 429", post_job().status_code == 429)
check("429 では回数が増えない", quota_count() == 2)

reset()
main.RECENT_INFERENCE_SECONDS.clear()
check("実測が無いうちは既定の見込み", main._expected_inference_seconds() == main.DEFAULT_INFERENCE_SECONDS)
cut()
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
