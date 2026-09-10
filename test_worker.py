"""worker_entrypoint の status 書き込みを、GCS を偽物に差し替えて確かめる。"""
import json, os, sys
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OUTPUTS_BUCKET", "b")

from google.api_core.exceptions import NotFound, PreconditionFailed

class FakeBlob:
    def __init__(self, bucket, name):
        self.bucket, self.name, self._gen = bucket, name, None
    @property
    def _rec(self): return self.bucket.store.get(self.name)
    @property
    def generation(self): return self._gen
    def reload(self):
        if not self._rec: raise NotFound(self.name)
        self._gen = self._rec["gen"]
    def download_as_text(self):
        if not self._rec: raise NotFound(self.name)
        return self._rec["data"]
    def upload_from_string(self, data, content_type=None, if_generation_match=None):
        cur = self._rec
        if if_generation_match is not None:
            expected = cur["gen"] if cur else 0
            if if_generation_match != expected: raise PreconditionFailed("mismatch")
        self.bucket.seq += 1
        self.bucket.store[self.name] = {"data": data, "gen": self.bucket.seq}
    def exists(self): return self._rec is not None

class FakeBucket:
    def __init__(self): self.store, self.seq = {}, 0
    def blob(self, name): return FakeBlob(self, name)

with mock.patch("google.cloud.storage.Client"):
    import worker_entrypoint as w

BUCKET = FakeBucket()
w._bucket = BUCKET

results = []
def check(name, cond, detail=""):
    results.append(cond)
    print(("  OK   " if cond else "  FAIL ") + name + (f"  <- {detail}" if detail and not cond else ""))

def put(job_id, d):
    BUCKET.blob(f"jobs/{job_id}/status.json").upload_from_string(json.dumps(d))
def get(job_id):
    return json.loads(BUCKET.blob(f"jobs/{job_id}/status.json").download_as_text())

J = "job_0000000000000001"

# --- 1. 通常の更新 -----------------------------------------------------
print("\n[1] 通常の更新")
BUCKET.store.clear(); BUCKET.seq = 0
put(J, {"jobId": J, "uid": "u", "state": "queued", "createdAt": "c"})
w._update_status(J, state="running", phase="preparing")
s = get(J)
check("変更が反映される", s["state"] == "running" and s["phase"] == "preparing", str(s))
check("API 層が書いた項目が残る", s["uid"] == "u" and s["createdAt"] == "c", str(s))

# --- 2. 読んでから書くまでに API が割り込んだ場合 ------------------------
print("\n[2] API 層の割り込み（executionName を消さないこと）")
BUCKET.store.clear(); BUCKET.seq = 0
put(J, {"jobId": J, "uid": "u", "state": "queued", "createdAt": "c"})

original_download = FakeBlob.download_as_text
intruded = {"done": False}
def download_then_intrude(self):
    data = original_download(self)
    # 読んだ直後、書き戻す前に API 層が executionName と slot を書く
    if self.name.endswith("status.json") and not intruded["done"]:
        intruded["done"] = True
        d = json.loads(data)
        d.update({"executionName": "exec-abc", "slot": 0})
        BUCKET.blob(self.name).upload_from_string(json.dumps(d))
    return data

with mock.patch.object(FakeBlob, "download_as_text", download_then_intrude):
    w._update_status(J, state="running", phase="preparing")

s = get(J)
check("割り込みが起きたこと（前提の再現）", intruded["done"])
check("executionName が消えていない", s.get("executionName") == "exec-abc", str(s))
check("slot が消えていない", s.get("slot") == 0, str(s))
check("こちらの変更も入っている", s["state"] == "running" and s["phase"] == "preparing", str(s))

# --- 3. 競合し続けたら諦めて例外を上げること ----------------------------
print("\n[3] 競合し続けた場合")
BUCKET.store.clear(); BUCKET.seq = 0
put(J, {"jobId": J, "state": "queued"})
def always_intrude(self):
    data = original_download(self)
    if self.name.endswith("status.json"):
        BUCKET.blob(self.name).upload_from_string(data)  # 毎回 generation を進める
    return data
try:
    with mock.patch.object(FakeBlob, "download_as_text", always_intrude):
        w._update_status(J, state="running")
    check("諦めて例外を上げる", False, "例外が出なかった")
except RuntimeError as e:
    check("諦めて例外を上げる", "更新できなかった" in str(e), str(e))

# --- 4. phase の書き込み失敗で生成を落とさないこと ----------------------
print("\n[4] phase の書き込み失敗")
BUCKET.store.clear(); BUCKET.seq = 0
put(J, {"jobId": J, "state": "running"})
with mock.patch.object(w, "_update_status", side_effect=RuntimeError("書けない")):
    try:
        w._set_phase(J, "generating_texture")
        check("例外を外に出さない", True)
    except Exception as e:
        check("例外を外に出さない", False, str(e))

# --- 5. 印が前から順に1つずつしか進まないこと ---------------------------
print("\n[5] 工程の印")
markers = [m for m, _ in w.PHASE_MARKERS]
check("印は5つ", len(w.PHASE_MARKERS) == 5, str(len(w.PHASE_MARKERS)))
# ログに前の印がもう一度出ても巻き戻らないことを、走査ロジックで再現
log = [markers[0], markers[1], markers[0], markers[2]]
seen, nxt = [], 0
for line in log:
    if nxt < len(w.PHASE_MARKERS):
        m, p = w.PHASE_MARKERS[nxt]
        if m in line:
            seen.append(p); nxt += 1
# 3行目で1つ目の印が再び出るが、待っているのは3つ目なので何も起きない。
# 4行目で3つ目が来て初めて進む
check("同じ印が再出現しても巻き戻らない",
      seen == ["loading_texture_model", "loading_shape_model", "generating_shape"], str(seen))
check("工程は必ず前に進む", seen == [p for _, p in w.PHASE_MARKERS][:len(seen)], str(seen))

print()
print(f"{sum(results)}/{len(results)} 成功")
sys.exit(0 if all(results) else 1)
