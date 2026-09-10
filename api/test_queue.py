"""待機列のロジックを、GCS を偽物に差し替えて動かす。

main.py は読み込み時に GCS / Firebase / google.auth へ繋ぎに行くので、
import する前に差し替える。
"""
import datetime, json, os, sys, types
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.update({
    "OUTPUTS_BUCKET": "b", "CONFIG_BUCKET": "c", "JOB_NAME": "j",
    "PROJECT_ID": "p", "REGION": "r", "ALLOWED_ORIGINS": "https://x",
    "DISPATCHER_SERVICE_ACCOUNT": "job@p.iam.gserviceaccount.com",
    "DISPATCH_AUDIENCE": "https://hunyuan3d-dispatch/p",
    "MAX_RUNNING_JOBS": "2", "MAX_QUEUED_JOBS_PER_UID": "3",
})

from google.api_core.exceptions import NotFound, PreconditionFailed


class FakeBlob:
    def __init__(self, bucket, name):
        self.bucket, self.name = bucket, name
    @property
    def _rec(self): return self.bucket.store.get(self.name)
    @property
    def generation(self): return self._rec["gen"] if self._rec else None
    def reload(self):
        if not self._rec: raise NotFound(self.name)
    def download_as_text(self):
        if not self._rec: raise NotFound(self.name)
        return self._rec["data"]
    def upload_from_string(self, data, content_type=None, if_generation_match=None):
        cur = self._rec
        if if_generation_match is not None:
            expected = cur["gen"] if cur else 0
            if if_generation_match != expected:
                raise PreconditionFailed("generation mismatch")
        self.bucket.seq += 1
        self.bucket.store[self.name] = {"data": data, "gen": self.bucket.seq}
    def delete(self, if_generation_match=None):
        cur = self._rec
        if not cur: raise NotFound(self.name)
        if if_generation_match is not None and if_generation_match != cur["gen"]:
            raise PreconditionFailed("generation mismatch")
        del self.bucket.store[self.name]


class FakeBucket:
    def __init__(self): self.store, self.seq = {}, 0
    def blob(self, name): return FakeBlob(self, name)
    def list_blobs(self, prefix=""):
        return [FakeBlob(self, n) for n in sorted(self.store) if n.startswith(prefix)]


with mock.patch("google.cloud.storage.Client"), \
     mock.patch("firebase_admin.initialize_app"), \
     mock.patch("google.auth.default", return_value=(mock.MagicMock(), "p")), \
     mock.patch("google.auth.transport.requests.AuthorizedSession"):
    import main

BUCKET = FakeBucket()
main._bucket = lambda: BUCKET

def iso(offset_seconds):
    t = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=offset_seconds)
    return t.isoformat(timespec="seconds")

def put_job(job_id, uid, state, created_at, **extra):
    status = {"jobId": job_id, "uid": uid, "state": state,
              "createdAt": created_at, "updatedAt": created_at,
              "executionName": None, "error": None, **extra}
    BUCKET.blob(f"jobs/{job_id}/status.json").upload_from_string(json.dumps(status))
    return status

def reset():
    BUCKET.store.clear(); BUCKET.seq = 0

results = []
def check(name, cond, detail=""):
    results.append((name, cond, detail))
    print(("  OK   " if cond else "  FAIL ") + name + (f"  <- {detail}" if detail and not cond else ""))


# --- 1. 目印の名前が往復すること ---------------------------------------
print("\n[1] 目印の名前")
reset()
created = iso(-100)
b = main._pending_marker_blob("uidA", created, "job_0000000000000001")
parsed = main._parse_pending_name(b.name)
check("uid・受付時刻・ジョブIDを復元できる", parsed == ("uidA", created, "job_0000000000000001"), str(parsed))
check("関係ない名前は None", main._parse_pending_name("jobs/x/status.json") is None)

# --- 2. 受付順に取り出すこと -------------------------------------------
print("\n[2] 受付順のディスパッチ")
reset()
for n, age in [(1, -300), (2, -100), (3, -200)]:
    jid = f"job_{n:016d}"; c = iso(age)
    put_job(jid, "uidA", "queued", c)
    main._pending_marker_blob("uidA", c, jid).upload_from_string("")
oldest = main._oldest_queued_job()
check("最も古い job_...0001 が選ばれる", oldest["jobId"] == "job_0000000000000001", oldest["jobId"])

# --- 3. 待機上限 --------------------------------------------------------
print("\n[3] 1利用者あたりの待機上限（3件）")
reset()
for n in range(3):
    jid = f"job_{n:016d}"; c = iso(-100 - n)
    put_job(jid, "uidA", "queued", c)
    main._pending_marker_blob("uidA", c, jid).upload_from_string("")
try:
    main._reserve_queue_space("uidA", iso(0), "job_000000000000000x")
    check("4件目は 429 で弾かれる", False, "例外が出なかった")
except main.HTTPException as e:
    check("4件目は 429 で弾かれる", e.status_code == 429, str(e.status_code))

# --- 4. 取り残しの掃除（通知の取りこぼしで永久に 429 にならないこと）----
print("\n[4] 取り残しの自己修復")
reset()
for n in range(3):
    jid = f"job_{n:016d}"; c = iso(-3600)
    # 起動済み（executionName あり）なのに目印が残っている状態
    put_job(jid, "uidA", "running", c, executionName="exec")
    main._pending_marker_blob("uidA", c, jid).upload_from_string("")
main._reserve_queue_space("uidA", iso(0), "job_00000000000000ff")
left = [b.name for b in BUCKET.list_blobs(prefix="queue/pending/uidA/")]
check("古い目印3件が掃除され、新しい1件だけ残る", len(left) == 1, str(left))

# --- 5. status を書く前の隙間が消されないこと --------------------------
print("\n[5] status.json を書く前の隙間")
reset()
c = iso(-1)
main._pending_marker_blob("uidA", c, "job_00000000000000aa").upload_from_string("")
main._reserve_queue_space("uidA", iso(0), "job_00000000000000bb")
left = [b.name for b in BUCKET.list_blobs(prefix="queue/pending/uidA/")]
check("取り立ての目印は掃除されない", len(left) == 2, str(left))
check("取り立ての目印は上限に数える", main._parse_pending_name(left[0]) is not None)

# --- 6. GPU 枠の回収 ----------------------------------------------------
print("\n[6] GPU 枠の回収")
reset()
# 枠0: status ごと消えたジョブ（30日で消える）
BUCKET.blob("queue/slots/0.json").upload_from_string(
    json.dumps({"jobId": "job_00000000000000de", "claimedAt": iso(-100)}))
# 枠1: OOM で running のまま止まったジョブ
put_job("job_00000000000000ad", "uidA", "running", iso(-3600), executionName="exec-dead")
BUCKET.blob("queue/slots/1.json").upload_from_string(
    json.dumps({"jobId": "job_00000000000000ad", "claimedAt": iso(-3600)}))
with mock.patch.object(main, "_execution_is_dead", return_value=True):
    main._release_finished_global_slots()
check("status ごと消えた枠が解放される", "queue/slots/0.json" not in BUCKET.store)
check("OOM で止まった枠が解放される", "queue/slots/1.json" not in BUCKET.store)
freed = json.loads(BUCKET.blob("jobs/job_00000000000000ad/status.json").download_as_text())
check("止まったジョブは failed になる", freed["state"] == "failed", freed["state"])

# --- 7. 生きているジョブの枠は解放しないこと ---------------------------
print("\n[7] 生きているジョブは触らない")
reset()
put_job("job_00000000000000c1", "uidA", "running", iso(-60), executionName="exec-live")
BUCKET.blob("queue/slots/0.json").upload_from_string(
    json.dumps({"jobId": "job_00000000000000c1", "claimedAt": iso(-60)}))
with mock.patch.object(main, "_execution_is_dead", return_value=False):
    main._release_finished_global_slots()
check("動作中の枠は保持される", "queue/slots/0.json" in BUCKET.store)

# --- 8. 同時実行数の上限 ------------------------------------------------
print("\n[8] GPU 同時実行数の上限（2本）")
reset()
s0 = main._claim_global_slot("job_0000000000000001")
s1 = main._claim_global_slot("job_0000000000000002")
s2 = main._claim_global_slot("job_0000000000000003")
check("2本まで取れる", {s0, s1} == {0, 1}, f"{s0},{s1}")
check("3本目は取れない", s2 is None, str(s2))
main._release_global_slot(0, "job_0000000000000009")
check("他人のジョブIDでは解放できない", "queue/slots/0.json" in BUCKET.store)
main._release_global_slot(0, "job_0000000000000001")
check("自分のジョブIDなら解放できる", "queue/slots/0.json" not in BUCKET.store)

# --- 9. 待機列全体の流れ ------------------------------------------------
print("\n[9] 3件待機 / GPU 2枠")
reset()
for n in range(3):
    jid = f"job_{n:016d}"; c = iso(-300 + n)
    put_job(jid, "uidA", "queued", c)
    main._pending_marker_blob("uidA", c, jid).upload_from_string("")

started = []
def fake_start(job_id, slot):
    started.append((job_id, slot)); return f"exec-{job_id}"

with mock.patch.object(main, "_start_job", side_effect=fake_start):
    main._dispatch_queued_jobs()
check("2件だけ起動する", len(started) == 2, str(started))
check("受付順に起動する",
      [j for j, _ in started] == ["job_0000000000000000", "job_0000000000000001"], str(started))
check("枠は0と1に割り当てられる", sorted(s for _, s in started) == [0, 1], str(started))
check("起動した2件の目印は外れる",
      len(BUCKET.list_blobs(prefix="queue/pending/uidA/")) == 1,
      str([b.name for b in BUCKET.list_blobs(prefix="queue/pending/uidA/")]))
third = json.loads(BUCKET.blob("jobs/job_0000000000000002/status.json").download_as_text())
check("3件目は queued のまま", third["state"] == "queued" and not third["executionName"], str(third["state"]))

# --- 10. 1件終わったら次が動くこと --------------------------------------
print("\n[10] 完了で次が動く")
reset_started = started.clear()
done = json.loads(BUCKET.blob("jobs/job_0000000000000000/status.json").download_as_text())
done["state"] = "succeeded"
BUCKET.blob("jobs/job_0000000000000000/status.json").upload_from_string(json.dumps(done))
with mock.patch.object(main, "_start_job", side_effect=fake_start), \
     mock.patch.object(main, "_execution_is_dead", return_value=False):
    main._dispatch_queued_jobs()
check("空いた枠で3件目が起動する",
      [j for j, _ in started] == ["job_0000000000000002"], str(started))
check("待機の目印が空になる",
      len(BUCKET.list_blobs(prefix="queue/pending/uidA/")) == 0)

# --- 11. 起動に失敗したら枠と回数を返すこと -----------------------------
print("\n[11] 起動失敗の後始末")
reset()
c = iso(-100)
put_job("job_00000000000000f0", "uidA", "queued", c)
main._pending_marker_blob("uidA", c, "job_00000000000000f0").upload_from_string("")
with mock.patch.object(main, "_start_job", side_effect=RuntimeError("起動できない")), \
     mock.patch.object(main, "_restore_quota") as restore:
    main._dispatch_queued_jobs()
failed = json.loads(BUCKET.blob("jobs/job_00000000000000f0/status.json").download_as_text())
check("failed になる", failed["state"] == "failed", failed["state"])
check("GPU 枠を握ったままにしない",
      not [n for n in BUCKET.store if n.startswith("queue/slots/")],
      str([n for n in BUCKET.store if n.startswith("queue/slots/")]))
check("待機の目印を外す", len(BUCKET.list_blobs(prefix="queue/pending/uidA/")) == 0)
check("1日の回数を返す", restore.called)


# --- 12. status.json への記録に失敗しても枠を回収できること --------------
print("\n[12] status への記録を取りこぼした枠")
reset()
c = iso(-3600)
put_job("job_00000000000000e1", "uidA", "queued", c)
main._pending_marker_blob("uidA", c, "job_00000000000000e1").upload_from_string("")

# _record_execution だけが競合で失敗し、status に execution 名が残らない状況
with mock.patch.object(main, "_start_job", return_value="exec-lost"), \
     mock.patch.object(main, "_record_execution"):
    main._dispatch_queued_jobs()

slot0 = json.loads(BUCKET.blob("queue/slots/0.json").download_as_text())
check("枠には execution 名が残る", slot0.get("executionName") == "exec-lost", str(slot0))
stored = json.loads(BUCKET.blob("jobs/job_00000000000000e1/status.json").download_as_text())
check("status 側には残っていない（前提の再現）", not stored.get("executionName"))

with mock.patch.object(main, "_execution_is_dead", return_value=True) as dead:
    main._release_finished_global_slots()
check("枠の記録で死活を照会する", dead.call_args[0][0] == "exec-lost", str(dead.call_args))
check("枠が回収される", "queue/slots/0.json" not in BUCKET.store)


print()
bad = [n for n, ok, _ in results if not ok]
print(f"{len(results)-len(bad)}/{len(results)} 成功")
sys.exit(1 if bad else 0)
