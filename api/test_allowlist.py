"""許可リストの判定を、GCS と Firebase を偽物に差し替えて動かす。

test_queue.py と同じ作り。main.py は読み込み時に外へ繋ぎに行くので、import する前に差し替える。
"""
import datetime, json, os, sys
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.update({
    "OUTPUTS_BUCKET": "b", "CONFIG_BUCKET": "c", "JOB_NAME": "j",
    "PROJECT_ID": "p", "REGION": "r", "ALLOWED_ORIGINS": "https://x",
    "DISPATCHER_SERVICE_ACCOUNT": "job@p.iam.gserviceaccount.com",
    "DISPATCH_AUDIENCE": "https://hunyuan3d-dispatch/p",
    "ENFORCE_ALLOWLIST": "true",
})

from fastapi import HTTPException


class FakeBlob:
    def __init__(self, store, name): self.store, self.name = store, name
    def exists(self): return self.name in self.store
    def download_as_text(self): return self.store[self.name]

class FakeBucket:
    def __init__(self): self.store = {}
    def blob(self, name): return FakeBlob(self.store, name)

class FakeStorage:
    def __init__(self): self.buckets = {}
    def bucket(self, name): return self.buckets.setdefault(name, FakeBucket())


with mock.patch("google.cloud.storage.Client"), \
     mock.patch("firebase_admin.initialize_app"), \
     mock.patch("google.auth.default", return_value=(mock.MagicMock(), "p")), \
     mock.patch("google.auth.transport.requests.AuthorizedSession"):
    import main

STORAGE = FakeStorage()
main._storage = STORAGE
CONFIG = STORAGE.bucket("c")

def set_list(kind, values):
    CONFIG.store[main.ALLOWLIST_PATHS[kind]] = json.dumps(values)

def reset():
    CONFIG.store.clear()
    main._allowlist_cache.clear()

def allowed(uid, email=None):
    try:
        main._check_allowed(uid, email)
        return True
    except HTTPException as e:
        assert e.status_code == 403
        return False

results = []
def check(name, cond, detail=""):
    results.append((name, cond, detail))
    print(("  OK   " if cond else "  FAIL ") + name + (f"  <- {detail}" if detail and not cond else ""))


# --- 1. トークンから uid とメールを取り出す -----------------------------
print("\n[1] トークンの読み取り")
with mock.patch.object(main.fb_auth, "verify_id_token", return_value={"uid": "U1", "email": "Me@Example.com", "email_verified": True}):
    check("確認済みメールは小文字で返る", main._identity_from_token("Bearer t") == ("U1", "me@example.com"))
with mock.patch.object(main.fb_auth, "verify_id_token", return_value={"uid": "U2", "email": "x@example.com", "email_verified": False}):
    check("未確認のメールは使わない", main._identity_from_token("Bearer t") == ("U2", None))
with mock.patch.object(main.fb_auth, "verify_id_token", return_value={"uid": "anon", "firebase": {"sign_in_provider": "anonymous"}}):
    check("匿名にはメールが無い", main._identity_from_token("Bearer t") == ("anon", None))
    check("_uid_from_token は uid だけ返す（従来どおり）", main._uid_from_token("Bearer t") == "anon")
try:
    main._identity_from_token(None); check("ヘッダ無しは 401", False)
except HTTPException as e:
    check("ヘッダ無しは 401", e.status_code == 401)

# --- 2. 判定 -------------------------------------------------------------
print("\n[2] 許可リストの判定")
reset()
set_list("uids", ["U1"])
set_list("emails", ["Friend@Example.com"])
check("uid が載っていれば通る", allowed("U1"))
check("uid が載っていなければ 403", not allowed("U9"))
check("メールが載っていれば uid が無くても通る", allowed("U9", "friend@example.com"))
check("メールは大文字小文字を区別しない（一覧側が大文字でも）", allowed("U9", "friend@example.com"))
check("載っていないメールは 403", not allowed("U9", "stranger@example.com"))
check("匿名（メール無し）は uid でしか通らない", not allowed("U9", None))

reset()
set_list("uids", ["U1"])
check("メールの一覧が無ければ、メールでは通らない（fail-closed）", not allowed("U9", "friend@example.com"))
check("メールの一覧が無くても uid の一覧は効く", allowed("U1"))

reset()
check("両方の一覧が無ければ誰も通らない", not allowed("U1", "friend@example.com"))

# --- 3. 覚えておく時間 ---------------------------------------------------
print("\n[3] キャッシュ")
reset()
set_list("emails", ["a@example.com"])
check("読んだ直後は通る", allowed("U9", "a@example.com"))
set_list("emails", [])
check("60 秒以内は前の内容を使う", allowed("U9", "a@example.com"))
main._allowlist_cache["emails"] = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=61), main._allowlist_cache["emails"][1])
check("60 秒を過ぎたら読み直す", not allowed("U9", "a@example.com"))

# --- 4. フラグ -------------------------------------------------------------
print("\n[4] ENFORCE_ALLOWLIST")
reset()
main.ENFORCE_ALLOWLIST = False
check("外せば誰でも通る", allowed("U9", None))
main.ENFORCE_ALLOWLIST = True

ok = sum(1 for _, c, _ in results if c)
print(f"\n{ok}/{len(results)} OK")
sys.exit(0 if ok == len(results) else 1)
