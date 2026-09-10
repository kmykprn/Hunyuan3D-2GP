# roomplanner-web から叩く API。仕様は SPEC.md。
#
# 段階1では認証コードを書かない。Cloud Run の IAM で閉じておけば
# 自分以外は叩けないので、認証の問題と配線の問題を混ぜずに確認できる。
# uid は固定値で、段階3で Firebase の匿名認証に差し替える。
#
# 状態は GCS のファイルだけで持つ。書き込みは1ジョブ数回、読みは
# ポーリングだけなので Firestore は要らない（SPEC.md 参照）。

import datetime
import io
import json
import os
import re
import traceback
import uuid

import firebase_admin
import google.auth
import google.auth.transport.requests
import pillow_heif
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from firebase_admin import auth as fb_auth
from google.api_core.exceptions import NotFound, PreconditionFailed
from google.cloud import storage
from google.oauth2 import id_token
from PIL import Image, ImageOps, UnidentifiedImageError

# iPhone が既定で送る HEIC/HEIF を Pillow で開けるようにする。
# import しただけでは有効にならず、この登録が要る
pillow_heif.register_heif_opener()

# --- 設定 ---------------------------------------------------------------

OUTPUTS_BUCKET = os.environ["OUTPUTS_BUCKET"]
PROJECT_ID = os.environ["PROJECT_ID"]
REGION = os.environ["REGION"]
JOB_NAME = os.environ["JOB_NAME"]

# この時間を過ぎても状態が動かないジョブは、execution を見に行って
# 死んでいれば failed にする。OOM の signal 9 ではジョブ自身が
# 何も書けないため、これが無いと永遠に running のままになる
STALE_AFTER = datetime.timedelta(minutes=15)

# 限定公開中は許可リストに載っている uid だけを通す。
# 製品版では false にする。変わるのはこのフラグと Cloud Run の入口設定だけで、
# コードの作り直しは発生しない
ENFORCE_ALLOWLIST = os.environ.get("ENFORCE_ALLOWLIST", "true").lower() == "true"

DAILY_LIMIT = int(os.environ.get("DAILY_LIMIT", "10"))

# 失敗は回数に数えないが、起動そのものには上限を置く。
# 無いと、生成に向かない画像で延々と再試行でき、そのたびGPUが起動する
DAILY_ATTEMPT_LIMIT = int(os.environ.get("DAILY_ATTEMPT_LIMIT", "20"))

# GPU の同時実行数は、利用者単位ではなくプロジェクト全体で制限する。
# L4 の初期割り当ては3枚だが、障害対応用に1枚の余白を残して2本にする。
MAX_RUNNING_JOBS = int(os.environ.get("MAX_RUNNING_JOBS", "2"))
MAX_QUEUED_JOBS_PER_UID = int(os.environ.get("MAX_QUEUED_JOBS_PER_UID", "5"))

# 許可リストは GCS に置く。uid を足すたびに再デプロイしないため。
# 毎リクエスト読むと待ち時間が延びるので短時間だけ覚えておく
# 生成物バケットとは別のバケットに置く。API はこちらに読み取り権限しか持たない。
# 同居させると、門番を門番自身が書き換えられる状態になる
CONFIG_BUCKET = os.environ["CONFIG_BUCKET"]
ALLOWLIST_PATH = "config/allowed_uids.json"
ALLOWLIST_TTL = datetime.timedelta(seconds=60)
_allowlist_cache: tuple[datetime.datetime, set] | None = None

# 枠を取った直後は status.json がまだ無い。その隙に別のリクエストが
# 「もう終わっている」と誤判定して枠を奪えてしまうため、取り立ての枠は
# 状態を問わず生きているものとして扱う。
# 逆に、枠を取った直後にインスタンスが死んだ場合はこの時間で解放される
SLOT_GRACE = datetime.timedelta(seconds=60)
DISPATCH_CLAIM_TTL = datetime.timedelta(seconds=60)
DISPATCHER_SERVICE_ACCOUNT = os.environ["DISPATCHER_SERVICE_ACCOUNT"]

# ワーカーの完了通知に使う ID トークンの aud。Cloud Run の URL を使いたいところ
# だが、サービスは自分の URL を Terraform から受け取れない（自己参照になる）ので、
# ワーカーと API で共有する固定の文字列にしてある
DISPATCH_AUDIENCE = os.environ["DISPATCH_AUDIENCE"]

# ブラウザから叩けるようにする（curl では要らないので、実装当初は抜けていた）。
# アプリは GitHub Pages、APIは Cloud Run と**別オリジン**なので、
# ここが無いとブラウザはリクエストを1本も通さない。
# しかも Authorization ヘッダ付きの multipart なので preflight(OPTIONS) が飛ぶ
ALLOWED_ORIGINS = [
    o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()
]

MAX_IMAGE_BYTES = 5 * 1024 * 1024

# Content-Type ではなく、実際にデコードできた形式で判定する。
# 拡張子や Content-Type は詐称できるうえ、実際に「PNG を名乗る壊れたファイル」が
# 検証をすり抜けて GPU を起動させ、10分後に失敗した実績がある（約39円の無駄）
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "HEIF", "HEIC", "MPO"}

# 生成パイプラインは長辺1024pxを前提にしている。クライアント側でも縮小するが、
# 携帯から直接上げられた場合に備えてサーバー側でも縮める
MAX_EDGE = 1024

# 署名付きURLの有効期限。長くするほど漏れたときの影響が伸びる
SIGNED_URL_TTL = datetime.timedelta(hours=1)

app = FastAPI()

# 許可するオリジンは列挙する。ワイルドカードにしない。
#
# allow_credentials は False のまま。認証は Cookie ではなく Authorization
# ヘッダの Bearer トークンで行っており、これは別オリジンのページからは
# 付けられない（他人の localStorage を読めないため）。True にすると
# ワイルドカードが使えなくなるうえ、Cookie を送る意図だと誤読される
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
    allow_credentials=False,
    max_age=3600,
)

_storage = storage.Client()

# ADC で初期化する。Cloud Run 上ではサービスアカウントの権限がそのまま使われる
firebase_admin.initialize_app()

# Cloud Run の管理APIを叩くための認証済みセッション。
# ジョブの起動と、死活の照会に使う
_credentials, _ = google.auth.default(
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)
_session = google.auth.transport.requests.AuthorizedSession(_credentials)

_JOB_BASE = (
    f"https://run.googleapis.com/v2/projects/{PROJECT_ID}"
    f"/locations/{REGION}/jobs/{JOB_NAME}"
)


def _start_job(job_id: str, slot: int) -> str:
    """ジョブを起動し、execution 名を返す。

    どの入力を処理するかは環境変数の上書きで伝える。ジョブ側は
    JOB_ID からマウント済みの /jobs/{jobId}/ を見る。
    """
    res = _session.post(
        f"{_JOB_BASE}:run",
        json={
            "overrides": {
                "containerOverrides": [
                    {
                        "env": [
                            {"name": "JOB_ID", "value": job_id},
                            {"name": "JOB_SLOT", "value": str(slot)},
                        ]
                    }
                ]
            }
        },
        timeout=30,
    )
    res.raise_for_status()
    # 返るのは長時間実行オペレーション。実際の execution 名はその中にある
    body = res.json()
    return body.get("metadata", {}).get("name") or body.get("name", "")


def _execution_is_dead(execution_name: str) -> bool:
    """execution が既に終わっているか。照会できないときは False（触らない）。"""
    if not execution_name:
        return False
    res = _session.get(f"https://run.googleapis.com/v2/{execution_name}", timeout=30)
    if res.status_code != 200:
        return False
    return bool(res.json().get("completionTime"))


def _now() -> str:
    """UTC の ISO8601。status.json のタイムスタンプはこの形で統一する。"""
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _bucket():
    return _storage.bucket(OUTPUTS_BUCKET)


# 採番は job_ + uuid4 の16桁。GCS のオブジェクト名は不透明な文字列で
# ".." を解決しないため実際にパス操作は成立しないが、その性質に依存せず入口で弾く
JOB_ID_PATTERN = re.compile(r"^job_[0-9a-f]{16}$")


def _validate_job_id(job_id: str) -> None:
    if not JOB_ID_PATTERN.match(job_id):
        # 存在しないものと同じ扱いにする。形式の違いを教えない
        raise HTTPException(status_code=404, detail="そのジョブは存在しない")


def _status_blob(job_id: str):
    return _bucket().blob(f"jobs/{job_id}/status.json")


def _write_status(job_id: str, status: dict) -> None:
    status["updatedAt"] = _now()
    _status_blob(job_id).upload_from_string(
        json.dumps(status, ensure_ascii=False), content_type="application/json"
    )


def _reconcile_if_stale(status: dict) -> dict:
    """しばらく動きの無いジョブについて、実際に死んでいないか確かめる。

    ジョブ側は失敗時に自分で failed を書くが、OOM の signal 9 では
    何も書けずに死ぬ。その取りこぼしをここで拾う。
    """
    if status["state"] not in ("queued", "running"):
        return status

    updated = datetime.datetime.fromisoformat(status["updatedAt"])
    if datetime.datetime.now(datetime.timezone.utc) - updated < STALE_AFTER:
        return status

    if not _execution_is_dead(status.get("executionName")):
        return status

    status["state"] = "failed"
    status["error"] = "ジョブが状態を残さずに終了した（OOM や強制終了の可能性）"
    _write_status(status["jobId"], status)
    return status


def _elapsed_seconds(since: str | None) -> int:
    """その時刻から何秒経ったか。

    引き算を**サーバー側でやる**のが肝。時刻そのものを返して端末に引かせると、
    端末の時計のずれがそのまま進捗のずれになる（数分ずれている端末は珍しくない）。
    """
    if not since:
        return 0
    try:
        started = datetime.datetime.fromisoformat(since)
    except ValueError:
        return 0
    delta = datetime.datetime.now(datetime.timezone.utc) - started
    return max(0, int(delta.total_seconds()))


def _signed_model_url(job_id: str) -> str:
    """成果物の署名付きURLを発行する。

    Cloud Run のサービスアカウントは秘密鍵を持たないため、
    generate_signed_url をそのまま呼ぶと
    「you need a private key to sign credentials」で失敗する。

    IAM の SignBlob API に署名を代行させることで回避する。そのために
    サービスアカウント自身に roles/iam.serviceAccountTokenCreator が要る
    （infra/api.tf で自分自身に付けている）。
    """
    # storage.Client() が持つ認証情報ではなく _credentials を使う。
    # 前者はストレージ用のスコープしか持たず、SignBlob を呼ぶと
    # 403 ACCESS_TOKEN_SCOPE_INSUFFICIENT になる。
    # _credentials は cloud-platform スコープで取得してある
    credentials = _credentials
    # token の有無ではなく valid を見る。トークンは存在していても期限切れがあり、
    # 有無だけで判定すると、インスタンスが1時間以上生きたあとで署名が失敗する。
    # service_account_email は refresh するまで埋まらないことがあるので併せて見る
    if not credentials.valid or not getattr(credentials, "service_account_email", None):
        credentials.refresh(google.auth.transport.requests.Request())

    return _bucket().blob(f"jobs/{job_id}/model.glb").generate_signed_url(
        version="v4",
        expiration=SIGNED_URL_TTL,
        method="GET",
        service_account_email=credentials.service_account_email,
        access_token=credentials.token,
    )


# パスが /health なのは、Cloud Run では /healthz が使えないため。
# Google Frontend が /healthz を横取りし、アプリに到達する前に
# HTML の404を返す（アプリ側で登録しても届かない）。
# SPEC.md は /healthz と書いているが、この環境では実現できない。
def _uid_from_token(authorization: str | None) -> str:
    """Authorization ヘッダの Firebase ID トークンを検証して uid を返す。"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization ヘッダが無い")
    try:
        decoded = fb_auth.verify_id_token(authorization[len("Bearer "):])
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"トークンが無効: {e}")
    return decoded["uid"]


def _allowed_uids() -> set:
    global _allowlist_cache
    now = datetime.datetime.now(datetime.timezone.utc)
    if _allowlist_cache and now - _allowlist_cache[0] < ALLOWLIST_TTL:
        return _allowlist_cache[1]
    blob = _storage.bucket(CONFIG_BUCKET).blob(ALLOWLIST_PATH)
    uids = set(json.loads(blob.download_as_text())) if blob.exists() else set()
    _allowlist_cache = (now, uids)
    return uids


def _check_allowed(uid: str) -> None:
    if ENFORCE_ALLOWLIST and uid not in _allowed_uids():
        raise HTTPException(status_code=403, detail="このアカウントはまだ利用できない")


def _read_slot(blob) -> tuple[dict, int | None]:
    """枠の中身と generation を返す。

    generation が None なら枠は存在しない。中身が読めないときは空の辞書を返し、
    握っている者が不明なものとして扱う。
    """
    try:
        blob.reload()
    except NotFound:
        return {}, None
    try:
        return json.loads(blob.download_as_text()), blob.generation
    except (ValueError, NotFound):
        return {}, blob.generation


def _slot_is_fresh(claimed_at: str | None, ttl: datetime.timedelta = SLOT_GRACE) -> bool:
    """直前に確保された枠か。途中状態を別の処理が奪わないための判定。"""
    if not claimed_at:
        return False
    try:
        claimed = datetime.datetime.fromisoformat(claimed_at)
    except ValueError:
        return False
    return datetime.datetime.now(datetime.timezone.utc) - claimed < ttl


# 待機中のジョブは、この目印を索引にして探す。
#
# status.json を全部読んで待機中を拾うと、バケットに溜まったジョブの数だけ
# 読み込みが増える。生成物は30日残るので、月500件なら毎回500個読むことになり、
# ポーリングのたびに数秒かかる。目印なら**実際に待っている数**しか見ない。
#
# 情報は全て名前に入れて中身は空にする。数えるだけなら一覧するだけで済み、
# ダウンロードが要らない。受付時刻を先頭に置くのは、名前順がそのまま
# 受付順になるようにするため。
_PENDING_ROOT = "queue/pending/"
_PENDING_SUFFIX = ".json"


def _pending_prefix(uid: str) -> str:
    return f"{_PENDING_ROOT}{uid}/"


def _pending_marker_blob(uid: str, created_at: str, job_id: str):
    return _bucket().blob(f"{_pending_prefix(uid)}{created_at}~{job_id}{_PENDING_SUFFIX}")


def _parse_pending_name(name: str) -> tuple[str, str, str] | None:
    """目印の名前から uid・受付時刻・ジョブIDを取り出す。読めなければ None。"""
    if not name.startswith(_PENDING_ROOT) or not name.endswith(_PENDING_SUFFIX):
        return None
    uid, _, tail = name[len(_PENDING_ROOT) : -len(_PENDING_SUFFIX)].partition("/")
    created_at, separator, job_id = tail.rpartition("~")
    if not uid or not separator or not job_id:
        return None
    return uid, created_at, job_id


def _drop_pending_marker(blob) -> None:
    """目印を消す。既に消えていても構わない。"""
    try:
        blob.delete()
    except NotFound:
        pass


def _still_queued(job_id: str) -> dict | None:
    """まだ起動を待っているジョブなら、その状態を返す。"""
    status, generation = _read_slot(_status_blob(job_id))
    if generation is None or status.get("state") != "queued" or status.get("executionName"):
        return None
    return status


def _reserve_queue_space(uid: str, created_at: str, job_id: str) -> None:
    """待機上限に収まることを確かめ、待機中の目印を置く。

    数え上げなので、同時に投稿されると上限を1件超えることがある。それでも
    GPU が余分に立つことはない。実行数を抑えているのは _claim_global_slot の
    条件付き書き込みで、こちらは待機列を1人に占有させないための制限だから。

    数えるついでに、起動済み・終了済みの取り残しを掃除する。これをしないと
    通知の取りこぼし1回ごとに枠が減り、最後はその利用者が永久に 429 になる。
    """
    live = 0
    for blob in _bucket().list_blobs(prefix=_pending_prefix(uid)):
        parsed = _parse_pending_name(blob.name)
        if not parsed:
            continue
        found_at, found_job_id = parsed[1], parsed[2]
        # 目印は status.json より先に置くので、その隙間では状態を読めない。
        # 取り立ての目印は生きているものとして扱う（掃除に消されないため）
        if _still_queued(found_job_id) or _slot_is_fresh(found_at):
            live += 1
        else:
            _drop_pending_marker(blob)

    if live >= MAX_QUEUED_JOBS_PER_UID:
        raise HTTPException(
            status_code=429,
            detail=f"同時に受け付けられる作成は{MAX_QUEUED_JOBS_PER_UID}件までです",
        )

    _pending_marker_blob(uid, created_at, job_id).upload_from_string(
        "", content_type="application/json"
    )


def _release_queue_space(status: dict) -> None:
    """起動済みまたは終了したジョブの目印を外す。"""
    _drop_pending_marker(
        _pending_marker_blob(status["uid"], status["createdAt"], status["jobId"])
    )


def _global_slot_blob(slot: int):
    return _bucket().blob(f"queue/slots/{slot}.json")


def _claim_global_slot(job_id: str) -> int | None:
    """空いている GPU 実行枠を1つだけ確保する。"""
    for slot in range(MAX_RUNNING_JOBS):
        blob = _global_slot_blob(slot)
        try:
            blob.upload_from_string(
                json.dumps({"jobId": job_id, "claimedAt": _now()}),
                content_type="application/json",
                if_generation_match=0,
            )
            return slot
        except PreconditionFailed:
            pass
    return None


def _release_global_slot(slot: int, job_id: str) -> None:
    """自分のジョブが保持する実行枠だけを解放する。"""
    blob = _global_slot_blob(slot)
    data, generation = _read_slot(blob)
    if generation is None or data.get("jobId") != job_id:
        return
    try:
        blob.delete(if_generation_match=generation)
    except (PreconditionFailed, NotFound):
        pass


def _oldest_queued_job() -> dict | None:
    """まだ起動していないジョブから、最も古いものを選ぶ。

    目印の名前だけを一覧して受付順に見る。中身を読むのは先頭の候補だけなので、
    溜まったジョブの数ではなく**実際に待っている数**しか触らない。
    """
    markers = []
    for blob in _bucket().list_blobs(prefix=_PENDING_ROOT):
        parsed = _parse_pending_name(blob.name)
        if parsed:
            markers.append((parsed[1], parsed[2], blob))
    markers.sort(key=lambda marker: marker[0])

    for created_at, job_id, blob in markers:
        status = _still_queued(job_id)
        if not status:
            if _slot_is_fresh(created_at):
                # status.json がまだ書かれていないだけ。次の機会に拾う
                continue
            # 起動済みか終了済みの取り残し。ここで掃除しておく
            _drop_pending_marker(blob)
            continue
        if _slot_is_fresh(status.get("dispatchClaimedAt"), DISPATCH_CLAIM_TTL):
            continue
        return status
    return None


def _claim_dispatch(job_id: str) -> dict | None:
    """同じ待機ジョブを複数の API インスタンスが起動しないよう予約する。"""
    blob = _status_blob(job_id)
    data, generation = _read_slot(blob)
    if generation is None or data.get("state") != "queued" or data.get("executionName"):
        return None
    if _slot_is_fresh(data.get("dispatchClaimedAt"), DISPATCH_CLAIM_TTL):
        return None
    data["dispatchClaimedAt"] = _now()
    try:
        blob.upload_from_string(
            json.dumps(data, ensure_ascii=False),
            content_type="application/json",
            if_generation_match=generation,
        )
    except PreconditionFailed:
        return None
    return data


def _clear_dispatch_claim(job_id: str, claimed_at: str) -> None:
    """GPU 枠を取れなかったジョブを、次回のディスパッチ対象へ戻す。"""
    blob = _status_blob(job_id)
    data, generation = _read_slot(blob)
    if generation is None or data.get("dispatchClaimedAt") != claimed_at:
        return
    data.pop("dispatchClaimedAt", None)
    try:
        blob.upload_from_string(
            json.dumps(data, ensure_ascii=False),
            content_type="application/json",
            if_generation_match=generation,
        )
    except PreconditionFailed:
        pass


def _record_execution(job_id: str, slot: int, execution_name: str) -> None:
    """起動済みの記録を残す。ワーカーの running 更新を上書きしない。"""
    blob = _status_blob(job_id)
    for _ in range(3):
        data, generation = _read_slot(blob)
        if generation is None:
            return
        data["executionName"] = execution_name
        data["slot"] = slot
        data.pop("dispatchClaimedAt", None)
        try:
            blob.upload_from_string(
                json.dumps(data, ensure_ascii=False),
                content_type="application/json",
                if_generation_match=generation,
            )
            return
        except PreconditionFailed:
            pass


def _record_slot_execution(slot: int, job_id: str, execution_name: str) -> None:
    """起動した execution を枠にも控える。

    status.json はワーカーも書くので、_record_execution は競合で書けないことが
    ある。そうなると execution 名がどこにも残らず、枠を回収する手掛かりが
    無くなって GPU が1本減ったままになる。枠はディスパッチ側しか書かないので、
    ここに控えておけば取りこぼしても後から死活を確かめられる。
    """
    blob = _global_slot_blob(slot)
    data, generation = _read_slot(blob)
    if generation is None or data.get("jobId") != job_id:
        return
    data["executionName"] = execution_name
    try:
        blob.upload_from_string(
            json.dumps(data),
            content_type="application/json",
            if_generation_match=generation,
        )
    except PreconditionFailed:
        pass


def _release_finished_global_slots() -> None:
    """終了したジョブが握ったままの枠を回収する。

    枠は全体で MAX_RUNNING_JOBS 本しかない。1本でも握られたままになると
    その分だけ同時実行が減り、全部塞がると**誰も生成できなくなる**ので、
    状態を素直に信じずに死活まで確かめる。

    取りこぼす経路が2つある。
      - OOM の signal 9 でワーカーが status を書けずに死ぬと running のまま残る
      - 生成物バケットは30日で消えるので、古い枠は status ごと消えている
    """
    for slot in range(MAX_RUNNING_JOBS):
        data, _ = _read_slot(_global_slot_blob(slot))
        job_id = data.get("jobId")
        if not job_id:
            continue

        status, generation = _read_slot(_status_blob(job_id))
        if generation is None:
            # status ごと無い。握らせ続ける理由がない
            _release_global_slot(slot, job_id)
            continue

        # status.json への記録に失敗していたら、枠に控えた分で補う。
        # これが無いと execution を照会できず、死活を判定できない
        if not status.get("executionName") and data.get("executionName"):
            status["executionName"] = data["executionName"]

        # running のまま止まっているものは、ここで死活を見て failed に倒す
        status = _reconcile_if_stale(status)
        if status["state"] in ("succeeded", "failed"):
            _release_global_slot(slot, job_id)


def _mark_start_failed(status: dict, slot: int, error: Exception) -> None:
    """Cloud Run Job を起動できなかった待機ジョブを終了扱いにする。"""
    _release_global_slot(slot, status["jobId"])
    _release_queue_space(status)
    _restore_quota(status["uid"])
    _write_status(
        status["jobId"],
        {
            **status,
            "state": "failed",
            "error": f"ジョブの起動に失敗: {error}",
            "quotaRestored": True,
        },
    )


# 1リクエストで起動を試みる回数の上限。枠は MAX_RUNNING_JOBS 本しか無いので
# 本来それで足りるが、起動に失敗した待機ジョブを畳む分だけ余裕を持たせる。
# 上限を置くのは、1件の通知が延々と待機列を捌き続けてタイムアウトしないため。
# 捌き残しは次の通知か次のポーリングで進む。
MAX_DISPATCH_PER_REQUEST = MAX_RUNNING_JOBS + 5


def _dispatch_queued_jobs() -> None:
    """空き GPU 枠を、古い待機ジョブから順に埋める。"""
    _release_finished_global_slots()
    for _ in range(MAX_DISPATCH_PER_REQUEST):
        candidate = _oldest_queued_job()
        if not candidate:
            return
        claimed = _claim_dispatch(candidate["jobId"])
        if not claimed:
            continue
        slot = _claim_global_slot(claimed["jobId"])
        if slot is None:
            # 枠が全部埋まっている。予約を戻して、空いたときに拾わせる
            _clear_dispatch_claim(claimed["jobId"], claimed["dispatchClaimedAt"])
            return
        try:
            execution_name = _start_job(claimed["jobId"], slot)
        except Exception as error:
            _mark_start_failed(claimed, slot, error)
            continue
        _record_slot_execution(slot, claimed["jobId"], execution_name)
        _record_execution(claimed["jobId"], slot, execution_name)
        _release_queue_space(claimed)


def _quota_blob(uid: str):
    """当日ぶんのカウンタ。日付が変わると別のファイルになり、自然に0から始まる。"""
    today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    return _bucket().blob(f"quota/{uid}/{today}.json")


def _read_quota(blob) -> tuple[dict, int]:
    """当日のカウンタの中身と generation を返す。まだ無ければ ({}, 0)。

    generation は条件付き書き込みに渡す。0 は「まだ無いときだけ書く」の意味になる。
    """
    if not blob.exists():
        return {}, 0
    blob.reload()
    try:
        return json.loads(blob.download_as_text()), blob.generation
    except ValueError:
        # 壊れていたら数え直す。多く数える側に倒れるので安全側
        return {}, blob.generation


def _consume_quota(uid: str) -> None:
    """当日の回数を1増やす。上限に達していれば 429。

    2つ数える。

      count    … 成功として数える回数。**失敗した生成はここから戻す**ので、
                 利用者はこちら都合の失敗で枠を失わない
      attempts … 起動した回数。**戻さない**

    attempts が要るのは、失敗が必ず戻るだけだと、生成に向かない画像で
    延々と再試行できてしまい、そのたびに GPU が起動するため。
    1回あたり約39円かかるので、失敗の繰り返しにも上限を置く。

    generation の指定で競合を検出する。同じ利用者が同時に投げても
    二重に数え落とさない。
    """
    blob = _quota_blob(uid)
    data, generation = _read_quota(blob)
    count = data.get("count", 0)
    attempts = data.get("attempts", 0)

    if count >= DAILY_LIMIT:
        raise HTTPException(
            status_code=429, detail=f"本日の上限（{DAILY_LIMIT}回）に達した"
        )
    if attempts >= DAILY_ATTEMPT_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"本日の試行回数の上限（{DAILY_ATTEMPT_LIMIT}回）に達した。"
                   f"失敗が続いている場合は、別の画像で試してほしい",
        )

    try:
        blob.upload_from_string(
            json.dumps({"count": count + 1, "attempts": attempts + 1}),
            content_type="application/json",
            if_generation_match=generation,
        )
    except PreconditionFailed:
        # 競合だけをここで扱う。通信障害などを 409 にすると原因を見誤る
        raise HTTPException(status_code=409, detail="同時に処理されたので、やり直してほしい")


def _restore_quota(uid: str) -> None:
    """失敗した生成を回数に数えない。

    count だけ戻し、attempts は戻さない（上の説明を参照）。
    """
    blob = _quota_blob(uid)
    data, generation = _read_quota(blob)
    count = data.get("count", 0)
    if count <= 0:
        return
    try:
        blob.upload_from_string(
            json.dumps({**data, "count": count - 1}),
            content_type="application/json",
            if_generation_match=generation,
        )
    except PreconditionFailed:
        # 同時に別のリクエストが数えた。戻せなかった1回分は諦める
        # （多く数えることはあっても、少なく数えることはない側に倒す）
        pass


def _restore_quota_if_failed(status: dict) -> dict:
    """失敗が確定したジョブの回数を戻す。

    失敗が分かるのは状態を見にきたときなので、ここで戻す。
    二重に戻さないよう、戻したことを状態に記録する。
    """
    if status["state"] != "failed" or status.get("quotaRestored"):
        return status
    _restore_quota(status["uid"])
    status["quotaRestored"] = True
    _write_status(status["jobId"], status)
    return status


def _check_dispatcher(authorization: str | None) -> None:
    """完了通知は GPU ワーカーのサービスアカウントだけに限定する。

    aud を照合するのは、そのサービスアカウントが**別の宛先に向けて**発行した
    ID トークンを、ここに転用されないようにするため。audience を渡さないと
    verify_oauth2_token は aud を検証しない。
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization ヘッダが無い")
    try:
        claims = id_token.verify_oauth2_token(
            authorization[len("Bearer "):],
            google.auth.transport.requests.Request(),
            audience=DISPATCH_AUDIENCE,
        )
    except Exception:
        raise HTTPException(status_code=401, detail="ワーカー認証が無効")
    if claims.get("email") != DISPATCHER_SERVICE_ACCOUNT or not claims.get("email_verified"):
        raise HTTPException(status_code=403, detail="ワーカー認証が許可されていない")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/internal/dispatch")
def dispatch(authorization: str | None = Header(default=None)):
    """GPU ワーカーの完了通知で、空いた枠に次の待機ジョブを割り当てる。"""
    _check_dispatcher(authorization)
    _dispatch_queued_jobs()
    return {"ok": True}


def _normalize_image(data: bytes) -> bytes:
    """受け取った画像を検証し、PNG に正規化して返す。

    ここで弾けなかった画像は GPU を10分間動かしたうえで失敗するので、
    受け付けの時点で確実に判定する。

    あわせて以下を吸収する。
      - iPhone の HEIC、Android の WebP
      - 写真の EXIF 回転（無視すると横倒しのまま生成される）
      - 携帯から直接上げられた大きすぎる画像
    """
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

    # 写真は EXIF に回転情報を持つ。これを反映しないと横倒しで生成される
    img = ImageOps.exif_transpose(img)

    # 透過を持つ画像はそのまま活かす（背景除去済みの素材が来ることがある）。
    # 持たないものは RGB に寄せる
    img = img.convert("RGBA" if img.mode in ("RGBA", "LA", "P") else "RGB")

    if max(img.size) > MAX_EDGE:
        img.thumbnail((MAX_EDGE, MAX_EDGE), Image.LANCZOS)

    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


@app.post("/jobs", status_code=202)
async def create_job(
    image: UploadFile = File(...), authorization: str | None = Header(default=None)
):
    uid = _uid_from_token(authorization)
    _check_allowed(uid)

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="画像が空")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"画像は {MAX_IMAGE_BYTES // (1024 * 1024)}MB まで。"
                   f"受け取ったのは {len(data) / (1024 * 1024):.1f}MB",
        )

    # GPU を起動する前にここで確実に弾く
    png = _normalize_image(data)

    job_id = f"job_{uuid.uuid4().hex[:16]}"
    now = _now()
    status = {
        "jobId": job_id,
        "uid": uid,
        "state": "queued",
        "createdAt": now,
        "updatedAt": now,
        "executionName": None,
        "error": None,
    }

    # 正当な画像だけを待機列に受け付ける。1利用者が待機枠を独占しないよう、
    # 待機上限もここで確かめる。
    queue_reserved = False
    quota_consumed = False
    try:
        _reserve_queue_space(uid, now, job_id)
        queue_reserved = True
        _consume_quota(uid)
        quota_consumed = True

        _bucket().blob(f"jobs/{job_id}/input.png").upload_from_string(
            png, content_type="image/png"
        )
        _write_status(job_id, status)

        # 空き GPU があればこのジョブを、なければ既存の先頭ジョブを起動する。
        # 一時的なディスパッチ失敗で受付は取り消さない。待機のまま残るので
        # 次の完了通知かポーリングで拾われる。ただし黙って捨てると、
        # ずっと queued のまま進まなくなったときに手掛かりが無くなる
        try:
            _dispatch_queued_jobs()
        except Exception:
            traceback.print_exc()
    except Exception:
        if quota_consumed:
            _restore_quota(uid)
        if queue_reserved:
            _release_queue_space(status)
        raise

    return {"jobId": job_id}


@app.get("/jobs/{job_id}")
def get_job(job_id: str, authorization: str | None = Header(default=None)):
    uid = _uid_from_token(authorization)
    _validate_job_id(job_id)
    blob = _status_blob(job_id)
    if not blob.exists():
        raise HTTPException(status_code=404, detail="そのジョブは存在しない")

    status = json.loads(blob.download_as_text())

    # 他人のジョブは存在自体を隠す（403 ではなく 404）
    if status.get("uid") != uid:
        raise HTTPException(status_code=404, detail="そのジョブは存在しない")

    status = _reconcile_if_stale(status)
    status = _restore_quota_if_failed(status)

    # OOM などでワーカー自身が完了通知を送れなかった場合も、状態を見に来た
    # タイミングで終了済み枠を回収し、次の待機ジョブを進める。
    if status["state"] in ("succeeded", "failed"):
        _dispatch_queued_jobs()

    body = {
        "state": status["state"],
        "createdAt": status["createdAt"],
    }
    # いまの工程。ジョブ側が書けたときだけ入る。
    # 書けていない場合でも state だけで画面は成立するので、必須にはしない
    phase = status.get("phase")
    if phase:
        body["phase"] = phase
        # 経過秒は動いている間だけ返す。終わったジョブにも返すと、
        # 最後の工程に入ってからの時間が取得のたびに伸び続け、
        # 「テクスチャ生成に3日かかっている」ような表示になる。
        # phase 自体は failed の原因を絞るのに使えるので残す
        if status["state"] == "running":
            body["phaseElapsedSeconds"] = _elapsed_seconds(status.get("phaseStartedAt"))
    if status["state"] == "succeeded":
        body["modelUrl"] = _signed_model_url(job_id)
    if status.get("error"):
        body["error"] = status["error"]
    return body
