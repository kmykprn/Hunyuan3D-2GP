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


def _start_job(job_id: str) -> str:
    """ジョブを起動し、execution 名を返す。

    どの入力を処理するかは環境変数の上書きで伝える。ジョブ側は
    JOB_ID からマウント済みの /jobs/{jobId}/ を見る。
    """
    res = _session.post(
        f"{_JOB_BASE}:run",
        json={
            "overrides": {
                "containerOverrides": [
                    {"env": [{"name": "JOB_ID", "value": job_id}]}
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


def _active_blob(uid: str):
    return _bucket().blob(f"quota/{uid}/active.json")


def _job_is_running(job_id: str | None) -> bool:
    """そのジョブがまだ動いているか。状態が読めないものは動いていない扱い。"""
    if not job_id:
        return False
    blob = _status_blob(job_id)
    if not blob.exists():
        return False
    return json.loads(blob.download_as_text())["state"] in ("queued", "running")


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


def _slot_is_fresh(claimed_at: str | None) -> bool:
    """取り立ての枠か。status.json が書かれる前の隙間を守るための判定。"""
    if not claimed_at:
        return False
    try:
        claimed = datetime.datetime.fromisoformat(claimed_at)
    except ValueError:
        return False
    return datetime.datetime.now(datetime.timezone.utc) - claimed < SLOT_GRACE


def _claim_slot(uid: str, job_id: str) -> int:
    """この利用者の実行枠を確保する。埋まっていれば 409。

    並行実行を許すとGPUが同時に立ち上がり、事故的に高くつく。

    「存在を確かめてから書く」ではなく、**GCS の条件付き書き込みで確保する**。
    if_generation_match=0 は「まだ無いときだけ作る」で、これが排他ロックになる。
    確認と書き込みを分けると、その間に通った2件目が同時にGPUを立ち上げる。
    ジョブの起動には数秒かかるので窓が広く、ボタンの二度押しで踏む。
    """
    blob = _active_blob(uid)
    payload = json.dumps({"jobId": job_id, "claimedAt": _now()})

    # 取れなかった理由は「まだ動いている」か「終わったのに枠が残っている」の
    # どちらか。後者なら1度だけ解放して取り直す（無限には繰り返さない）
    for attempt in (1, 2):
        try:
            blob.upload_from_string(
                payload, content_type="application/json", if_generation_match=0
            )
            # 解放するときに「自分が取った枠か」を確かめるために覚えておく
            return blob.generation
        except PreconditionFailed:
            pass

        slot, generation = _read_slot(blob)
        if generation is None:
            # 読みに行く間に解放された。空いたので取り直す
            continue

        holder = slot.get("jobId")
        # 取り立ての枠は、status.json がまだ無くても生きているものとして扱う
        if _slot_is_fresh(slot.get("claimedAt")) or (holder and _job_is_running(holder)):
            raise HTTPException(
                status_code=409, detail=f"生成中のジョブがある: {holder}"
            )

        if attempt == 1:
            # 終わったジョブが枠を握ったままなので解放する。
            # generation を指定するので、その隙に他が取っていれば空振りする
            try:
                blob.delete(if_generation_match=generation)
            except (PreconditionFailed, NotFound):
                pass

    raise HTTPException(
        status_code=409, detail="実行枠を確保できなかった。少し待ってやり直してほしい"
    )


def _release_slot(uid: str, generation: int) -> None:
    """自分が取った実行枠を解放する。

    generation を指定するのは、既に別のリクエストが取り直した枠を
    誤って消さないため。解放しきれなくても、次の投稿が終了済みと
    判断して解放するので詰まりはしない。
    """
    try:
        _active_blob(uid).delete(if_generation_match=generation)
    except (PreconditionFailed, NotFound):
        pass


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


@app.get("/health")
def health():
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

    # 画像が正当だと分かってから枠を取る。壊れた画像で枠を失わせない。
    # 確保はジョブの起動より先に済ませる。あとに回すと、起動にかかる数秒の間に
    # 2件目が通ってGPUが2本立つ
    slot_generation = _claim_slot(uid, job_id)
    try:
        _consume_quota(uid)

        _bucket().blob(f"jobs/{job_id}/input.png").upload_from_string(
            png, content_type="image/png"
        )

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
        _write_status(job_id, status)

        try:
            execution_name = _start_job(job_id)
        except Exception as e:
            # 起動できなかったことを状態に残す。残さないと queued のまま放置される
            _write_status(
                job_id, {**status, "state": "failed", "error": f"ジョブの起動に失敗: {e}"}
            )
            # 起動していないので枠を返す。生成できていないのに1回分を失わせない
            _restore_quota(uid)
            raise HTTPException(status_code=502, detail=f"ジョブの起動に失敗: {e}")

        _write_status(job_id, {**status, "executionName": execution_name})
    except Exception:
        # ここまでに失敗したなら、そのジョブは動いていない。
        # 枠を握ったままにすると、次の投稿まで利用者が締め出される
        _release_slot(uid, slot_generation)
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

    body = {
        "state": status["state"],
        "createdAt": status["createdAt"],
    }
    if status["state"] == "succeeded":
        body["modelUrl"] = _signed_model_url(job_id)
    if status.get("error"):
        body["error"] = status["error"]
    return body
