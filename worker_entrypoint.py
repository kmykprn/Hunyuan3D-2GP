# Cloud Run Job の入口。API から JOB_ID を渡されて1件だけ生成する。
#
# 入出力は GCS API で直接読み書きする。
#
# 当初は生成物バケットを /jobs に gcsfuse でマウントする設計だったが、
# 重みのマウントと合わせて2つになった時点で OOM した。16GiB のうち
# 4GiB をファイルキャッシュが占めており、マウント1つ分の余裕しか
# 残っていなかった。キャッシュは unet(3.41GB)が収まる大きさが要るので
# ほとんど削れない。
#
# 書き込むのは status.json（数百バイト）と model.glb（4.8MB）だけなので、
# FUSE の書き込みセマンティクスに頼る必要もない。
#
# 重要なのは「失敗を必ず status.json に書いてから死ぬ」こと。
# 既存の api_server.py の /status はファイルの有無しか見ないため、
# 失敗すると永遠に processing を返す。同じ作りにしない。
#
# ただし OOM の signal 9 のように、ここで何も書けずに死ぬ落ち方も実在する
# （PR #6 の実測で踏んだ）。そちらは API 層の失敗検知が拾う。

import datetime
import glob
import json
import os
import subprocess
import sys
import traceback

from google.cloud import storage

REPO_DIR = "/app"
OUTPUTS_BUCKET = os.environ["OUTPUTS_BUCKET"]

_storage = storage.Client()
_bucket = _storage.bucket(OUTPUTS_BUCKET)

# minimal_demo_mmgp.py の --cache-path は効かない。
# 同スクリプトは自分の名前空間の SAVE_DIR を書き換えるが、実際に参照されるのは
# hunyuan3d_mmpg.utils 側の SAVE_DIR で、そちらは既定の "gradio_cache" のまま。
# 実行のたびに新しいコンテナなので、ここには常に1件だけ出来る
CACHE_GLOB = os.path.join(REPO_DIR, "gradio_cache", "*", "textured_mesh.glb")


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _update_status(job_id: str, **changes) -> None:
    """status.json を読んで変更を反映して書き戻す。

    API 層が書いた createdAt や uid を消さないよう、全体を上書きせず
    読んでからマージする。
    """
    blob = _bucket.blob(f"jobs/{job_id}/status.json")
    status = json.loads(blob.download_as_text())
    status.update(changes)
    status["updatedAt"] = _now()
    blob.upload_from_string(
        json.dumps(status, ensure_ascii=False), content_type="application/json"
    )


def main() -> int:
    job_id = os.environ.get("JOB_ID")
    if not job_id:
        print("JOB_ID が渡されていない", file=sys.stderr)
        return 1

    input_path = f"/tmp/{job_id}_input.png"

    try:
        _update_status(job_id, state="running")
    except Exception:
        # ここで失敗すると状態を一切更新できないので、諦めて落ちる。
        # API 層の失敗検知が拾う
        traceback.print_exc()
        return 1

    try:
        src = _bucket.blob(f"jobs/{job_id}/input.png")
        if not src.exists():
            raise FileNotFoundError(f"入力画像が無い: jobs/{job_id}/input.png")
        src.download_to_filename(input_path)

        # 別プロセスにするのは、minimal_demo_mmgp.py が終了時フックや
        # グローバル状態を持っており、import して呼ぶと副作用が読みにくいため
        result = subprocess.run(
            [
                sys.executable,
                "minimal_demo_mmgp.py",
                "--input-image", input_path,
                "--output", "/tmp/output",
                "--texture",
                "--profile", "3",
            ],
            cwd=REPO_DIR,
            # 出力は捕まえずに素通しする。捕まえると親が全部メモリに溜め、
            # ただでさえ余裕の無いメモリを圧迫する。
            # ログは素通しでも Cloud Logging に残る
        )
        if result.returncode != 0:
            # 詳細は Cloud Logging 側にある。-9 は OOM による SIGKILL
            reason = "OOM で強制終了された可能性が高い" if result.returncode == -9 else "詳細はログを参照"
            raise RuntimeError(
                f"生成が終了コード {result.returncode} で失敗（{reason}）"
            )

        produced = glob.glob(CACHE_GLOB)
        if not produced:
            # テクスチャ生成が無効なまま完走すると white_mesh しか出ない。
            # PR #2 で潰した「失敗しているのに成功と報告する」経路がこれ
            raise RuntimeError(
                "textured_mesh.glb が出力されていない。"
                "テクスチャ生成が無効になっていないか、ログを確認すること"
            )
        if len(produced) > 1:
            raise RuntimeError(f"出力が複数ある: {produced}")

        _bucket.blob(f"jobs/{job_id}/model.glb").upload_from_filename(produced[0])
        _update_status(job_id, state="succeeded", error=None)
        print(f"完了: {job_id}")
        return 0

    except Exception as e:
        traceback.print_exc()
        try:
            _update_status(job_id, state="failed", error=str(e)[:500])
        except Exception:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
