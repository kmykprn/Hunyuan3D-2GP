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

import collections
import datetime
import glob
import json
import os
import subprocess
import sys
import threading
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


# 生成の工程。minimal_demo_mmgp.py が標準出力に出す印と、状態に書く名前の対。
#
# **前から順に1つずつしか探さない。** 全行を全印と照合すると、ログに
# たまたま同じ文字列が現れたときに工程が巻き戻る。生成は一本道なので、
# 次に来るはずの印だけを待てばよい。
PHASE_MARKERS: tuple[tuple[str, str], ...] = (
    ("=== Loading texture generation model ===", "loading_texture_model"),
    ("=== Loading i23d model ===", "loading_shape_model"),
    ("Generating 3D model with texture...", "generating_shape"),
    ("Generating texture...", "generating_texture"),
    ("3D model with texture generated successfully!", "finishing"),
)

# 標準エラーのうち手元に残す行数。失敗の理由に使うのは末尾だけで、
# 全文は素通し済みなので Cloud Logging にある
STDERR_KEEP_LINES = 200

# 残す1行の長さの上限。tqdm の進捗表示は改行なしで伸び続けるため、
# 行数だけで抑えても1行が巨大になりうる
STDERR_KEEP_CHARS = 2000


def _set_phase(job_id: str, phase: str) -> None:
    """いまの工程を状態に書く。

    書けなくても生成は続ける。1回およそ40円かかる処理を、
    進捗表示のために落とすのは割に合わない。
    """
    try:
        _update_status(job_id, phase=phase, phaseStartedAt=_now())
    except Exception:
        traceback.print_exc()


def _drain_stderr(stream, keep: collections.deque) -> None:
    """標準エラーを読み続け、素通ししつつ末尾だけ残す。

    別スレッドにするのは**デッドロックを避けるため**。標準出力を1行ずつ
    読んでいる間に標準エラー側のパイプ（64KB）が埋まると、子は書き込みで
    止まり、親は標準出力の続きを待ち続けて、両者が動かなくなる。

    全文を溜めないのは 16GiB しかないメモリを守るため。従来の
    subprocess.run(stderr=PIPE) は全文をメモリに持っていたので、
    ここは以前より軽くなっている。
    """
    for line in stream:
        sys.stderr.write(line)
        keep.append(line.rstrip()[:STDERR_KEEP_CHARS])
    sys.stderr.flush()


def main() -> int:
    job_id = os.environ.get("JOB_ID")
    if not job_id:
        print("JOB_ID が渡されていない", file=sys.stderr)
        return 1

    input_path = f"/tmp/{job_id}_input.png"

    try:
        _update_status(job_id, state="running", phase="preparing", phaseStartedAt=_now())
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
        # グローバル状態を持っており、import して呼ぶと副作用が読みにくいため。
        #
        # 標準出力を捕まえるのは、工程の切り替わりを拾って status.json に
        # 書くため。**1行読んだらその場で素通しし、溜めない。** 溜めると
        # 親が全出力をメモリに持つことになり、余裕の無いメモリを圧迫する
        # （実測で OOM の一因になった）
        process = subprocess.Popen(
            [
                sys.executable,
                # 出力を行ごとに流させる。既定のままだと標準出力がパイプ相手に
                # ブロックバッファされ、工程の印が数KB溜まるまで届かない
                "-u",
                "minimal_demo_mmgp.py",
                "--input-image", input_path,
                "--output", "/tmp/output",
                "--texture",
                "--profile", "3",
            ],
            cwd=REPO_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        stderr_tail: collections.deque = collections.deque(maxlen=STDERR_KEEP_LINES)
        stderr_reader = threading.Thread(
            target=_drain_stderr, args=(process.stderr, stderr_tail), daemon=True
        )
        stderr_reader.start()

        next_marker = 0
        for line in process.stdout:
            # まず素通し。Cloud Logging に出る内容は今までと変わらない
            sys.stdout.write(line)
            sys.stdout.flush()
            if next_marker < len(PHASE_MARKERS):
                marker, phase = PHASE_MARKERS[next_marker]
                if marker in line:
                    _set_phase(job_id, phase)
                    next_marker += 1

        process.stdout.close()
        returncode = process.wait()
        # 子が終わればパイプは閉じるので、読み手もすぐ抜ける。
        # 万一抜けなければ待ち続けても仕方がないので打ち切る
        stderr_reader.join(timeout=30)

        if returncode != 0:
            if returncode == -9:
                # OOM による SIGKILL。この落ち方では stderr も残らない
                raise RuntimeError("メモリ不足で強制終了された")
            # 最後の例外行だけを理由にする。全文は Cloud Logging にある
            lines = [ln for ln in stderr_tail if ln and not ln.startswith(" ")]
            detail = lines[-1] if lines else "詳細はログを参照"
            raise RuntimeError(f"生成に失敗: {detail}")

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
