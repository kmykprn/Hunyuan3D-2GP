# 実測用の Cloud Run Job。
#
# なぜサービスではなくジョブなのか:
#   目的は「コールドスタートと L4 での生成時間を測ること」であって、
#   リクエストを受けることではない。イメージの CMD が既に CLI
#   （minimal_demo_mmgp.py）なので、**HTTPサーバを1行も書かずにそのまま動く**。
#   ジョブは常駐しないので、消し忘れによる課金の垂れ流しも起きない。

resource "google_service_account" "job" {
  account_id   = "hunyuan3d-job"
  display_name = "Hunyuan3D 実測ジョブ"

  # iam.googleapis.com の有効化を待つ。これが無いと新規プロジェクトへの
  # 初回 apply で、API有効化と並行して作成が走り accessNotConfigured で落ちる。
  # 2回目の apply では通ってしまうので、一度動いた環境では気づけない
  depends_on = [google_project_service.required]
}

# 重みの読み出しと生成物の書き込みだけを許可する。
# 権限は必要な2つのバケットに絞り、プロジェクト全体には与えない
resource "google_storage_bucket_iam_member" "job_reads_weights" {
  bucket = google_storage_bucket.weights.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.job.email}"
}

resource "google_storage_bucket_iam_member" "job_writes_outputs" {
  bucket = google_storage_bucket.outputs.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.job.email}"
}

# イメージをまだ push していない段階でも apply できるようにする。
# 先にレジストリとバケットを作り、そこへ push してから image を指定して
# apply し直す、という順序を踏めるようにするため
resource "google_cloud_run_v2_job" "measure" {
  count = var.image == "" ? 0 : 1

  name     = "hunyuan3d-measure"
  location = var.region

  # GPU を使うジョブは起動が遅く、既定の待ち方だと失敗扱いになりうる
  deletion_protection = false

  template {
    template {
      service_account = google_service_account.job.email

      # 外向きの通信を VPC 経由にする（network.tf）。重みのダウンロードが速くなる
      vpc_access {
        network_interfaces {
          network    = google_compute_network.jobs.name
          subnetwork = google_compute_subnetwork.jobs.name
        }
        egress = "ALL_TRAFFIC"
      }

      # 失敗したジョブを黙って再実行させない。
      # GPU は秒課金なので、リトライは費用がそのまま倍になる。
      # 失敗したら理由を見てから手で回す
      max_retries = 0

      timeout = "${var.job_timeout_seconds}s"

      # ゾーン冗長を切る。初回に自動付与されるクォータは
      # 「zonal redundancy off」の3枚なので、有効にすると
      # クォータ不足でデプロイが弾かれる。費用も冗長のほうが高い
      gpu_zonal_redundancy_disabled = true

      node_selector {
        accelerator = var.gpu_type
      }

      # 重みは gcsfuse でマウントしない。起動時にワーカーが GCS API で並列ダウンロードして
      # /tmp（メモリ）に置き、そこから読む（worker_entrypoint.py の _fetch_weights）。
      #
      # 以前は gcsfuse + メモリ上のファイルキャッシュ 4GiB で読んでいたが、実効 50MB/s で
      # 読み込みに 263 秒（1 件 522 秒の半分）かかっていた。公式も「FUSE は初回ダウンロードを
      # 並列化しないので大きな重みでは gcloud storage cp より遅い」としている。
      # 重みを fp16 にそろえて 7GB にしたので、/tmp に置いても 32GiB に収まる
      containers {
        image = var.image

        # 重みの置き場（ダウンロード先）。HF_HOME の hub/ 配下に、バケットの hub-fp16/ と
        # 同じ構造（~/.cache/huggingface/hub と同じ）で置く。hy3dgen も diffusers もそこを探す
        env {
          name  = "HF_HOME"
          value = "/tmp/models"
        }
        env {
          name  = "WEIGHTS_BUCKET"
          value = google_storage_bucket.weights.name
        }
        env {
          name  = "WEIGHTS_PREFIX"
          value = "hub-fp16"
        }
        # rembg が依存する pymatting は import 時に numba で JIT コンパイルし、コンテナを
        # 立てるたびに 23 秒かかっていた（実測。他の import は合計 3 秒）。JIT を切ると 0.5 秒。
        # コンパイルされる関数はアルファマッティング用で、この生成では使わない。
        # 3D 生成のコードは numba を使っていないので、切っても速さは変わらない
        env {
          name  = "NUMBA_DISABLE_JIT"
          value = "1"
        }

        # trust_remote_code でカスタムパイプラインを読むとき、diffusers は
        # コードを HF_HOME/modules に書き出す。gcsfuse で読み取り専用に
        # マウントしていた頃は「[Errno 30] Read-only file system: '/models/modules'」で
        # テクスチャ生成モデルのロードが失敗した。いまは /tmp なので書けるが、
        # 重みと混ぜないために別の場所のまま残している。
        #
        # refs/main への書き込み失敗は Ignored error として無視されるが、
        # modules は無視されない。ローカル実行では ~/.cache/huggingface が
        # 書き込み可能なため顕在化しない。
        env {
          name  = "HF_MODULES_CACHE"
          value = "/tmp/hf_modules"
        }

        # 生成物と状態の置き場。gcsfuse でマウントせず GCS API で読み書きする。
        # 上限が 16GiB だった頃、マウントを2つにすると超えて OOM した
        # （キャッシュが4GiBを占めており、マウント1つ分の余裕しか無かった）。
        # 32GiB にした今も、マウントを増やす理由が無いのでこのままにしてある
        env {
          name  = "OUTPUTS_BUCKET"
          value = google_storage_bucket.outputs.name
        }

        # 完了時に API へ通知し、空いた枠で次の待機ジョブを始める。
        #
        # **API サービスは count 付き**（api_image が空なら作られない）なので、
        # そのまま [0] を書くと Invalid index で apply ごと落ちる。
        # イメージを push する前に apply する手順が variables.tf に書いてあり、
        # そこを通れない。空を渡せばワーカー側が通知を諦めるだけで済むので、
        # サービスが無いときは空にする（枠の回収は get_job 側の経路が拾う）
        env {
          name = "API_URL"
          value = (
            length(google_cloud_run_v2_service.api) > 0
            ? google_cloud_run_v2_service.api[0].uri
            : ""
          )
        }

        env {
          name  = "DISPATCH_AUDIENCE"
          value = local.dispatch_audience
        }



        # GPU ごとに下限が決まっている。
        #   nvidia-l4            : 4 CPU / 16GiB 以上
        #   nvidia-rtx-pro-6000  : 20 CPU / 80GiB 以上
        # 下限を割ると deploy 時に弾かれる
        resources {
          limits = {
            cpu              = var.job_cpu
            memory           = var.job_memory
            "nvidia.com/gpu" = "1"
          }
        }

        # 入力画像は API が GCS に置き、JOB_ID で場所を伝える。
        # 固定の画像を生成していた計測用の指定は worker_entrypoint.py に移した
        args = ["python", "worker_entrypoint.py"]
      }
    }
  }

  depends_on = [google_project_service.required]
}
