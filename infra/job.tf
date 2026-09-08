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
        accelerator = "nvidia-l4"
      }

      # 重みをバケットから読む。gcsfuse でマウントするので、
      # イメージにもコードにも手を入れずに済むのが利点。
      #
      # ただし公式ガイドは「FUSE は初回ダウンロードを並列化しないので、
      # 大きな重みでは gcloud storage cp より遅い」としている。
      # ここで測った値が、並列ダウンロード方式と比較する際の基準になる
      volumes {
        name = "weights"
        gcs {
          bucket    = google_storage_bucket.weights.name
          read_only = true

          # ファイルキャッシュを有効にする。既定では無効で、その状態だと
          # diffusers の from_pretrained が致命的に遅い。
          # 実測ではテクスチャモデルのロードに25分40秒かかり（約2.2MB/s）、
          # 30分のタイムアウトに達して生成まで到達しなかった。
          #
          # 原因は safetensors の mmap で、ページフォルトが1回ずつ
          # GCS へのレンジリクエストになること。gcsfuse からは
          # ランダムアクセスに見えるため先読みが効かない。
          # 同じ症状が gcsfuse#2828 と diffusers#10280 に報告されている。
          #
          # 対比として、形状モデル(7.2GB の単一 safetensors)は同じマウントを
          # 通して67秒(110MB/s)で読めている。gcsfuse が一律に遅いのではなく、
          # diffusers の読み方が問題という切り分けになる。
          mount_options = [
            # ファイルキャッシュは cache-dir を指定して初めて有効になる。
            # サイズ上限だけ書いても有効化されず、gcsfuse が
            # 「file cache should be enabled for parallel download support」
            # で起動に失敗する。
            # Cloud Run では in-memory ボリュームを cr-volume:{名前} で参照する
            "cache-dir=cr-volume:cache",
            # このキャッシュは各ファイルを1回しか読まないので、再利用のためでは
            # なく「gcsfuse に並列ダウンロードさせる」ために置いている。
            # よって必要なのは最大の単一ファイル(unet 3.41GB)が収まる大きさだけ。
            # 6144 にしたところ、アプリ側と合わせて 16GiB を超えて OOM で
            # 落ちた（テクスチャ側は paint と delight の2本がロードされる）
            "file-cache-max-size-mb=4096",
            # 大きなファイルの初回読み込みを並列化する
            "file-cache-enable-parallel-downloads=true",
            # 重みは実行中に変わらないので、メタデータは無期限にキャッシュしてよい
            "metadata-cache-ttl-secs=-1",
          ]
        }
      }

      # 上の cache-dir が指す実体。Cloud Run にローカルディスクは無いので
      # メモリ上に置くしかなく、その分はコンテナのメモリ上限(16GiB)に
      # カウントされる。4GiB 使うと、アプリ側に残るのは約12GiB。
      # 6GiB にしたときは OOM(signal 9)で落ちた
      volumes {
        name = "cache"
        empty_dir {
          medium     = "MEMORY"
          size_limit = "4Gi"
        }
      }

      containers {
        image = var.image

        # HF_HOME 配下の hub/ をそのまま使うので、
        # ~/.cache/huggingface を丸ごとバケットに上げておけばよい
        env {
          name  = "HF_HOME"
          value = "/models"
        }

        # trust_remote_code でカスタムパイプラインを読むとき、diffusers は
        # コードを HF_HOME/modules に書き出す。HF_HOME はバケットの
        # 読み取り専用マウントなので書けず、
        # 「[Errno 30] Read-only file system: '/models/modules'」で
        # テクスチャ生成モデルのロードが失敗する。
        #
        # refs/main への書き込み失敗は Ignored error として無視されるが、
        # modules は無視されない。ローカル実行では ~/.cache/huggingface が
        # 書き込み可能なため顕在化しない。
        env {
          name  = "HF_MODULES_CACHE"
          value = "/tmp/hf_modules"
        }

        volume_mounts {
          name       = "weights"
          mount_path = "/models"
        }

        # gcsfuse のキャッシュ実体。cache-dir から cr-volume: で参照するだけでなく、
        # in-memory ボリュームはコンテナにマウントする必要がある
        volume_mounts {
          name       = "cache"
          mount_path = "/gcsfuse-cache"
        }

        # L4 を使う場合、4vCPU / 16GiB が下限として要求される
        resources {
          limits = {
            cpu              = "4"
            memory           = "16Gi"
            "nvidia.com/gpu" = "1"
          }
        }

        args = [
          "python", "minimal_demo_mmgp.py",
          "--input-image", "assets/example_images/052.png",
          "--output", "/tmp/output",
          "--texture",
          "--profile", "3",
        ]
      }
    }
  }

  depends_on = [google_project_service.required]
}
