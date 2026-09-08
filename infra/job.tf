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

        volume_mounts {
          name       = "weights"
          mount_path = "/models"
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
