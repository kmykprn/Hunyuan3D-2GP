# 切り抜きサービス（cutout/）。写真から家具だけを切り抜いて透過 PNG を返す。
#
# 3D 生成の API 層とは別のサービスにする。あちらは 1 CPU / 512Mi の薄い受付で、
# こちらは推論のために 4 CPU / 12Gi 要る（実測 6.4GB）。同居させると
# 受付側のコールドスタートまで太る。3D をやめてもこちらは単独で残せる。
#
# 費用は 1 枚あたり約 0.1 円（4 vCPU × 6 秒）。無料枠（月 18 万 vCPU 秒）に収まる。

# 専用のサービスアカウント。持つのは回数を数えるバケットの読み書きだけ。
# 生成物バケットにも許可リストにも触れない
resource "google_service_account" "cutout" {
  account_id   = "cutout"
  display_name = "切り抜きサービス"

  depends_on = [google_project_service.required]
}

# 1 日の回数を数えるファイルの置き場。数 KB × 利用者数しか無い
resource "google_storage_bucket" "cutout_state" {
  name     = "${var.project_id}-cutout-state"
  location = upper(var.region)

  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  # カウンタは日付ごとに別ファイルなので、古い日のものは消してよい
  lifecycle_rule {
    condition { age = 7 }
    action { type = "Delete" }
  }

  depends_on = [google_project_service.required]
}

resource "google_storage_bucket_iam_member" "cutout_rw_state" {
  bucket = google_storage_bucket.cutout_state.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.cutout.email}"
}

resource "google_cloud_run_v2_service" "cutout" {
  count = var.cutout_image == "" ? 0 : 1

  name     = "cutout"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.cutout.email

    # 1 件 6 秒。60 秒あれば十分で、それ以上は何かが壊れている
    timeout = "60s"

    # 推論は 1 件で 6.4GB 使う。2 件同時に受けるとメモリが足りない。
    # 並列は台数で取る（max_instance_count）
    max_instance_request_concurrency = 1

    scaling {
      min_instance_count = 0
      # 上限は費用の頭打ち。2 台が 24 時間動き続けても月 7 万円で止まる
      max_instance_count = 2
    }

    containers {
      image = var.cutout_image

      env {
        name  = "STATE_BUCKET"
        value = google_storage_bucket.cutout_state.name
      }
      env {
        name  = "DAILY_LIMIT"
        value = tostring(var.cutout_daily_limit)
      }
      env {
        name  = "ALLOWED_ORIGINS"
        value = join(",", var.allowed_origins)
      }

      resources {
        limits = {
          # 4 vCPU で 1 枚 5 秒、2 vCPU なら 8 秒。費用は同じ（vCPU × 秒）なので速いほうを取る
          cpu = "4"
          # 実測のピーク 6.4GB に、Python 本体と読み込み中の画像ぶんの余裕を足す
          memory = "12Gi"
        }
        # 起動時に CPU を多めに割り当て、モデルの読み込み（224MB）を短くする
        startup_cpu_boost = true
      }
    }
  }

  # イメージは GitHub Actions が差し替える（deploy-cutout.yml）。api.tf と同じ理由
  lifecycle {
    ignore_changes = [
      template[0].containers[0].image,
      client,
      client_version,
    ]
  }

  depends_on = [google_project_service.required]
}

# 入口は API 層と同じ扱い。ブラウザからは Cloud Run IAM を使えないので開け、
# アプリケーション側の Firebase 認証（Google ログイン必須）で守る
resource "google_cloud_run_v2_service_iam_member" "cutout_invoker" {
  count = var.cutout_image == "" ? 0 : 1

  name     = google_cloud_run_v2_service.cutout[0].name
  location = var.region
  role     = "roles/run.invoker"
  member   = var.public_access ? "allUsers" : "user:${var.operator_email}"
}

# GitHub Actions からのデプロイ（deploy.tf の deployer をそのまま使う）
resource "google_cloud_run_v2_service_iam_member" "deployer_updates_cutout" {
  count = var.cutout_image == "" ? 0 : 1

  name     = google_cloud_run_v2_service.cutout[0].name
  location = var.region
  role     = "roles/run.developer"
  member   = "serviceAccount:${google_service_account.deployer.email}"
}

resource "google_service_account_iam_member" "deployer_acts_as_cutout" {
  service_account_id = google_service_account.cutout.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.deployer.email}"
}

output "cutout_url" {
  description = "切り抜き API の入口。roomplanner-web の config/api.ts に書く"
  value       = length(google_cloud_run_v2_service.cutout) > 0 ? google_cloud_run_v2_service.cutout[0].uri : "cutout_image を指定して apply するとサービスが作られる"
}
