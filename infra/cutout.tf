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

# 回数のカウンタと、預かった切り抜き（入力 200KB・結果 400KB × 件数）の置き場
resource "google_storage_bucket" "cutout_state" {
  name     = "${var.project_id}-cutout-state"
  location = upper(var.region)

  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  # カウンタは日付ごとに別ファイル、預かりは取りに来たら用済み。古いものは消してよい
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

# 預かった切り抜きの処理を積む順番待ち。
#
# 切り抜きは「預けて、あとで取りに行く」形（cutout/SPEC.md の /cutout-jobs）。受け付けの
# リクエストはすぐ返し、処理は Cloud Tasks がこのサービス自身の /run を叩いて行う。
# 応答を返したあとの処理はリクエストに紐づかず CPU が付かないので、別のリクエストにする。
# 費用は無料枠（月 100 万件）に収まる
resource "google_cloud_tasks_queue" "cutout" {
  name     = "cutout"
  location = var.region

  rate_limits {
    # 同時に処理するのはサービスの台数まで。それ以上積んでも 429 で待たされるだけ
    max_concurrent_dispatches = 2
    max_dispatches_per_second = 5
  }

  retry_config {
    # 処理側は推論の失敗を failed として 200 で返す。再試行が要るのは
    # 台数の上限で弾かれたときと、処理中にインスタンスが消えたときだけ
    max_attempts  = 5
    min_backoff   = "5s"
    max_backoff   = "60s"
    max_doublings = 3
  }

  depends_on = [google_project_service.required]
}

resource "google_cloud_tasks_queue_iam_member" "cutout_enqueues" {
  name     = google_cloud_tasks_queue.cutout.name
  location = var.region
  role     = "roles/cloudtasks.enqueuer"
  member   = "serviceAccount:${google_service_account.cutout.email}"
}

# 積む処理に自分の OIDC トークンを付けるには、自分自身を「使う」権限が要る
resource "google_service_account_iam_member" "cutout_acts_as_self" {
  service_account_id = google_service_account.cutout.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.cutout.email}"
}

# 推論に使う CPU 数。resources と INFERENCE_THREADS の両方に同じ値を入れる。
# コンテナの中で os.cpu_count() を見るとホストの CPU 数が返るので、ここから渡す
locals {
  cutout_cpu = 4
  # サービス自身の URL。Cloud Run の決定的な URL（サービス名-プロジェクト番号）。
  # サービスの属性から取ると自分自身を参照する循環になるので、形から組む
  cutout_self_url = "https://cutout-${data.google_project.this.number}.${var.region}.run.app"
}

resource "google_cloud_run_v2_service" "cutout" {
  count = var.cutout_image == "" ? 0 : 1

  name     = "cutout"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.cutout.email

    # 1 件 6 秒（処理の /run はモデルの読み込み込みで 30 秒ほど）。60 秒あれば十分
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
      env {
        name  = "INFERENCE_THREADS"
        value = tostring(local.cutout_cpu)
      }
      env {
        name  = "TASK_QUEUE"
        value = google_cloud_tasks_queue.cutout.id
      }
      env {
        name  = "SELF_URL"
        value = local.cutout_self_url
      }
      env {
        name  = "TASK_SERVICE_ACCOUNT"
        value = google_service_account.cutout.email
      }

      resources {
        limits = {
          # 4 vCPU で 1 枚 5 秒、2 vCPU なら 8 秒。費用は同じ（vCPU × 秒）なので速いほうを取る
          cpu = tostring(local.cutout_cpu)
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
