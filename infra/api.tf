# ワーカーが完了通知に使う ID トークンの aud。
#
# 本来は通知先の URL を使うが、Cloud Run のサービスは自分の URL を
# Terraform から受け取れない（google_cloud_run_v2_service.api[0].uri を
# 自分自身の env に入れると循環参照になる）。そこでワーカーと API で
# 共有する固定の文字列にしてある。
locals {
  dispatch_audience = "https://hunyuan3d-dispatch/${var.project_id}"
}

# アプリから叩くAPI層。仕様は api/SPEC.md。
#
# GPU ジョブとは別の Cloud Run サービスにする。API層は薄くて速く、
# 生成は重くて遅いという、寿命もスケール特性も違うものを分けるため。

# 生成用ジョブのサービスアカウントとは別にする。
# API層に必要なのは「生成物バケットの読み書き」と「ジョブの起動」で、
# 重みバケットへのアクセスは要らない
resource "google_service_account" "api" {
  account_id   = "hunyuan3d-api"
  display_name = "Hunyuan3D API層"

  depends_on = [google_project_service.required]
}

# 状態・入力画像・成果物の置き場。API層はここだけを触る
resource "google_storage_bucket_iam_member" "api_rw_outputs" {
  bucket = google_storage_bucket.outputs.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.api.email}"
}

# 署名付きURLの発行に要る。
#
# Cloud Run のサービスアカウントは秘密鍵を持たないため、
# generate_signed_url をそのまま呼ぶと署名できずに失敗する。
# IAM の SignBlob API に署名を代行させる必要があり、そのために
# 「自分自身に対する」トークン作成権限を付ける
# 許可リストは**読むだけ**。書き込みは運用者が gcloud で行う。
# objectAdmin を与えると、門番を門番自身が書き換えられる状態になる
resource "google_storage_bucket_iam_member" "api_reads_config" {
  bucket = google_storage_bucket.config.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.api.email}"
}

resource "google_service_account_iam_member" "api_can_sign_as_itself" {
  service_account_id = google_service_account.api.name
  role               = "roles/iam.serviceAccountTokenCreator"
  member             = "serviceAccount:${google_service_account.api.email}"
}

resource "google_cloud_run_v2_service" "api" {
  count = var.api_image == "" ? 0 : 1

  name     = "hunyuan3d-api"
  location = var.region

  # 未認証を拒否する。段階3で Firebase のトークン検証が動くまでは
  # これがただ一つの防壁になる
  ingress = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.api.email

    # API層は状態を持たないので、使われていないときは畳んでよい。
    # GPU と違い CPU のみなのでコールドスタートも軽い
    scaling {
      min_instance_count = 0
      max_instance_count = 3
    }

    containers {
      image = var.api_image

      env {
        name  = "OUTPUTS_BUCKET"
        value = google_storage_bucket.outputs.name
      }

      # 起動するジョブの場所。run.googleapis.com を直接叩くのに要る
      env {
        name  = "PROJECT_ID"
        value = var.project_id
      }
      env {
        name  = "REGION"
        value = var.region
      }
      env {
        name  = "JOB_NAME"
        value = "hunyuan3d-measure"
      }

      # 限定公開のあいだは許可リストに載っている uid だけを通す。
      # 製品版では false にする。変わるのはこの値と、下の invoker の
      # 付与先（user → allUsers）だけで、コードの作り直しは発生しない
      env {
        name  = "ENFORCE_ALLOWLIST"
        value = tostring(var.enforce_allowlist)
      }

      env {
        name  = "DAILY_LIMIT"
        value = tostring(var.daily_limit)
      }

      env {
        name  = "DAILY_ATTEMPT_LIMIT"
        value = tostring(var.daily_attempt_limit)
      }

      env {
        name  = "MAX_RUNNING_JOBS"
        value = tostring(var.max_running_jobs)
      }

      env {
        name  = "MAX_QUEUED_JOBS_PER_UID"
        value = tostring(var.max_queued_jobs_per_uid)
      }

      env {
        name  = "DISPATCHER_SERVICE_ACCOUNT"
        value = google_service_account.job.email
      }

      env {
        name  = "DISPATCH_AUDIENCE"
        value = local.dispatch_audience
      }

      env {
        name  = "CONFIG_BUCKET"
        value = google_storage_bucket.config.name
      }

      env {
        name  = "ALLOWED_ORIGINS"
        value = join(",", var.allowed_origins)
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "512Mi"
        }
      }
    }
  }

  depends_on = [google_project_service.required]
}

# 呼び出し権限は自分のアカウントにだけ付ける。
#
# allUsers に付けるのは Firebase のトークン検証が動作確認できてから。
# 先に開けると、認証が効いていない状態で誰でもGPUを起動できてしまう。
#
# ブラウザのSPAからは Cloud Run IAM は使えない（IDトークンの audience が
# 合わない）ので、アプリ連携時はここを allUsers にし、
# アプリケーション側の Firebase 認証で守る形になる
# ブラウザからは Cloud Run IAM を使えないため入口は開ける。
#
# 開けても、アプリケーション側で二重に守っている。
#   1. Firebase の ID トークンが無い・不正なら 401
#   2. config/allowed_uids.json（uid）にも allowed_emails.json（メール）にも載っていなければ 403
#
# 2 が効いているので、トークン検証に穴があっても GPU は起動できない。
# 許可リストは GCS に置いてあり、uid を足すたびの再デプロイは要らない。
#
# 閉じたいときは var.public_access を false にすれば運用者だけに戻る
resource "google_cloud_run_v2_service_iam_member" "api_invoker" {
  count = var.api_image == "" ? 0 : 1

  name     = google_cloud_run_v2_service.api[0].name
  location = var.region
  role     = "roles/run.invoker"
  member   = var.public_access ? "allUsers" : "user:${var.operator_email}"
}

# ジョブを起動するのに要る。
#
# roles/run.invoker では足りない（あれはサービスを呼ぶ権限）。
# ジョブの実行は run.jobs.run で、developer に含まれる
resource "google_project_iam_member" "api_runs_jobs" {
  project = var.project_id
  role    = "roles/run.developer"
  member  = "serviceAccount:${google_service_account.api.email}"
}

# 起動したジョブは GPU ジョブ用のサービスアカウントで動く。
# 別のSAとしてジョブを動かすには、その SA を「使う」権限が要る
resource "google_service_account_iam_member" "api_acts_as_job" {
  service_account_id = google_service_account.job.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.api.email}"
}
