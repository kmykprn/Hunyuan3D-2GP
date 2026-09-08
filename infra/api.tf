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
# allUsers に付けるのは段階3-2（Firebase のトークン検証）が動いてから。
# 先に開けると、認証が効いていない状態で誰でもGPUを起動できてしまう
resource "google_cloud_run_v2_service_iam_member" "api_invoker" {
  count = var.api_image == "" ? 0 : 1

  name     = google_cloud_run_v2_service.api[0].name
  location = var.region
  role     = "roles/run.invoker"
  member   = "user:${var.operator_email}"
}
