# GitHub Actions から API 層をデプロイするための入口。
#
# main にマージされたら .github/workflows/deploy-api.yml が
#   1. api/ のイメージを組んで Artifact Registry に push
#   2. Cloud Run のサービス hunyuan3d-api をそのイメージに更新
# する。そのために GitHub が GCP に入る許可をここで作る。
#
# 鍵ファイルは作らない。Workload Identity 連携で、GitHub が発行する
# 短命のトークンをこのプロジェクトのサービスアカウントに引き換える。
# 引き換えられるのは var.github_repository のワークフローだけ。
#
# 一度 apply したら、outputs の deploy_workload_identity_provider と
# deploy_service_account を GitHub のリポジトリ変数に入れる（HANDOFF.md 参照）。

locals {
  github_principal = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github.name}/attribute.repository/${var.github_repository}"
}

resource "google_iam_workload_identity_pool" "github" {
  workload_identity_pool_id = "github"
  display_name              = "GitHub Actions"

  depends_on = [google_project_service.required]
}

resource "google_iam_workload_identity_pool_provider" "github" {
  workload_identity_pool_id          = google_iam_workload_identity_pool.github.workload_identity_pool_id
  workload_identity_pool_provider_id = "github"
  display_name                       = "GitHub Actions"

  # 他のリポジトリのワークフローが、このプールでトークンを引き換えるのを防ぐ
  attribute_condition = "assertion.repository == \"${var.github_repository}\""

  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.repository" = "assertion.repository"
    "attribute.ref"        = "assertion.ref"
  }

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

# デプロイ専用のサービスアカウント。API 層や GPU ジョブのものとは分ける。
# 持たせるのは「イメージを置く」「サービスを更新する」だけ
resource "google_service_account" "deployer" {
  account_id   = "hunyuan3d-deployer"
  display_name = "GitHub Actions からのデプロイ"

  depends_on = [google_project_service.required]
}

# GitHub のワークフローがこのサービスアカウントになれるようにする
resource "google_service_account_iam_member" "github_impersonates_deployer" {
  service_account_id = google_service_account.deployer.name
  role               = "roles/iam.workloadIdentityUser"
  member             = local.github_principal
}

# イメージの置き場に書ける（このリポジトリだけ）
resource "google_artifact_registry_repository_iam_member" "deployer_pushes_images" {
  location   = google_artifact_registry_repository.images.location
  repository = google_artifact_registry_repository.images.name
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${google_service_account.deployer.email}"
}

# サービスを新しいイメージに更新できる。
# run.developer は services.update を含む（admin は要らない）
resource "google_cloud_run_v2_service_iam_member" "deployer_updates_api" {
  count = var.api_image == "" ? 0 : 1

  name     = google_cloud_run_v2_service.api[0].name
  location = var.region
  role     = "roles/run.developer"
  member   = "serviceAccount:${google_service_account.deployer.email}"
}

# 新しいリビジョンは API 層のサービスアカウントで動く。
# 別の SA でサービスを動かすには、その SA を「使う」権限が要る
resource "google_service_account_iam_member" "deployer_acts_as_api" {
  service_account_id = google_service_account.api.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.deployer.email}"
}

output "deploy_workload_identity_provider" {
  description = "GitHub のリポジトリ変数 GCP_WORKLOAD_IDENTITY_PROVIDER に入れる値"
  value       = google_iam_workload_identity_pool_provider.github.name
}

output "deploy_service_account" {
  description = "GitHub のリポジトリ変数 GCP_DEPLOY_SERVICE_ACCOUNT に入れる値"
  value       = google_service_account.deployer.email
}
