# コンテナイメージの置き場所。
#
# イメージは実測で 9.47GB。Artifact Registry は $0.10/GB/月 なので月$1程度。
resource "google_artifact_registry_repository" "images" {
  location      = var.region
  repository_id = "hunyuan3d"
  format        = "DOCKER"
  description   = "Hunyuan3D-2GP のコンテナイメージ"

  depends_on = [google_project_service.required]
}
