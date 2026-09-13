# GPU ジョブの外向き通信を VPC 経由（Direct VPC egress）にするための最小の網。
#
# 目的は重みの読み込みを速くすること。Cloud Run の既定の経路では、GCS からの
# ダウンロードが 1 インスタンスあたり 60MB/s 前後で頭打ちだった（1 本でも並列でも同じ。
# イメージの取り込みも同じ速さ）。公式は「Direct VPC + Private Google Access で
# 重みの通信を Google 内部の経路に乗せる」ことを読み込み高速化の手段として挙げている。
# https://docs.cloud.google.com/run/docs/configuring/jobs/gpu-best-practices
#
# NAT は置かない。ジョブが外へ出るのは GCS と API 層（*.run.app）だけで、
# どちらも Private Google Access で届く。インターネットへは出られない（出る必要も無い）

resource "google_compute_network" "jobs" {
  name                    = "hunyuan3d-jobs"
  auto_create_subnetworks = false

  depends_on = [google_project_service.required]
}

resource "google_compute_subnetwork" "jobs" {
  name          = "hunyuan3d-jobs"
  region        = var.region
  network       = google_compute_network.jobs.id
  ip_cidr_range = "10.10.0.0/24"

  # GCS などの Google API へ、外部 IP 無しで届くようにする
  private_ip_google_access = true
}
