output "image_repository" {
  description = "docker push 先。イメージはここに置く"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}"
}

output "weights_bucket" {
  description = "重みを上げる先（gcloud storage cp --recursive でアップロードする）"
  value       = "gs://${google_storage_bucket.weights.name}"
}

output "run_job_command" {
  description = "実測ジョブを1回動かすコマンド。image 変数を指定するまでは案内文が出る"
  value = length(google_cloud_run_v2_job.measure) > 0 ? (
    "gcloud run jobs execute ${google_cloud_run_v2_job.measure[0].name} --region ${var.region} --wait"
  ) : "イメージを push し、image 変数を指定して apply し直すとジョブが作られる"
}
