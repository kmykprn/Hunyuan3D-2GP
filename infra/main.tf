# Hunyuan3D-2GP を Cloud Run(GPU) で動かすためのインフラ定義。
#
# ここで作るものは「まず実測するため」の最小構成。
# コールドスタートと L4 での生成時間という2つの推定値を確定させるのが目的で、
# 本番のAPI層（Cloud Run CPU + Cloud Tasks + Firestore）はまだ含まない。
#
# 使い方は README.md を参照。

terraform {
  required_version = ">= 1.5"

  # state は GCS に置く。ローカルに置くと、作った本人の端末にしか
  # 状態が無く、他の人が apply すると既存リソースを認識できずに
  # 重複作成やエラーになる。
  #
  # このバケットだけは Terraform では作れない（自分の state の置き場を
  # 自分で管理できないため）。手で作ってある:
  #   gcloud storage buckets create gs://<project-id>-tfstate \
  #     --location=asia-southeast1 --uniform-bucket-level-access
  #   gcloud storage buckets update gs://<project-id>-tfstate \
  #     --versioning --public-access-prevention
  #
  # bucket は変数を使えない（初期化時点では変数が解決されない）ので直書きする
  backend "gcs" {
    bucket = "project-db31f07b-2895-48b8-8bb-tfstate"
    prefix = "hunyuan3d"
  }

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 8.1"
    }
  }
}

# billing_project / user_project_override は billingbudgets のように
# 「プロジェクトに属さない」APIを、ユーザーのADCで叩くために要る。
# 無いと呼び出しが gcloud の既定クライアントプロジェクトに紐づけられ、
# そちらでAPIが無効なため SERVICE_DISABLED で失敗する
# （consumer が自分のプロジェクトではない番号になっているのが目印）。
provider "google" {
  project               = var.project_id
  region                = var.region
  billing_project       = var.project_id
  user_project_override = true
}

# 使う API を明示的に有効化する。
#
# disable_on_destroy を false にしているのは、terraform destroy のたびに
# API が無効化されると、他のリソースの削除自体が失敗しうるため。
# API を有効にしておくこと自体に費用は発生しない。
resource "google_project_service" "required" {
  for_each = toset([
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "storage.googleapis.com",
    "billingbudgets.googleapis.com",
    "cloudbilling.googleapis.com",
    # job.tf がジョブ用のサービスアカウントを作るのに要る。
    # 無いと google_service_account の作成が accessNotConfigured で落ちる
    "iam.googleapis.com",
    # API層が署名付きURLを発行するのに要る。Cloud Run のサービスアカウントは
    # 秘密鍵を持たないので、SignBlob API に署名を代行させる
    "iamcredentials.googleapis.com",
  ])

  service            = each.value
  disable_on_destroy = false
}
