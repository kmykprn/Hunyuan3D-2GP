# モデルの重み（約20GB: 形状 7.2GB ＋ テクスチャ 12.7GB）の置き場所。
#
# イメージに焼き込む案もあるが（COPY するだけで済み実装が小さい）、
# 公式ガイドは「重みが10GB未満ならイメージ、超えるなら Cloud Storage」としている。
# どちらが速いかは未検証なので、まずこちらで測ってから判断する。
#
# 重要: バケットはジョブと**同じリージョン**に置くこと。
# リージョンをまたぐと転送が遅くなり、コールドスタートの測定値が濁る。
resource "google_storage_bucket" "weights" {
  name     = "${var.project_id}-hunyuan3d-weights"
  location = upper(var.region)

  # 誤って公開設定にすること自体を不可能にする。
  # いま公開権限は付いていないが、これは事故を防ぐガード
  public_access_prevention = "enforced"

  # 重みは HuggingFace から再取得できるので、消えても復旧可能。
  # バージョニングは容量を食うだけなので付けない
  uniform_bucket_level_access = true

  depends_on = [google_project_service.required]
}

# 生成された 3D モデル（1件あたり約4.8MB）の置き場所。
resource "google_storage_bucket" "outputs" {
  name     = "${var.project_id}-hunyuan3d-outputs"
  location = upper(var.region)

  # 誤って公開設定にすること自体を不可能にする。
  # 署名付きURLはこの設定の影響を受けない（IAMではなく署名で許可するため）
  public_access_prevention = "enforced"

  uniform_bucket_level_access = true

  # 署名付きURLで GLB を取るのはブラウザなので、ここにも CORS が要る。
  # API層とは別のオリジン（storage.googleapis.com）になるため、
  # API層に CORS を入れただけでは GLB のダウンロードで落ちる。
  # three.js の GLTFLoader も内部で fetch を使うので同じ制約を受ける
  cors {
    origin          = var.allowed_origins
    method          = ["GET", "HEAD"]
    response_header = ["Content-Type", "Content-Length"]
    max_age_seconds = 3600
  }

  # 生成物は再生成できるうえ溜まる一方なので、30日で自動的に消す。
  # 消し忘れによる保管費の積み上がりを防ぐ
  lifecycle_rule {
    condition { age = 30 }
    action { type = "Delete" }
  }

  depends_on = [google_project_service.required]
}

# 許可リストの置き場。生成物バケットとは**分ける**。
#
# API のサービスアカウントは生成物バケットに objectAdmin を持つ。
# 許可リストを同じバケットに置くと、門番を門番自身が書き換えられる。
# いまユーザー入力が書き込み先を操作できる箇所は無いが、
# 万一APIに書き込みの穴が空いたときに、許可リストまで到達させない。
#
# ここへの書き込みは運用者が gcloud で行う。APIは読むだけ。
resource "google_storage_bucket" "config" {
  name     = "${var.project_id}-hunyuan3d-config"
  location = upper(var.region)

  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  # 許可リストを消してしまっても戻せるようにする。
  # 数KBしか無いので費用は無視できる
  versioning {
    enabled = true
  }

  depends_on = [google_project_service.required]
}
