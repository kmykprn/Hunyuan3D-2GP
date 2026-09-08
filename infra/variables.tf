variable "project_id" {
  description = "GCP プロジェクトID。コンソールで作成済みのものを指定する"
  type        = string
}

variable "billing_account" {
  description = "請求先アカウントID（例: 01A2B3-C4D5E6-F7890A）。予算アラートに使う"
  type        = string
}

# L4 は東京(asia-northeast1)では提供されていない。
# アジアで選べるのは asia-southeast1（シンガポール）か asia-south1（ムンバイ）。
# 生成は非同期（投げてポーリング）なので、日本からの往復レイテンシは体感に出ない。
variable "region" {
  description = "GPU ワーカーを置くリージョン"
  type        = string
  default     = "asia-southeast1"
}

variable "budget_amount" {
  description = "月額予算。通貨は請求先アカウントに従う（このアカウントは JPY）。超過してもリソースは止まらず、通知が飛ぶだけ"
  type        = number
  default     = 5000
}

variable "image" {
  description = "実行するコンテナイメージ。Artifact Registry に push したものを指定する"
  type        = string
  default     = ""
}

# 秒課金の環境で最も高くつく事故は「終わらないジョブ」なので、上限を必ず持たせる。
# 生成は実測で約2分。重みの読み込みを含めても10分あれば足りる想定で、
# 余裕を見て既定を30分にしてある（L4 は $1.16/時 なので30分で約$0.58）。
variable "job_timeout_seconds" {
  description = "ジョブ1回の実行時間の上限"
  type        = number
  default     = 1800
}

variable "api_image" {
  description = "API層のコンテナイメージ。push 前は空にしておき、サービスを作らせない"
  type        = string
  default     = ""
}

variable "operator_email" {
  description = "API を叩く運用者のGoogleアカウント。段階3で allUsers を許可するまで、この人だけが呼べる"
  type        = string
}
