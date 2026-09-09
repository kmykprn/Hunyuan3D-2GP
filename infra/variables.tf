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

variable "gpu_type" {
  description = <<-EOT
    使う GPU。Cloud Run は nvidia-l4 と nvidia-rtx-pro-6000 の2種類しか提供していない。

    nvidia-rtx-pro-6000 は現状のイメージでは動かない。Blackwell(sm_120)だが
    torch 2.5.1+cu124 は sm_90 までしか対応せず、
    「CUDA error: no kernel image is available for execution on the device」で落ちる。
    使うには torch を cu128 以降に上げ、TORCH_CUDA_ARCH_LIST に 12.0 を足して
    CUDA拡張を再ビルドする必要がある。
    しかも単価が L4(8CPU/32GiB)の2.24倍なので、224秒を切らないと費用は増える。
  EOT
  type        = string
  default     = "nvidia-l4"
}

variable "job_cpu" {
  description = "生成ジョブのCPU数。L4 は4以上、RTX Pro 6000 は20以上が必須。CPUとメモリには比率の制約があり、32Gi には8CPUが要る（4CPUだと上限16Gi）"
  type        = string
  default     = "8"
}

variable "job_memory" {
  description = "生成ジョブのメモリ。16Gi では2本目のパイプラインのロード中に OOM する（実測）"
  type        = string
  default     = "32Gi"
}

variable "enforce_allowlist" {
  description = "限定公開中は true。config/allowed_uids.json に載っている uid だけを通す"
  type        = bool
  default     = true
}

variable "allowed_origins" {
  description = <<-EOT
    ブラウザから叩くことを許すオリジン。

    アプリは GitHub Pages、APIは Cloud Run と別オリジンなので、
    ここに載っていないとブラウザはリクエストを通さない。
    API層のCORSと、生成物バケット（署名付きURLでGLBを取る先）の
    両方に同じ値を使う。

    Capacitor で iOS アプリにするときは capacitor://localhost を足す。
  EOT
  type        = list(string)
  default = [
    "https://kmykprn.github.io",
    "http://localhost:5173",
  ]
}

variable "daily_limit" {
  description = "1 uid あたりの1日の生成回数。1回あたり約39円かかる"
  type        = number
  default     = 10
}

variable "daily_attempt_limit" {
  description = <<-EOT
    1 uid あたり1日に起動できる回数。失敗も数え、戻さない。

    失敗した生成は daily_limit には数えない（利用者がこちら都合の失敗で
    枠を失わないため）が、それだけだと生成に向かない画像で延々と
    再試行でき、そのたびに GPU が起動する。1回約39円なので上限を置く。
  EOT
  type        = number
  default     = 20
}

variable "public_access" {
  description = <<-EOT
    Cloud Run の入口を開けるか。

    ブラウザのSPAからは Cloud Run IAM を使えない（IDトークンの audience が
    合わない）ため、アプリ連携には true が要る。

    開ける前に、Firebase のトークン検証がデプロイ済みで、かつ
    config/allowed_uids.json による制限が効いていることを必ず確認すること。
    許可リストが空なら誰も GPU を起動できないので、その状態で開けるのが安全。
  EOT
  type        = bool
  default     = false
}
