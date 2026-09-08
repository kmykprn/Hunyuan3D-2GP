# 予算アラート。**他のどのリソースより先に作る。**
#
# この構成で唯一の重大な事故は、GPU インスタンスが起動しっぱなしになること。
# L4 は $1.16/時 なので、消し忘れると月$835 になる。
# ジョブ（常駐しない）で始めるのはそのためだが、将来サービス化して
# min_instances を誤って1以上にすると即座にこの状態になる。
#
# 注意: 予算アラートは**通知するだけで、課金を止めない**。
# 止めたい場合は別途 Pub/Sub と関数で自動停止を組む必要がある。
resource "google_billing_budget" "monthly" {
  billing_account = var.billing_account
  display_name    = "hunyuan3d-${var.project_id}"

  budget_filter {
    projects = ["projects/${data.google_project.this.number}"]
  }

  amount {
    specified_amount {
      currency_code = "USD"
      units         = tostring(var.budget_amount_usd)
    }
  }

  # 50% で気づき、90% で警戒し、100% で確実に止まる判断ができるようにする。
  # forecasted_spend は「このペースだと超える」時点で飛ぶので、
  # 使い切ってからではなく、進行中に気づける
  threshold_rules {
    threshold_percent = 0.5
  }
  threshold_rules {
    threshold_percent = 0.9
  }
  threshold_rules {
    threshold_percent = 1.0
  }
  threshold_rules {
    threshold_percent = 1.0
    spend_basis       = "FORECASTED_SPEND"
  }

  depends_on = [google_project_service.required]
}

data "google_project" "this" {
  project_id = var.project_id
}
