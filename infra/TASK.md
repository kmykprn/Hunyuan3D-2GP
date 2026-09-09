# 作業依頼: GCP に載せて、コールドスタートと L4 での生成時間を実測する

> **この作業は完了している。** 結果は `MEASURE_RESULT.md`、
> いま何がどうなっているかは `HANDOFF.md` を参照。
> これから着手する作業は `../api/TASK.md`。
>
> 以下は当時の手順で、環境を再現するときの参考として残してある。

WSL2 の実機（GPU・Docker・重み一式がある側）で実行してもらう作業。

## ゴール

`CLOUDRUN_MEASURE.md` に残っている**2つの推定値を実測値に置き換える**こと。

| | 現在 | 確定させたい |
|---|---|---|
| コールドスタート | 60〜180秒（**3倍の幅**） | 実測値 |
| L4 での生成時間 | 75〜95秒（4070からの外挿） | 実測値 |

月額見積もり $36 のうち $33 がこの推定に乗っている。ここが確定するまで、
「重みをイメージに焼くか GCS から落とすか」も「profile を上げるべきか」も決められない。

**数字を取るのが目的で、きれいに作ることは目的ではない。**

## 前提（人間が済ませてあるはず）

- GCP の課金アカウントが有効（クレジットカード登録済み）
- `gcloud auth login` と `gcloud auth application-default login` が実行済み
- WSL2 に Docker + NVIDIA Container Toolkit、`~/.cache/huggingface` に重み約20GB

確認:

```bash
gcloud auth list
gcloud auth application-default print-access-token >/dev/null && echo "ADC OK"
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
du -sh ~/.cache/huggingface
```

どれかが通らなければ、**そこで止めて人間に報告すること。** ブラウザ承認や
カード登録は代行できない。

## 手順

### 1. プロジェクトの準備

```bash
export PROJECT_ID="<決めたID>"      # 全世界で一意。既存があればそれを使う
gcloud config set project "$PROJECT_ID"
gcloud services enable cloudresourcemanager.googleapis.com serviceusage.googleapis.com
gcloud billing accounts list          # ACCOUNT_ID を控える
```

### 2. 予算アラートを最初に作る

**他のどのリソースより先に。** この構成で唯一の重大な事故は GPU が
起動しっぱなしになること（L4 は $1.16/時 ＝ 月$835）。

```bash
cd infra
cp terraform.tfvars.example terraform.tfvars   # project_id と billing_account を埋める
terraform init
terraform apply -target=google_billing_budget.monthly
```

**注意: 予算アラートは通知するだけで課金を止めない。** 止める仕組みではなく
気づく仕組みなので、通知先メールが届くことを確認しておくこと。

### 3. レジストリとバケットを作る

`image` 変数が空のうちは実測ジョブは作られない（`count` で分岐させてある）ので、
先に器だけ作る。

```bash
terraform plan     # ← 目を通してから
terraform apply
terraform output   # push 先とバケット名が出る
```

`plan` で確認すること:

- `gpu_zonal_redundancy_disabled = true`（初回に自動付与されるクォータは
  「zonal redundancy off」の3枚。有効にするとクォータ不足で弾かれる）
- `max_retries = 0`（秒課金なので、黙ってリトライされると費用が倍になる）

### 4. イメージを push

```bash
REPO=$(terraform output -raw image_repository)
gcloud auth configure-docker "${REPO%%/*}"

docker build -t "$REPO/hunyuan3d:v1" ..
docker push "$REPO/hunyuan3d:v1"
```

イメージは 9.47GB。初回の push は回線次第で時間がかかる。

### 5. 重みを GCS に上げる

```bash
BUCKET=$(terraform output -raw weights_bucket)
gcloud storage cp --recursive ~/.cache/huggingface/hub "$BUCKET/hub"
```

約20GB。`gcloud storage cp` は並列転送するので、`gsutil` より速い。

**バケットはジョブと同じリージョンにある**こと（Terraform でそう作ってある）。
リージョンをまたぐと転送が遅くなり、測定値が濁る。

### 6. ジョブを作って実行する

```bash
terraform apply -var="image=$REPO/hunyuan3d:v1"
gcloud run jobs execute hunyuan3d-measure --region asia-southeast1 --wait
```

**1回目がコールドスタート込みの数字**になる。続けてもう1回流すと、
温まった状態との差が出る。**両方測ること。**

```bash
# ログから各工程の時間を拾う
gcloud run jobs executions list --job hunyuan3d-measure --region asia-southeast1
gcloud logging read \
  'resource.type="cloud_run_job" AND resource.labels.job_name="hunyuan3d-measure"' \
  --limit 200 --format='value(timestamp,textPayload)' | tac
```

### 7. 片付ける

**測り終わったら必ず消すこと。** ジョブ自体は常駐しないので放置しても
課金され続けはしないが、消し忘れの習慣をつけない。

```bash
terraform destroy -target=google_cloud_run_v2_job.measure
```

バケットとレジストリは残してよい（保管費は月$1程度）。

## 報告してほしいこと

このPRにコメントで書き戻す。

1. **コールドスタート込みの所要時間**（1回目）と、**温まった状態**（2回目）
2. **各工程の内訳** — モデルロード / 形状生成 / 面数削減 / テクスチャ生成。
   ローカル実測（14.6 / 7.0 / 45.6 秒、全体125.0秒）と並べて比較できる形で
3. **重みの読み込みにかかった時間** — ここが推定60〜180秒だった部分。**最重要**
4. **1回あたりの実費用** — 課金レポートか、実行秒数 × $0.000323 で概算
5. **失敗したら、失敗したコマンドとエラー全文**

## 詰まりそうなところ

**この手順は誰も実機で試していない。** 以下は想定であって、確認済みではない。

| 症状 | 見立て |
|---|---|
| GPU クォータ不足でデプロイが弾かれる | `gpu_zonal_redundancy_disabled = true` になっているか確認。初回自動付与は「off」の3枚 |
| リージョンで L4 が使えない | **L4 は東京(asia-northeast1)に無い**。`asia-southeast1` か `asia-south1` |
| ジョブが起動直後に落ちる | 重みのマウント先が違う可能性。`HF_HOME=/models` に対して `$BUCKET/hub` を上げているか |
| モデルのダウンロードが始まる | 同上。マウントされた中身が `hub/` の階層になっていない |
| 極端に遅い | Cloud Storage FUSE は初回ダウンロードを並列化しない。**それ自体が測りたい値**なので、遅くても数字を取ること |
| `terraform apply` が権限エラー | ADC のアカウントにプロジェクトの権限があるか。`gcloud auth application-default login` をやり直す |

## 判断が要るときは止めること

以下は勝手に進めず、PRにコメントして人間の判断を仰ぐこと。

- **リージョンを変える**（レイテンシとデータ所在地に関わる）
- **マシンサイズを上げる**（費用が変わる）
- **`min-instances` を1以上にする**（**絶対にやらない**。月$835になる）
- **予算額を上げる**
