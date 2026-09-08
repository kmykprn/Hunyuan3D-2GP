# インフラ定義（Terraform）

Hunyuan3D-2GP を Cloud Run(GPU) で動かすための構成。

**まず実測するための最小構成**で、本番のAPI層（Cloud Run CPU + Cloud Tasks +
Firestore）はまだ含まない。目的は `CLOUDRUN_MEASURE.md` に残っている2つの推定値
——コールドスタート（60〜180秒）と L4 での生成時間（75〜95秒）——を確定させること。

## なぜ Terraform なのか

- **`terraform plan` が適用前の差分レビューになる。** この構成で唯一の重大な事故は
  GPU が起動しっぱなしになること（L4 は $1.16/時 ＝ 月$835）。設定ミスを
  適用前に目で確認できる意味は大きい
- **`terraform destroy` で確実に片付く。** 測るためだけに作るリソースなので、
  消し忘れが起きない仕組みが要る
- **構成がリポジトリに残る。** コンソールでポチポチ作った設定は半年後に
  理由が誰にも分からなくなる

## 事前に手でやること

プロジェクトの作成と課金の紐付けだけは、先にコンソールで済ませておく
（`google_project` リソースもあるが、組織の権限が絡んで個人アカウントでは詰まりやすい）。

1. GCP プロジェクトを作る
2. 請求先アカウントを紐付ける
3. `gcloud auth application-default login` で認証する

## 使い方

```bash
cd infra
cp terraform.tfvars.example terraform.tfvars   # 値を埋める
terraform init
terraform plan       # ← 必ず目を通す
terraform apply
```

`plan` で確認すべき点:

- `node_selector.accelerator = "nvidia-l4"` になっているか
- `gpu_zonal_redundancy_disabled = true` か（初回に自動付与されるクォータは
  「zonal redundancy off」の3枚。有効にするとクォータ不足で弾かれる）
- `max_retries = 0` か（GPU は秒課金なので、黙ってリトライされると費用が倍になる）
- `timeout` が意図した値か

## リージョンについて

**L4 は東京（asia-northeast1）では提供されていない。**
アジアで選べるのは `asia-southeast1`（シンガポール）か `asia-south1`（ムンバイ）。

生成は非同期（投げてポーリング）なので、日本からの往復レイテンシは体感に出ない。
将来API層を作るときは、ユーザーが叩く側だけ東京に置き、GPUワーカーだけ
シンガポールに置く形にできる。

## 実測までの手順

```bash
# 1. イメージを push
terraform output image_repository       # push 先が出る
docker tag hunyuan3d:local <出力>/hunyuan3d:v1
docker push <出力>/hunyuan3d:v1

# 2. 重みを上げる（約20GB。並列ダウンロードのため cp を使う）
terraform output weights_bucket
gcloud storage cp --recursive ~/.cache/huggingface/* gs://.../

# 3. image 変数にタグを入れて apply し直す
terraform apply -var="image=<出力>/hunyuan3d:v1"

# 4. 実行して測る
terraform output run_job_command        # 実行コマンドが出る
```

## 重みの供給方法について

現在は **Cloud Storage FUSE でマウント**している。イメージにもコードにも
手を入れずに済むのが利点。

ただし公式ガイドは「FUSE は初回ダウンロードを並列化しないので、
大きな重みでは `gcloud storage cp` の並列ダウンロードより遅い」としている。
イメージに焼き込む案（`COPY` するだけ）もある。

**どれが速いかは未検証。** まずこの構成で測り、その値を基準に比較する。

## 状態ファイルの扱い

現在はローカル状態（`terraform.tfstate`）。**リポジトリにコミットしないこと**
（`.gitignore` 済み）。状態ファイルにはリソースの詳細が入り、機密が含まれうる。

複数の環境から触るようになったら GCS バックエンドに移す。
