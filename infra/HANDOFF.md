# 引き継ぎ

Cloud Run(GPU) 上で動いている3D生成の、いま何がどうなっているかと、
触るときに必要なもの。

計測の数字そのものは `MEASURE_RESULT.md`、APIの取り決めは `../api/SPEC.md` にある。

**分かっているが直していないものは、クライアント側のリポジトリにまとめてある。**
2つのリポジトリにまたがるため1箇所に置いた。着手前に読むこと。
https://github.com/kmykprn/roomplanner-web/blob/main/docs/OPEN_ISSUES.md

このリポジトリに関わる未解決事項は**1つ**。

- **OAuth 同意画面が「テスト中」のまま。** Google ログインは使えるが、
  テストユーザーに登録したアカウントしか通らない（上限100人）。人に配る前に
  本番へ切り替える必要がある。切り替えにはホームページとプライバシーポリシーの
  URL が要り、プライバシーポリシーは未作成。**公開ステータスもテストユーザーの
  追加も API が無く、コンソールでしか操作できない**

かつて挙げていた「ワーカーのメモリ未測定」と「完了通知が2件目で効くか未確認」は、
**どちらも実機で確認して解決した**（メモリはピーク51%、通知は3枚同時投稿で動作を確認）。
経緯と再測定の手順は OPEN_ISSUES.md の末尾に残してある。

## いまできること

**`roomplanner-web` から画像を投げて GLB を受け取れる状態が動いている。**

| | |
|---|---|
| 認証 | Firebase の匿名認証 |
| 生成 | L4 8CPU/32GiB、1生成 **502秒** |
| 出力 | 25,093頂点 / 40,000面 / 2048px テクスチャ / 約4.0MB |
| 入力形式 | JPEG / PNG / WebP / HEIC・HEIF |
| 費用 | **1生成39円**、月500生成で約19,600円（1ドル165円） |

---

# 使うだけなら、環境構築は要らない

APIは公開されているので、**curl があればどこからでも叩ける**。
GPU も Docker も重みもローカルには要らない。

| | |
|---|---|
| API | `https://hunyuan3d-api-yvl3t4jpxa-as.a.run.app` |
| Firebase ウェブAPIキー | **ここには書かない**（下記の方法で取得する） |

ウェブAPIキーはブラウザのJSに埋め込まれる前提の値で、これ単体では何もできない
（後述の許可リストで弾かれる）。ただし**このリポジトリは公開されている**ので、
書けば誰でも匿名トークンを作れる状態になる。許可リストがあるのでGPUは
起動できないが、認証エンドポイントとAPI層に負荷をかけられるため載せない。

権限のある人は次で取得できる。

```bash
P=project-db31f07b-2895-48b8-8bb
APPID=$(gcloud auth print-access-token > /dev/null; \
  curl -s -H "Authorization: Bearer $(gcloud auth print-access-token)" \
    -H "X-Goog-User-Project: $P" \
    "https://firebase.googleapis.com/v1beta1/projects/$P/webApps" \
  | python3 -c "import json,sys;print(json.load(sys.stdin)['apps'][0]['appId'])")

curl -s -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "X-Goog-User-Project: $P" \
  "https://firebase.googleapis.com/v1beta1/projects/$P/webApps/$APPID/config" \
  | python3 -c "import json,sys;print(json.load(sys.stdin)['apiKey'])"
```

Firebase コンソールの「プロジェクトの設定 → マイアプリ」からも見られる。

## 手順

```bash
API=https://hunyuan3d-api-yvl3t4jpxa-as.a.run.app
KEY=<上記の方法で取得したウェブAPIキー>

# 1. 匿名ユーザーを作る。localId が uid、idToken が1時間有効なトークン
curl -s -X POST -H "Content-Type: application/json" -d '{"returnSecureToken":true}' \
  "https://identitytoolkit.googleapis.com/v1/accounts:signUp?key=$KEY"

# 2. 返ってきた localId を許可リストに追加してもらう（次節）。しないと 403

# 3. 生成を投げる
curl -H "Authorization: Bearer $ID_TOKEN" -F image=@photo.jpg "$API/jobs"
#    → {"jobId":"job_..."}

# 4. 状態を見る。succeeded になったら modelUrl から落とす（1時間有効）
curl -H "Authorization: Bearer $ID_TOKEN" "$API/jobs/$JOB_ID"
```

トークンは1時間で切れる。切らさずに使うなら `refreshToken` で更新する。

```bash
curl -s -X POST -d "grant_type=refresh_token&refresh_token=$RT" \
  "https://securetoken.googleapis.com/v1/token?key=$KEY"
```

生成は**約8〜9分**かかる。`state` は `queued` → `running` → `succeeded`。

## 応答の意味

| | |
|---|---|
| 401 | トークンが無い、または無効 |
| 403 | 許可リストに載っていない |
| 409 | 同時更新を処理できなかった。少し待って再試行する |
| 429 | 当日の上限、起動回数の上限、または待機上限に達した |
| 400 | 画像として読めない、または5MB超 |

409 と 400 は回数を消費しない。

## 複数写真を作るとき

写真ごとにジョブを受け付ける。GPU が空いていればすぐ実行し、埋まっていれば
`queued` として受付順に待機する。既定ではプロジェクト全体で**同時に2本**まで実行し、
1 uid は**待機5件**まで受け付ける。完了したワーカーが次の待機ジョブを開始するため、
利用者が画面を閉じても待機列は止まらない。

上限は `terraform.tfvars` の `max_running_jobs` と `max_queued_jobs_per_uid` で変更する。
L4 の割り当てはプロジェクト・リージョン単位なので、`max_running_jobs` は実際に付与された
Cloud Run GPU クォータ以下にする。

## 許可リストに uid を追加する

GCS のファイルを1つ書き換えるだけ。**再デプロイは要らない**（60秒で反映）。

```bash
B=gs://project-db31f07b-2895-48b8-8bb-hunyuan3d-config/config/allowed_uids.json
gcloud storage cat $B > /tmp/a.json
# ["既存のuid", "追加するuid"] の形にする
gcloud storage cp /tmp/a.json $B
```

書き込みには GCS への権限が要る。

**生成物バケットとは別のバケットに置いてある。** APIと生成ジョブは生成物
バケットに書き込み権限を持つため、同居させると門番を門番自身が
書き換えられる状態になる。APIはこのバケットを読むだけ。

---

# セキュリティ

**`../api/SECURITY.md` を参照。** 何が門番で、何が守っていないかと、
Firebase 側（Terraform 管理外）で確認すべき項目がまとまっている。

**ブラウザから叩けるオリジンは本番だけ。** `localhost` は既定に入れていない。
開発でローカルから実APIを叩く必要が出たときだけ `terraform.tfvars` に足し、
用が済んだら戻す。

# 誰が叩けるのか（重要）

**Cloud Run の入口は `allUsers` に開いている。** ブラウザのSPAからは
Cloud Run IAM を使えない（IDトークンの audience が合わない）ため。

守っているのは**アプリケーション側の二段構え**。

1. Firebase の ID トークンが無い・不正なら 401
2. **`config/allowed_uids.json` に載っていない uid は 403**

**ウェブAPIキーを知っていても、2 で止まるのでGPUは起動できない。**
実質的な門番は許可リストであって、APIキーではない。

API層への通信自体は届くが、CPUのみで最大3インスタンスに制限してあるので
被害は限定的。GPUの課金は発生しない。

## 製品版にするときの注意

`ENFORCE_ALLOWLIST=false` にすると、**この2枚目の壁が無くなる。**

そのとき「1日10回」の制限は**ほとんど意味を持たない**。匿名アカウントは
いくらでも作れるので、100個作れば1000回になる。1回39円なので39,000円。

**許可リストを外すなら App Check が前提**。`../api/SPEC.md` が App Check を
製品版の必須項目に挙げているのはこのため。

移行で変わるのは2箇所だけにしてある。

  - `enforce_allowlist` を false にする
  - App Check を足す

コードの作り直しは発生しない。

---

# インフラを触るなら

`terraform apply` やイメージの再ビルドをする場合。

## 環境を用意する

この端末では `sudo` にパスワードが要るため、どちらも `$HOME` に入れてある。

```bash
curl -sfLO https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
tar -xzf google-cloud-cli-linux-x86_64.tar.gz -C "$HOME"
"$HOME/google-cloud-sdk/install.sh" --quiet --path-update=false

curl -sfLO https://releases.hashicorp.com/terraform/1.9.8/terraform_1.9.8_linux_amd64.zip
mkdir -p "$HOME/.local/bin" && unzip -o terraform_1.9.8_linux_amd64.zip -d "$HOME/.local/bin"

export PATH="$HOME/google-cloud-sdk/bin:$HOME/.local/bin:$PATH"
```

認証は**ブラウザ承認が要るので代行できない**。

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project project-db31f07b-2895-48b8-8bb
```

## terraform.tfvars を用意する

gitignore してあるのでリポジトリには無い。`terraform.tfvars.example` を写して埋める。

```hcl
project_id      = "project-db31f07b-2895-48b8-8bb"
billing_account = "<別途受け取ること>"
region          = "asia-southeast1"
budget_amount   = 5000
operator_email  = "<自分のGoogleアカウント>"

# ↓ ここから下を書き忘れると、既定値に落ちて本番が壊れる
public_access   = true
image           = "asia-southeast1-docker.pkg.dev/project-db31f07b-2895-48b8-8bb/hunyuan3d/hunyuan3d:v3"
api_image       = "asia-southeast1-docker.pkg.dev/project-db31f07b-2895-48b8-8bb/hunyuan3d/api:v8"
```

**請求先アカウントIDはここに書かない。** 安全な経路で受け取る。

state は GCS にあるので、`terraform init` すれば同じ状態を参照できる。

### ⚠️ `public_access` と イメージタグ を tfvars に書く理由

`public_access` の既定は **`false`**、イメージの既定は **空文字**。
つまり **tfvars に書かないまま `terraform apply` すると、
`allUsers` が剥がれてSPAが全部落ち、動いているイメージも外れる。**

これは「`allowed_origins` を1行直すだけ」のような無関係な変更でも起きる。
`-var` で毎回渡す運用にすると、渡し忘れた回に事故る。**tfvars に書いて、
素の `terraform apply` が常に正しくなる状態にしておくこと。**

## 適用する

```bash
terraform apply
```

`plan` の差分が**自分が意図した分だけ**になっているか必ず見る。
`api_invoker ... must be replaced` や `image -> null` が出ていたら、
上の変数が tfvars に入っていない。

入口を閉じて運用者だけに戻すときは `public_access = false` にする。

---

# いまクラウド上にあるもの

プロジェクト: `project-db31f07b-2895-48b8-8bb`（表示名 `roomplanner`）/ `asia-southeast1`

| リソース | 状態 |
|---|---|
| Cloud Run サービス `hunyuan3d-api` | API層。CPUのみ、最大3インスタンス、使わなければ0 |
| Cloud Run ジョブ `hunyuan3d-measure` | 生成本体。**常駐しない**。APIから起動される |
| Artifact Registry `hunyuan3d` | `hunyuan3d:v3`（9.6GB）と `api:v7`（207MB） |
| 重みバケット | 21.1GB |
| 生成物バケット | 入力画像・状態・成果物・回数。30日で自動削除 |
| 設定バケット | 許可リスト。**APIは読むだけ**。versioning 有効 |
| state バケット | Terraform の state |
| 予算アラート | **JPY 5,000円**（50% / 90% / 100% / 予測100%で通知） |

**GPU は常駐していないので、放置しても課金は増えない。**
保管費（イメージ + 重み + state）が月200円程度かかるだけ。

## 費用の内訳（月500生成、1ドル165円）

| | |
|---|---|
| GPUジョブ | 19,616円 |
| API層 | 1,059円 |
| ストレージ | 171円 |
| 下り通信 | 40円 |
| **合計** | **約20,900円** |

単価は Cloud Billing の価格カタログAPIから取得した実データ。

**API層の1,059円は見落としやすい。** 生成中はクライアントがポーリングするため、
API層のインスタンスが9分間起きたまま課金される。

---

# 次にやること

## 未着手

**`roomplanner-web` 側の実装**（別リポジトリ）。`../api/TASK.md` の段階3-4 に詳細がある。

  - `signInAnonymously()` と `src/platform/api.ts` の新設
  - 画像を長辺1024pxへ縮小してから送る
  - `jobId` を `localStorage` に保存し、起動時にポーリングを再開する
  - GLB を Cache Storage に明示的に保存する
    （既存の PWA 設定の `urlPattern: /\.glb$/` は署名付きURLにマッチしない）

**App Check**。製品版で許可リストを外すなら必須。

## 費用を下げる余地

502秒のうち **263秒（52%）がモデルのロード**で、実行のたびに発生している。

| 打ち手 | 効果 | 前提 |
|---|---|---|
| 常駐サーバー化 | 502秒 → 239秒、月額ほぼ半減 | **リクエストが時間的に固まること** |
| safetensors 化 | ロード185秒 → 約95秒（推定） | 前提なし。ただし効果は小さい |

常駐サーバー化は「1セッションあたり何点生成するか」で損得が変わる。
GPU付きの Cloud Run サービスは**待機時間も課金される**ため、1件ずつ孤立して
発生するなら、いまのジョブ方式より高くなる。

**限定公開で実利用のデータを取ってから判断するのが妥当。**

GPUの変更による高速化は打ち止め。Cloud Run が提供するのは L4 と
RTX Pro 6000 の2種類だけで、後者は現状のイメージでは動かない
（Blackwell sm_120 に torch 2.5.1 が非対応）。詳細は `variables.tf` の
`gpu_type` に書いてある。
