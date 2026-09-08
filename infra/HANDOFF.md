# 引き継ぎ

`TASK.md` の作業を実機で実施した結果と、いま何がどうなっているか。
これから触る人がまず読む想定。

計測の数字そのものは `MEASURE_RESULT.md` にある。ここには
**状態と、次に判断すべきこと**を書く。

## 結論（先に）

Cloud Run(GPU) でテクスチャ付き生成が**動くことは確認できた**。
ただし**費用は当初見積もりの約2.6倍**になる。

| | 見積もり | 実測 |
|---|---|---|
| 1回あたり | $0.066 | **$0.172** |
| 月500生成 | $33 | **$86** |

原因は、生成そのものではなく**毎回発生する263秒のモデルロード**。
これが全体533秒の49%を占める。

## いまクラウド上にあるもの

プロジェクト: `project-db31f07b-2895-48b8-8bb` / リージョン `asia-southeast1`

| リソース | 状態 |
|---|---|
| 予算アラート | **あり**（JPY 5,000円 / 50%・90%・100%・予測100%で通知） |
| Artifact Registry `hunyuan3d` | **あり**（イメージ `hunyuan3d:v1` 9.47GB を push 済み） |
| 重み用バケット `...-hunyuan3d-weights` | **あり**（21.1GB） |
| 生成物用バケット `...-hunyuan3d-outputs` | **あり**（空。ジョブにマウントされていない） |
| state 用バケット `...-tfstate` | **あり** |
| サービスアカウント `hunyuan3d-job` | あり |
| **Cloud Run ジョブ** | **削除済み**（計測後に destroy した） |

**GPU は常駐していないので、放置しても課金は増えない。**
保管費（イメージ + 重み + state）が月$1程度かかるだけ。

ジョブを再作成するには image を指定して apply する。

```bash
terraform apply -var="image=asia-southeast1-docker.pkg.dev/project-db31f07b-2895-48b8-8bb/hunyuan3d/hunyuan3d:v1"
```

## 手元の環境を再現する

この端末では sudo にパスワードが要るため、どちらも `$HOME` に入れてある。

```bash
# gcloud
curl -sfLO https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
tar -xzf google-cloud-cli-linux-x86_64.tar.gz -C "$HOME"
"$HOME/google-cloud-sdk/install.sh" --quiet --path-update=false

# terraform
curl -sfLO https://releases.hashicorp.com/terraform/1.9.8/terraform_1.9.8_linux_amd64.zip
mkdir -p "$HOME/.local/bin" && unzip -o terraform_1.9.8_linux_amd64.zip -d "$HOME/.local/bin"

export PATH="$HOME/google-cloud-sdk/bin:$HOME/.local/bin:$PATH"
```

認証（**ブラウザ承認が要るので代行できない**）。

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project project-db31f07b-2895-48b8-8bb
```

`terraform.tfvars` は gitignore してある。`terraform.tfvars.example` を写して
`project_id` と `billing_account` を埋める。**請求先アカウントIDは別途受け取ること。**

state は GCS にあるので、`terraform init` すれば同じ状態を参照できる。

## 確定した制約

**テクスチャは必須。** 当初「形状のみにすれば月$39」という案があったが、
製品要件として却下された。この前提で打ち手を考える必要がある。

## 次に判断すべきこと

残る打ち手は2つ。**どちらもまだ実施していない。**

### 打ち手1: サービス化（効果は最大、ただし条件付き）

いまはジョブなので実行のたびにモデルを読み直している。HTTPサービスにすれば
インスタンスが生きている間はロードを再利用できる。

ただし [Cloud Run の GPU はインスタンス課金が必須](https://docs.cloud.google.com/run/docs/configuring/services/gpu)で、
**アイドル時間も課金される**。したがって得かどうかは
**リクエストが時間的に固まるかどうか**で決まる。

| 利用パターン | サービス | ジョブ |
|---|---|---|
| 1件ずつ孤立 | 409秒 + アイドル | 533秒 → **ほぼ引き分けか、むしろ高い** |
| 1セッションで5件 | 1件あたり約210秒 | 533秒 → **約2.5倍安い** |

**先に決めるべきは「1セッションあたり何件生成されるか」。**
roomplanner-web で1人が続けて家具を並べるなら効果は大きい。

### 打ち手4: safetensors 化（効果は小さいが確実）

`is_safetensors_compatible` が False を返しており、`text_encoder` に `.bin` しか
無いせいで **3.41GB の unet も pickle 版が読まれている**（検証済み）。

`hy3dgen/texgen/utils/multiview_utils.py` の `from_pretrained` に
`use_safetensors=True` を渡すのが正攻法。イメージの再ビルドが要る。

推定効果はテクスチャのロード 185秒 → 約95秒、月 $86 → $71（約17%）。
**利用パターンに依存せず効く**ので、先に片付けておく価値はある。

**バケットから .bin を消す方法は失敗する。** HF がキャッシュを Hub と照合して
再ダウンロードを試み、読み取り専用マウントに書けずに落ちる。
`HF_HUB_OFFLINE=1` を足しても、展開済みコピーは blobs/ を持たないため
リビジョンを解決できない。詳細は `MEASURE_RESULT.md`。

### 検証済みで効果が無かったもの

**gcsfuse のチューニング（打ち手3）。** `file-cache-cache-file-for-range-read`
を有効にしても 533秒 → 531秒 で差が無かった。同条件4回が 516〜533秒
（ばらつき3%）なので、この結論は安全に出せる。

## 踏むと痛い落とし穴

実際に踏んだもの。同じ失敗を繰り返さないために残す。

| 症状 | 原因と対処 |
|---|---|
| モデルロードが25分かかる | gcsfuse のファイルキャッシュが既定で無効。`cache-dir` の指定が要る |
| `file cache should be enabled for parallel download support` | `file-cache-max-size-mb` だけではキャッシュは有効にならない |
| 起動失敗・ログが一切出ない | in-memory ボリュームは `cr-volume:` で参照するだけでなく、コンテナへのマウントも必要 |
| OOM (signal 9) | キャッシュは**コンテナのメモリ上限にカウントされる**。さらに `download-chunk-size-mb` × `parallel-downloads-per-file` のバッファは**キャッシュ上限とは別枠**（512MB×32 = 最大16GB） |
| `Read-only file system: '/models/modules'` | `trust_remote_code` は `HF_HOME/modules` に書く。`HF_MODULES_CACHE` を `/tmp` に向ける |
| 予算作成が `INVALID_ARGUMENT` | 請求先アカウントが JPY 建て。`currency_code` を指定しない |
| 予算作成が `SERVICE_DISABLED` | provider に `billing_project` / `user_project_override` が要る（消費先が自分のプロジェクトでない番号になっているのが目印） |
| サービスアカウント作成が `accessNotConfigured` | `iam.googleapis.com` の有効化と `depends_on`。**2回目の apply では通るので新規プロジェクトでしか踏まない** |

## 積み残し

- **生成物バケットがジョブにマウントされていない。** 権限は付与済みだが
  `volume_mounts` が無く、出力は `/tmp/output` に書かれて消える。
  時間の計測が目的なら実害はないが、生成物を残すなら追加が要る
- PR #6 へのコメント投稿ができていない。内容は `MEASURE_RESULT.md` にある
