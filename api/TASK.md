# 作業依頼: アプリから叩けるAPIを作る

仕様は `SPEC.md`。ここには**手順**を書く。WSL2 実機で実行する。

**前提: PR #6（`feat/terraform-infra`）がマージ済みであること。**
このブランチはその上に載っている。

## ゴール

`roomplanner-web` から画像を投げると、GLB が返ってくる状態にする。

3段階に分ける。**各段階の終わりで必ず動作確認し、次に進む前に報告すること。**
段階1と2は認証を一切通さないので、先に配線だけ確かめられる。

```
段階1  API層を作る（Cloud Run IAM で閉じたまま、curl で疎通）
段階2  ジョブをAPIから起動できる形にする（画像を渡し、GLBを受け取る）
段階3  Firebase 匿名認証を入れて、アプリから叩けるようにする
```

---

# 段階1: API層を作り、curl で疎通確認する

この段階では**認証コードを書かない。** Cloud Run の IAM で閉じておけば、
自分以外は叩けない。

## 1-1. API を実装する

`api/` に FastAPI で作る。既存の `api_server.py` と同じ枠組みなので合わせる。

必要なもの（`api/requirements.txt`）:

```
fastapi
uvicorn[standard]
google-cloud-storage
google-auth
python-multipart
```

`firebase-admin` は段階3で足す。**この段階では入れない。**

実装する3つ:

| | 中身 |
|---|---|
| `GET /healthz` | `{"ok": true}` を返すだけ |
| `POST /jobs` | 画像を検証 → `jobId` を採番 → GCS に `input.png` と `status.json`（`queued`）を書く → **まだジョブは起動しない**（段階2で足す） |
| `GET /jobs/{id}` | `status.json` を読んで返す。`succeeded` なら署名付きURLを付ける |

uid はこの段階では固定値（`"local-test"`）でよい。段階3で差し替える。

**署名付きURLの発行には注意点がある。** Cloud Run のサービスアカウントは
秘密鍵を持たないため、`generate_signed_url` がそのままでは失敗する。
`iam.serviceAccounts.signBlob` 権限を付けたうえで、IAM SignBlob API を
使う形にする（`google.auth.iam.Signer`）。**ここは詰まる可能性が高い。**

## 1-2. Terraform に API サービスを足す

`infra/api.tf` を新規作成する。

- `google_cloud_run_v2_service` — CPU のみ、GPU なし、`min_instance_count = 0`
- 専用のサービスアカウント（GPU ジョブ用とは別にする）
- 生成物バケットへの `roles/storage.objectAdmin`
- **`roles/run.invoker` は自分のアカウントにだけ付ける。`allUsers` には付けない**
- 自分自身への `roles/iam.serviceAccountTokenCreator`（署名付きURL用）

APIサービス用のAPI有効化が要る:

```
run.googleapis.com          （既に有効）
iamcredentials.googleapis.com  ← 署名付きURLに要る。追加する
```

## 1-3. 確認する

```bash
URL=$(gcloud run services describe hunyuan3d-api --region asia-southeast1 --format='value(status.url)')
TOKEN=$(gcloud auth print-identity-token)

# 認証なしで弾かれること（403 が返れば正しい）
curl -s -o /dev/null -w '%{http_code}\n' "$URL/healthz"

# 認証ありで通ること
curl -H "Authorization: Bearer $TOKEN" "$URL/healthz"

# ジョブを作る
curl -H "Authorization: Bearer $TOKEN" -F image=@assets/example_images/052.png "$URL/jobs"

# 状態を見る（queued のはず）
curl -H "Authorization: Bearer $TOKEN" "$URL/jobs/<返ってきたjobId>"
```

**「認証なしで403」を必ず確認すること。** ここが開いていると、
以降どれだけ作り込んでも財布が開いたままになる。

---

# 段階2: ジョブをAPIから起動する

## 2-1. ジョブ側にラッパーを足す

現在の `job.tf` は `--input-image assets/example_images/052.png` を固定で渡している。
これを、GCS に置かれた画像を読む形にする。

`worker_entrypoint.py` を新規作成（数十行）:

1. 環境変数 `JOB_ID` を読む
2. `status.json` を `running` に更新
3. `/jobs/{JOB_ID}/input.png` を入力に生成を実行
4. 出力を `/jobs/{JOB_ID}/model.glb` に置く
5. `status.json` を `succeeded` に更新
6. **例外は握りつぶさず、`failed` と理由を書いてから終了する**

## 2-2. 生成物バケットを read-write でマウントする

`job.tf` に2本目の GCS ボリュームを足す。**これは PR #6 で積み残しとして
挙がっていた件でもある**（現状は出力が `/tmp/output` に書かれて消える）。

- バケット: 生成物バケット
- `read_only = false`
- マウント先: `/jobs`

重みのマウント（`/models`、read-only、キャッシュ設定つき）は**そのまま残すこと**。
あのキャッシュ設定を消すとモデルのロードが25分に戻る。

## 2-3. API からジョブを起動する

`POST /jobs` の最後に、Cloud Run Jobs の実行APIを呼ぶ処理を足す。

```
POST https://run.googleapis.com/v2/projects/{PROJECT}/locations/{REGION}/jobs/{JOB}:run
{
  "overrides": {
    "containerOverrides": [{ "env": [{ "name": "JOB_ID", "value": "job_..." }] }]
  }
}
```

返ってきた execution 名を `status.json` に記録する（失敗検知に使う）。

APIのサービスアカウントに `roles/run.developer` が要る。

## 2-4. 失敗検知を足す

`GET /jobs/{id}` で、`updatedAt` が15分以上古く、かつ `state` が
`queued` / `running` のとき、execution の状態を照会して死んでいれば `failed` にする。

**OOM で `signal 9` に落ちる実績があるため、これは必須。**

## 2-5. 確認する

```bash
curl -H "Authorization: Bearer $TOKEN" -F image=@<任意の家具写真> "$URL/jobs"
# 10分ほど待ちながらポーリング
watch -n 30 "curl -s -H 'Authorization: Bearer $TOKEN' $URL/jobs/<jobId>"
```

`succeeded` になり、`modelUrl` から GLB がダウンロードでき、
**`textured_mesh` 相当（`white_mesh` ではない）であること**を確認する。

わざと壊れた画像を投げて `failed` になることも確認すること。

---

# 段階3: Firebase 匿名認証でアプリから叩く

## 3-1. Firebase を有効にする

既存の GCP プロジェクトにそのまま乗せられる。

```bash
# コンソールで既存プロジェクトに Firebase を追加し、匿名認証を有効化する
# （Authentication → Sign-in method → 匿名 → 有効）
```

**ブラウザ操作なので人間に依頼すること。**

## 3-2. API 側でトークンを検証する

`firebase-admin` を足し、`Authorization: Bearer` の ID トークンを検証して uid を得る。

- `config/allowed_uids.json` に uid が無ければ **403**
- 当日の回数が10回以上なら **429**
- 実行中のジョブがあれば **409**

**許可リストを GCS に置くのは、uid を足すたびに再デプロイしないため。**

## 3-3. Cloud Run の入口を開ける

```bash
# allUsers に invoker を付ける（Firebase 認証が中で効いている状態にしてから）
```

**順序を間違えないこと。** 3-2 が動く前にここを開けると、誰でも叩ける状態になる。

## 3-4. クライアント側（`roomplanner-web`）

別リポジトリの作業。

1. `firebase/app` と `firebase/auth` を足し、起動時に `signInAnonymously()`
2. `src/platform/` に `api.ts` を新設し、**ネットワーク越しの処理をここに隔離**する
   （`CLAUDE.md` の「OS依存の処理は `src/platform/` 経由でのみ呼ぶ」に倣う）
3. 画像は送信前に**長辺1024pxへ縮小**する（canvas で十分）
4. `jobId` を `localStorage` に保存し、起動時にポーリングを再開する
5. 完成した GLB は **Cache Storage に明示的に保存**する

**既存の PWA 設定の `urlPattern: /\.glb$/` は署名付きURLにマッチしない**
（クエリパラメータが付くため）。Service Worker 任せにせず、アプリから入れること。

自分の uid を画面のどこかに表示しておくと、許可リストに足すときに楽。

---

## 報告してほしいこと

**段階ごとに**、このPRにコメントで。

1. その段階の確認コマンドの結果
2. 詰まった箇所と、実際のエラー全文
3. `SPEC.md` から変えた点があれば、その理由

## 未検証の箇所（設計した側が実機で試していない）

| 箇所 | 懸念 |
|---|---|
| 署名付きURLの発行 | Cloud Run のサービスアカウントは秘密鍵を持たない。IAM SignBlob 経由にする必要があり、**ここが一番詰まりそう** |
| `executions:run` の env 上書き | ドキュメント上は `overrides.containerOverrides.env` で可能。実際に効くかは未確認 |
| 生成物バケットの read-write マウント | 重みの read-only マウントと共存できるか未確認 |
| Firebase 匿名認証の uid の安定性 | ストレージを消すと変わる。許可リストの運用に影響する |

## 判断が要るときは止めること

- **`allUsers` に invoker を付けるのは段階3-2が動いてから。** 順序を守る
- 回数制限の値を変える
- リージョンを変える
- ジョブをサービスに変える（**この手順書の範囲外**）
