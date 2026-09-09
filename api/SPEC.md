# アプリから叩くAPIの仕様

`roomplanner-web` が画像から3Dモデルを作るために叩くAPIの取り決め。
実装手順は `TASK.md` を参照。

## 決まっていること

| 論点 | 決定 | 理由 |
|---|---|---|
| 同期／非同期 | **非同期**（投げて後で取りに行く） | 生成に146〜533秒かかる。同期は物理的に無理 |
| 待ち時間中のUX | **アプリを閉じても結果が残る** | 9分はアプリを閉じるのに十分な長さ |
| 認証 | **Firebase 匿名認証** | ログイン画面なしで安定したIDが得られ、製品版までそのまま使える |
| 疎通確認 | **Cloud Run IAM**（未認証拒否のまま） | コード変更なしで完全に閉じられる |
| 実行方式 | **Cloud Run Job のまま**（サービス化しない） | 1セッションあたりの生成数が未確定で、損得を判断できない |
| 生成物の受け渡し | **署名付きURL**（1時間） | base64 は 4.8MB を 6.4MB に膨らませるだけ |
| 回数制限 | **1日10回 / uid** | 1回$0.172 |
| 状態の保存先 | **GCS のファイルのみ** | 書き込みは1ジョブ数回、読みはポーリングだけ。Firestore は要らない |

Cloud Tasks も使わない。Cloud Run Jobs の実行API自体がキューの役割を果たす。

## エンドポイント

### `POST /jobs`

```
Authorization: Bearer <Firebase ID トークン>
Content-Type: multipart/form-data
  image: JPEG または PNG（5MBまで）

→ 202 { "jobId": "job_7f3a2b..." }
```

| 状態 | 意味 |
|---|---|
| 400 | 画像が不正、または5MB超 |
| 401 | トークンが無効 |
| 403 | 許可リストに無い uid（限定公開中のみ） |
| 409 | このユーザーに実行中のジョブがある |
| 429 | 本日の上限（10回）、または起動回数の上限（20回）に達した |

**409 を返すのは意図的。** 同一ユーザーの並行実行を許すとGPUが同時に立ち上がり、
事故的に高くつく。

**判定は「確かめてから書く」ではなく、GCS の条件付き書き込みで枠を取る。**
`quota/{uid}/active.json` を `if_generation_match=0`（まだ無いときだけ作る）で
書き、失敗したら 409。確認と書き込みを分けると、その間に2件目が通ってしまう。
ジョブの起動には数秒かかるので窓が広く、ボタンの二度押しで踏む。

枠を取るのは**ジョブの起動より先**。起動できなかった場合は枠と回数の両方を戻す。

### `GET /jobs/{jobId}`

```
Authorization: Bearer <Firebase ID トークン>

→ 200 {
    "state": "queued" | "running" | "succeeded" | "failed",
    "createdAt": "2026-09-08T12:00:00Z",
    "modelUrl": "https://storage.googleapis.com/...",  // succeeded のみ、1時間有効
    "error": "..."                                      // failed のみ
  }
```

他人の `jobId` には 404 を返す（存在を隠すため 403 ではなく 404）。

### `GET /health`

認証不要。疎通確認用。

**`/healthz` は使えない。** Cloud Run の Google Frontend がこのパスを横取りし、
アプリに到達する前に HTML の404を返す（アプリ側で登録しても届かない）。
`Content-Type` を見ると切り分けられる（アプリなら `application/json`、
横取りされていれば `text/html`）。

## ブラウザから叩くための CORS

アプリ（GitHub Pages）とAPI（Cloud Run）は**別オリジン**なので、
CORS が無いとブラウザはリクエストを1本も通さない。`curl` では
起きないため、実装当初は抜けていた。

**2箇所に要る。片方だけでは動かない。**

| | 何のため |
|---|---|
| API層（`hunyuan3d-api`） | `POST /jobs` と `GET /jobs/{id}` |
| **生成物バケット** | **署名付きURLで GLB を取るのはブラウザ**。`storage.googleapis.com` は API層とは別オリジンで、three.js の `GLTFLoader` も内部で `fetch` を使うので同じ制約を受ける |

許可するオリジンは `infra/variables.tf` の `allowed_origins` に列挙する。
ワイルドカードにはしない。

```
https://kmykprn.github.io   GitHub Pages
http://localhost:5173       ローカル開発（vite dev）
capacitor://localhost       将来 iOS アプリにするとき
```

`Authorization` ヘッダ付きの multipart なので、ブラウザは本番リクエストの前に
**preflight（OPTIONS）** を投げる。許可ヘッダに `Authorization` と
`Content-Type` が要る。

**`allow_credentials` は false のまま。** 認証は Cookie ではなく Bearer トークンで
行っており、これは別オリジンのページからは付けられない（他人の `localStorage` を
読めないため）。true にするとワイルドカードが使えなくなるうえ、Cookie を送る
意図だと誤読される。

## 回数の数え方 ── 失敗は数えない

**こちら都合の失敗で利用者の枠を失わせない。** 2つ数える。

| | 意味 | 失敗したとき |
|---|---|---|
| `count` | 成功として数える回数（上限10） | **戻す** |
| `attempts` | 起動した回数（上限20） | **戻さない** |

`attempts` が要るのは、失敗が必ず戻るだけだと**生成に向かない画像で延々と
再試行でき、そのたびに GPU が起動する**ため。1回あたり約39円かかる。

戻すのは、状態を見にきたとき（`GET /jobs/{jobId}`）に失敗が確定した時点。
二重に戻さないよう `status.json` に `quotaRestored` を記録する。

**商用化するときもこの構造のまま使える。** `count` を「クレジット残高」に
読み替えればよく、判定の位置も変わらない。

## GCS のレイアウト

生成物バケット（`...-hunyuan3d-outputs`）を状態の置き場としても使う。

```
jobs/{jobId}/input.png       クライアントが上げた画像
jobs/{jobId}/status.json     状態
jobs/{jobId}/model.glb       成果物
quota/{uid}/{YYYY-MM-DD}.json  当日の回数（count と attempts）
quota/{uid}/active.json      実行中のジョブ（並行実行を防ぐ排他ロック）
```

**許可リストは別のバケット**（`...-hunyuan3d-config`）に置く。

```
config/allowed_uids.json     限定公開中の許可リスト
```

同居させない理由は、API のサービスアカウントが生成物バケットに
`objectAdmin` を持つため。**門番を門番自身が書き換えられる状態を作らない。**
API はこのバケットに読み取り権限しか持たず、書き込みは運用者が行う。

`status.json`:

```json
{
  "jobId": "job_7f3a2b...",
  "uid": "AbCdEf...",
  "state": "running",
  "createdAt": "2026-09-08T12:00:00Z",
  "updatedAt": "2026-09-08T12:04:31Z",
  "executionName": "projects/.../executions/hunyuan3d-job-x7k2m",
  "error": null
}
```

## 失敗の扱い ── ここは既存のバグを潰す箇所

`api_server.py` の `/status` は**ファイルの有無しか見ないため、失敗すると
永遠に `processing` を返す**。同じ作りにしない。

1. ジョブが開始時に `running`、終了時に `succeeded` / `failed`（理由つき）を書く
2. **それでも書けずに死ぬ場合がある**ので、API層は `updatedAt` が15分以上古い
   ジョブについて Cloud Run の execution 状態を照会し、死んでいれば `failed` にする

2 が必要なのは、実測で **OOM による `signal 9`** を実際に踏んでいるため。
あの落ち方ではジョブ自身は何も書けない。

## 入出力

| | 仕様 |
|---|---|
| 入力形式 | **JPEG / PNG / WebP / HEIC・HEIF** |
| 形式の判定 | **Content-Type ではなく実際にデコードして判定する** |
| EXIF回転 | サーバー側で反映する |
| クライアント側で縮小 | 長辺1024px、JPEG 品質0.85（通常300KB以下になる） |
| サーバー側でも縮小 | 長辺1024pxを超えていれば縮める |
| サーバー側の上限 | 5MB |
| 保存形式 | PNG に正規化して `input.png` に置く |
| 出力 | GLB 約4.8MB / テクスチャ2048px |
| 背景除去 | サーバー側（`rembg` が既に入っている） |

**HEIC を受けるのは iPhone が既定でこの形式を使うため。** WebP は Android の
ブラウザから来ることがある。

**Content-Type を信用しない。** 実際に「PNG を名乗る壊れたファイル」が検証を
すり抜けてGPUを起動し、10分後に失敗した（約39円の無駄）。受け付けの時点で
デコードできるか確かめれば、GPUを起動する前に 400 で弾ける。

生成側は PNG しか受け取らないので、**GPUイメージには HEIC の対応を入れていない。**
形式の吸収はすべてAPI層で完結させている。

## 限定公開から製品版への移行

**捨てる実装が無いように組む。**

| | 限定公開（いま） | 製品版 |
|---|---|---|
| Cloud Run の入口 | 未認証拒否（IAM） | 未認証許可 |
| uid の判定 | `config/allowed_uids.json` に載っているか | 判定しない |
| 回数制限 | 10回/日 | そのまま（値は調整） |
| App Check | 無し | **追加する** |

移行時に変わるのは**入口の設定と、許可リスト判定を飛ばすフラグだけ**。
コードの作り直しは発生しない。

`allowed_uids.json` を GCS に置くのは、uid を足すたびに再デプロイしないため。

## 受け入れ確認の観点

変更を入れたら、少なくとも以下を実機で確認する。

### 認可

- [ ] トークン無し → **401**
- [ ] 不正なトークン → **401**
- [ ] 許可リストに無い uid → **403**
- [ ] 他人の `jobId` を引く → **404**（403 ではない。存在を隠す）
- [ ] 許可リストが空／読めない → **全員 403**（fail-closed であること）

### 回数

- [ ] **生成が失敗したとき、回数（`count`）が減っていないこと** ← 最重要
  - `status.json` が `failed` になったあと `GET /jobs/{id}` を叩き、
    `quota/{uid}/{日付}.json` の `count` が消費前に戻っていること
  - 同じ応答を2回取っても二重に戻らないこと（`quotaRestored`）
- [ ] 成功したときは `count` が減らずに残ること
- [ ] 失敗を繰り返すと `attempts` の上限で **429** になること
- [ ] 409（実行中）と 400（画像が不正）は回数を消費しないこと

### 並行実行

- [ ] 同時に複数投げて、**GPU が1本しか起動しないこと**
- [ ] 完了後は次を投げられること（枠が解放されている）

### ブラウザから

- [ ] preflight（OPTIONS）が 200 で、許可オリジンが返ること
- [ ] 許可外のオリジンには CORS ヘッダが付かないこと
- [ ] **署名付きURLから GLB をブラウザで取得できること**（バケット側の CORS）

### 生成物

- [ ] `textured_mesh` 相当であること（`white_mesh` ではない）

## この仕様に含めていないもの

- **App Check**（製品版で追加。「本当に自分のアプリからの呼び出しか」を証明する）
- **`appState` の永続化**（`roomplanner-web` 側の別作業。これが無いと、
  生成した家具を置いてもリロードで部屋ごと消える）
- **サービス化**（実利用の生成頻度データが出てから判断）
