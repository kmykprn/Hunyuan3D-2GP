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
| 429 | 本日の上限（10回）に達した |

**409 を返すのは意図的。** 同一ユーザーの並行実行を許すとGPUが同時に立ち上がり、
事故的に高くつく。

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

### `GET /healthz`

認証不要。疎通確認用。

## GCS のレイアウト

生成物バケット（`...-hunyuan3d-outputs`）を状態の置き場としても使う。

```
jobs/{jobId}/input.png       クライアントが上げた画像
jobs/{jobId}/status.json     状態
jobs/{jobId}/model.glb       成果物
quota/{uid}/{YYYY-MM-DD}.json  当日の実行回数
config/allowed_uids.json     限定公開中の許可リスト
```

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
| 入力形式 | JPEG / PNG |
| クライアント側で縮小 | 長辺1024px、JPEG 品質0.85（通常300KB以下になる） |
| サーバー側の上限 | 5MB |
| 出力 | GLB 約4.8MB / テクスチャ2048px |
| 背景除去 | サーバー側（`rembg` が既に入っている） |

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

## この仕様に含めていないもの

- **App Check**（製品版で追加。「本当に自分のアプリからの呼び出しか」を証明する）
- **`appState` の永続化**（`roomplanner-web` 側の別作業。これが無いと、
  生成した家具を置いてもリロードで部屋ごと消える）
- **サービス化**（実利用の生成頻度データが出てから判断）
