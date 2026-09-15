# 切り抜き API

写真から家具だけを切り抜いて、透過 PNG を返す。roomplanner-web の「作る」（写真から・商品の URL から）が叩く。

**預けて、あとで取りに行く**形（`/cutout-jobs`）。3D 生成（`api/`）と同じ考え方で、
受け付けたらすぐ受付番号を返し、処理はサーバー側で進む。画面は受付番号で状態を見に来る。
iPhone は PWA を裏に回すと数秒で通信を切るので、1 本の接続で 30〜50 秒待たせる形（旧 `/cutouts`）は
楽天のページに URL をコピーしに行く間に切れていた。

## 入口

```
POST {cutout_url}/cutout-jobs
Authorization: Bearer <Firebase ID トークン>
Content-Type: multipart/form-data; image=<JPEG / PNG / WebP / HEIC, 5MB まで>

202 {"id": "<受付番号>", "expectedSeconds": 6.4}
400 … 画像が読めない・対応外の形式・5MB 超。回数は数えない
401 … トークンが無い・無効・Google に紐づいていない
409 … 同時に処理された。やり直せばよい
429 … 本日の上限（既定 50 枚）に達した
503 … 預かる仕組み（Cloud Tasks）が設定されていない・積めなかった。回数は戻す

GET {cutout_url}/cutout-jobs/{id}
Authorization: Bearer <Firebase ID トークン>

200 {"phase": "queued" | "running" | "done" | "failed", "expectedSeconds": 6.4, "elapsed": 3.2, "error": null}
404 … 無い・他人のもの・消えた（7 日）

GET {cutout_url}/cutout-jobs/{id}/result
200 image/png … 透明な余白は切り落としてある（長辺 1024px まで）
404 … まだ done でない
```

`elapsed` はサーバーの時計で測った、いまの工程に入ってからの秒数（queued は受付から、
running は処理開始から）。端末の時計とずれても円の進み方が狂わないよう、画面はこれを使う。
`expectedSeconds` は推論の直近 5 回の平均。

### 処理の流れ

1. 受付: 検証 → 回数を数える → 入力を `jobs/{uid}/{id}/input.jpg` に置く（長辺 1024・向きを直した JPEG）
   → `status.json` を queued で書く → Cloud Tasks に `POST /cutout-jobs/{uid}/{id}/run` を積む → 202
2. 処理（Cloud Tasks がこのサービス自身を叩く。別のリクエストなので CPU が付く）:
   running → 推論 → `result.png` → done。推論の失敗は failed にして **200 で返す**（やり直しても同じ。回数は戻す）。
   台数の上限（429/503）や処理中の消滅は Cloud Tasks が再試行する。done / failed なら何もしない
3. `/run` は Cloud Tasks が付ける OIDC トークン（サービスアカウント、audience はサービスの URL）を検証し、
   それ以外は 403。環境変数 `TASK_QUEUE` / `SELF_URL` / `TASK_SERVICE_ACCOUNT` は Terraform（infra/cutout.tf）が入れる

### 旧: 1 本の接続で待つ形（消してよい）

画面が `/cutout-jobs` に切り替わったら消す。

```
POST {cutout_url}/cutouts
Authorization: Bearer <Firebase ID トークン>
Content-Type: multipart/form-data; image=<JPEG / PNG / WebP / HEIC, 5MB まで>

200 application/x-ndjson … Accept: application/x-ndjson を付けたとき。工程を 1 行ずつ流す（下記）
200 image/png            … Accept を付けないとき。切り抜きだけを返す（古い画面向け。画面が流す形に切り替わったら消してよい）
400             … 画像が読めない・対応外の形式・5MB 超。回数は数えない
401             … トークンが無い・無効・Google に紐づいていない
409             … 同時に処理された。やり直せばよい
429             … 本日の上限（既定 50 枚）に達した
```

### 200 の中身

1 行 1 JSON。順に流れる。行が届くまでの空白が「起動待ち（コールドスタート）」なので、
画面側はそれを区別して出せる。

```
{"phase":"received"}                                   受け付けた（起動待ちが終わった）
{"phase":"cutting","expectedSeconds":6.4,"elapsed":0}  推論中。expectedSeconds は直近 5 回の平均秒数。円の進み方の目安
{"phase":"cutting","expectedSeconds":6.4,"elapsed":2}  同じ行を 2 秒ごとに繰り返す（elapsed は工程に入ってからの秒数）
{"phase":"finishing","elapsed":0}                      余白の切り落としと PNG 化（0.3 秒）。長引けば同様に繰り返す
{"phase":"done","png":"<base64>"}                      切り抜き。透明な余白は切り落としてある（長辺 1024px まで）
{"phase":"failed","error":"…"}                         推論中の失敗。done の代わりに来る。回数は戻す
```

推論そのものは途中経過を出せない（onnxruntime の中で止まる）ので、工程はこの 4 つまで。

同じ工程の行を繰り返すのは、接続を黙らせないため。ヘッダを送ったあと 20 秒ほど無通信が続くと、
途中の経路が応答を閉じてしまい、画面には done が届かないことがあった（iPhone、HTTP/3）。
画面側は工程が変わったときだけ円の基準時刻を動かし、繰り返しの行では動かさない。
間隔は環境変数 HEARTBEAT_SECONDS（既定 2）。

`GET /health` は `{"ok": true}`。

## 誰が叩けるか

**Google ログイン済みのアカウントだけ**（`email_verified` のメールを持つトークン）。
匿名アカウントは 401。

Web の API キーは公開値なので、匿名アカウントは `curl` だけで作れる。それを通すと
アプリを触っていない人でも叩けてしまう。Google アカウントを要ることにすれば、
乱用の単位が「アカウント 1 つ」になり、1 日の上限と合わせて費用が読める。
将来課金するときの「誰か」もこのメールになる。

許可リスト（`config/allowed_*.json`）は**見ない**。1 枚 0.1 円程度なので門番は要らない。

## 回数

uid ごとに 1 日 `DAILY_LIMIT` 枚（UTC の日付で区切る）。カウンタは
`gs://{project}-cutout-state/quota/{uid}/{YYYY-MM-DD}.json` に置き、generation の
条件付き書き込みで同時の投稿でも数え落とさない。こちら都合の失敗（推論の例外）は戻す。

## モデル

BiRefNet-general-lite（MIT、rembg の配布 ONNX、224MB）。入力 1024×1024 固定、CPU で
4 vCPU なら約 5 秒。メモリのピークは 6.4GB（onnxruntime のアリーナを切った値）。

前処理は rembg と同じ: 1024×1024 に伸ばし、ImageNet の平均・分散で正規化。
後処理: sigmoid → 最小・最大で引き伸ばし → 元の大きさに戻してアルファに入れる → 余白を切り落とす。

比べた 4 モデルのうち、複数の物が写った写真（トレーの上の皿、通販サイトのスクショ）で
「家具だけ」を切れたのはこれだけだった。u2netp / silueta は影が残り、
isnet-general-use は写っている物を全部残す。

## 費用

4 vCPU × 6 秒 × $0.000018 + 12 GiB × 6 秒 × $0.000002 ≈ $0.0006 ≈ 0.1 円/枚。
無料枠（月 18 万 vCPU 秒）で月 7,000 枚まで無料。上限は `max_instance_count = 2` で頭打ち。
