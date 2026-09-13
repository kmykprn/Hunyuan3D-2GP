# 切り抜き API

写真から家具だけを切り抜いて、透過 PNG を返す。roomplanner-web の「写真から家具を作る」が叩く。

3D 生成（`api/`）と違い、**同期で数秒で返る**。ジョブも待機列も無い。

## 入口

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
{"phase":"received"}                       受け付けた（起動待ちが終わった）
{"phase":"cutting","expectedSeconds":6.4}  推論中。直近 5 回の平均秒数。円の進み方の目安
{"phase":"finishing"}                      余白の切り落としと PNG 化（0.3 秒）
{"phase":"done","png":"<base64>"}          切り抜き。透明な余白は切り落としてある（長辺 1024px まで）
{"phase":"failed","error":"…"}             推論中の失敗。done の代わりに来る。回数は戻す
```

推論そのものは途中経過を出せない（onnxruntime の中で止まる）ので、工程はこの 4 つまで。

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
