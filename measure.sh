#!/usr/bin/env bash
#
# Hunyuan3D-2GP の生成時間とVRAM使用量を測る。
# Cloud Run(L4 24GB) に載せる前提で、profile ごとの速度差を確認するのが目的。
#
# 使い方:
#   Hunyuan3D-2GP のリポジトリ直下に置いて、仮想環境を有効にした状態で
#   bash measure.sh
#
# 結果は measure_result.txt に追記される。
#
set -u

IMAGE="${1:-assets/example_images/052.png}"
OUT="measure_result.txt"

echo "計測結果を $OUT に書き出します"
{
  echo "════════════════════════════════════════"
  echo "計測日時: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "入力画像: $IMAGE"
  echo
  echo "── 環境 ──"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
    || echo "nvidia-smi が見つかりません"
  python -c "import torch; print(f'torch {torch.__version__} / CUDA {torch.version.cuda} / 利用可能 {torch.cuda.is_available()}')" 2>&1
  echo "システムRAM: $(free -g 2>/dev/null | awk '/^Mem:/{print $2 "GB"}' || echo '不明')"
  echo
} >> "$OUT"

# 実行中の VRAM を 1 秒間隔で記録し、最大値を返す
run_case() {
  local label="$1"; shift
  echo "▶ $label を実行中..."

  local vram_log
  vram_log=$(mktemp)
  ( while true; do
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$vram_log" 2>/dev/null
      sleep 1
    done ) &
  local watcher=$!

  local start end status
  start=$(date +%s.%N)
  # 出力は捨てずに残す（失敗時の原因調査用）
  "$@" > "measure_${label}.log" 2>&1
  status=$?
  end=$(date +%s.%N)

  kill "$watcher" 2>/dev/null; wait "$watcher" 2>/dev/null

  local elapsed peak
  # bc は Ubuntu/WSL の標準構成に入っていないため awk で計算する
  elapsed=$(awk -v e="$end" -v s="$start" 'BEGIN{printf "%.3f", e - s}')
  peak=$(sort -n "$vram_log" 2>/dev/null | tail -1)
  rm -f "$vram_log"

  {
    printf "── %s ──\n" "$label"
    if [ "$status" -eq 0 ]; then
      printf "  結果      : 成功\n"
    else
      printf "  結果      : 失敗 (終了コード %s / measure_%s.log を参照)\n" "$status" "$label"
    fi
    printf "  所要時間  : %.1f 秒\n" "$elapsed"
    printf "  VRAM最大  : %s MiB\n" "${peak:-取得できず}"
    echo
  } >> "$OUT"

  echo "  → $(printf '%.1f' "$elapsed") 秒 / VRAM最大 ${peak:-?} MiB"
}

# profile: 1=HighRAM_HighVRAM(最速) 〜 5=VerylowRAM_LowVRAM(最省)
# Cloud Run は秒課金なので、速い側（小さい番号）が安くなる。
# 既定の 3 と、L4 24GB で狙いたい 1・2 を比較する。
for p in 3 2 1; do
  run_case "profile${p}_shape" \
    python minimal_demo_mmgp.py --input-image "$IMAGE" --output ./output --profile "$p"

  run_case "profile${p}_texture" \
    python minimal_demo_mmgp.py --input-image "$IMAGE" --output ./output --texture --profile "$p"
done

{
  echo "── モデルの重み ──"
  # イメージサイズの見積もりに使うので、Hunyuan3D の重みだけを測る。
  # ~/.cache/huggingface 全体では他プロジェクトの重みまで数えてしまう。
  # blob は複数スナップショットで共有されるため -L で実体を辿る。
  # 同じモデルでも過去に取得したスナップショットが残っていることがあり、
  # それらを合算すると実際に必要な容量より大幅に大きく出る。
  # どのスナップショットが今回の実行で使われたかは上の各ログに出ているので、
  # 突き合わせられるようスナップショット単位で出す。
  echo "  Hunyuan3D の重み（スナップショット単位）:"
  for d in ~/.cache/huggingface/hub/models--tencent--Hunyuan3D-2*; do
    [ -d "$d" ] || continue
    echo "    $(basename "$d"):"
    for snap in "$d"/snapshots/*/; do
      [ -d "$snap" ] || continue
      echo "      $(basename "$snap" | cut -c1-12)  $(du -shL "$snap" 2>/dev/null | cut -f1 || echo '不明')"
    done
  done
  echo "  HuggingFaceキャッシュ全体（参考）: $(du -sh ~/.cache/huggingface 2>/dev/null | cut -f1 || echo '不明')"
  echo
  echo "── 生成物 ──"
  ls -la output/ gradio_cache/*/ 2>/dev/null | grep -E '\.glb|\.obj' | head -10
  echo
} >> "$OUT"

echo
echo "════════ 完了 ════════"
cat "$OUT"
