#!/bin/bash
# Re-run Wan2.2 Stage 2 for a hand-picked list of trajectories with a fresh seed.
# Runs one job per GPU in parallel. Designed to be launched inside tmux.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${R4D_RECON_PY:-python}"

SEED="${1:-42}"

JOBS=(
  "camel_IJ4YajWrDcA_027_001_seq1/yaw_-120_pitch_-10_roll_0_scale_1p1"
  "cat_l-Tzteg9ksM_007_001_seq1/yaw_100_pitch_-30_roll_0_scale_1p8"
  "dancer_0GwFqyM-qMM_015_001_seq1/yaw_-120_pitch_0_roll_0_scale_1"
  "dancer_DDuz_HVkvR8_011_001_seq2/yaw_-100_pitch_0_roll_0_scale_1p1"
  "dancer_KOy8OCTlZaw_015_001_seq1/yaw_-110_pitch_-10_roll_0_scale_1p1"
  "zebra_qvRTslcIeSk_002_001_seq1/yaw_120_pitch_0_roll_0_scale_1"
  "dancer_0GwFqyM-qMM_015_001_seq1/yaw_120_pitch_0_roll_0_scale_1"
)

mkdir -p batch_logs
echo "=== Re-running ${#JOBS[@]} trajectories with seed=$SEED ==="

# Step 1: clear old output_video + preview copies so the re-run actually writes fresh
for tt in "${JOBS[@]}"; do
  track="${tt%%/*}"
  traj="${tt#*/}"
  rm -vf "outputs/rendering/${track}/${traj}/inference/output_video.mp4"
  rm -vf "outputs/rendering_preview/${track}__${traj}.mp4"
done

# Step 2: launch 7 parallel jobs, each pinned to one GPU
pids=()
for i in "${!JOBS[@]}"; do
  tt="${JOBS[$i]}"
  track="${tt%%/*}"
  traj="${tt#*/}"
  gpu=$i
  logf="batch_logs/rerun_seed${SEED}_${track}__${traj}.log"
  echo "[gpu$gpu] launching $tt"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/2_0_Wan2.2-VACE-Fun-A14B.py \
      --data_dir "outputs/rendering/${track}/${traj}" \
      --use_saved_prompt \
      --seed "$SEED" \
      > "$logf" 2>&1 &
  pids+=($!)
done

# Step 3: wait for all
echo "Waiting for ${#pids[@]} jobs (~25 min)..."
rc_total=0
for i in "${!pids[@]}"; do
  wait "${pids[$i]}"
  rc=$?
  tt="${JOBS[$i]}"
  if [ "$rc" -eq 0 ]; then
    echo "[OK ] $tt"
  else
    echo "[FAIL rc=$rc] $tt (see batch_logs/rerun_seed${SEED}_...)"
    rc_total=$((rc_total+1))
  fi
done

# Step 4: copy new output_video.mp4 back to rendering_preview/
echo "=== Copying new videos to rendering_preview/ ==="
for tt in "${JOBS[@]}"; do
  track="${tt%%/*}"
  traj="${tt#*/}"
  src="outputs/rendering/${track}/${traj}/inference/output_video.mp4"
  dst="outputs/rendering_preview/${track}__${traj}.mp4"
  if [ -f "$src" ]; then
    cp -v "$src" "$dst"
  else
    echo "[MISS] $src — job likely failed"
  fi
done

echo "=== Done. Failures: $rc_total / ${#JOBS[@]} ==="
