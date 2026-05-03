# Scale Up With New Videos

Scale-up starts from a user-prepared source RGB video and a matching foreground
mask. Large-scale video search, downloading, and filtering are outside this
repo; for that stage, refer to
[Animal-in-Motion](https://github.com/briannlongzhao/Animal-in-Motion), then
bring the selected source video and source mask here.

## Inputs

You need:

- one source RGB video with a single main foreground subject
- one source foreground mask video, or a directory of 45 PNG masks
- one target trajectory, specified by yaw/pitch/roll/scale

The source RGB and source mask are resized/sampled into a 45-frame 832x480
clip. Wan generates `generated_prompt.txt` from the source clip during the run,
and the packaged case stores it as `prompt.txt`.

## Environments

```bash
bash scripts/env/create_env.sh redirect4d-bench
bash scripts/env/create_reconstruction_env.sh redirect4d-recon
bash scripts/models/download_reconstruction_checkpoints.sh required
bash scripts/env/create_sam3_env.sh redirect4d-sam3
bash scripts/env/check_all_envs.sh
```

## Concrete Sample Command

This uses the public bear sample:

```bash
SAMPLE_TRACK=data/redirect4d_bench/sample/tracks/bear_NnAlfavy2us_003_001_seq1

python scripts/pipeline/run_scale_up_case.py \
  --case-name bear_sample_scaleup \
  --input-video $SAMPLE_TRACK/video.mp4 \
  --mask-video $SAMPLE_TRACK/mask_video.mp4 \
  --yaw 120 \
  --pitch 0 \
  --roll 0 \
  --scale 1 \
  --workspace outputs/scale_up/bear_sample_scaleup \
  --gpu 0
```

## Generic Command

```bash
python scripts/pipeline/run_scale_up_case.py \
  --case-name my_case \
  --input-video /path/to/source_video.mp4 \
  --mask-video /path/to/source_mask.mp4 \
  --yaw 120 \
  --pitch 0 \
  --roll 0 \
  --scale 1 \
  --workspace outputs/scale_up/my_case \
  --gpu 0
```

For a directory of mask PNGs, replace `--mask-video` with:

```bash
--mask-dir /path/to/source_masks
```

For a custom camera path, replace the four trajectory flags with
`--trajectory-json /path/to/trajectory.json`.

## Output

The final case is written to:

```text
outputs/scale_up/<case-name>/release/tracks/<case-name>/
  mask_video.mp4
  masks/00000.png ... 00044.png
  pointcloud/
    global_background.ply
    00000/
    ...
  redirected/<trajectory>/
    trajectory.json
    depth.mp4
    mask.mp4
    prompt.txt
```

By default, source RGB is not included in the release-style output. Add
`--include-source-rgb` only if you have permission to redistribute the source
RGB.
