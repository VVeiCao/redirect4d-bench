# Redirect4D-Bench

Redirect4D-Bench evaluates camera redirection for monocular dynamic videos.
The public Hugging Face dataset contains generated benchmark assets. Source RGB
is included only for the two public sample tracks; full-dataset source RGB is
provided separately through a gated Hugging Face dataset.

## Install

```bash
git clone --recursive https://github.com/VVeiCao/redirect4d-bench.git Redirect4D_Bench
cd Redirect4D_Bench

bash scripts/env/create_env.sh redirect4d-bench
conda activate redirect4d-bench
```

The base environment is enough for downloading data, validating folders, and
previewing the sample. Full evaluation and scale-up use two auxiliary
environments, created only when needed:

```bash
bash scripts/env/create_sam3_env.sh redirect4d-sam3
bash scripts/env/create_reconstruction_env.sh redirect4d-recon
```

The evaluation script automatically uses these two environments if they exist.

## Sample Quick View

Download the public sample:

```bash
hf download vveicao/redirect4d-bench \
  --repo-type dataset \
  --include 'sample/**' \
  --local-dir data/redirect4d_bench
```

Open the Viser preview:

```bash
python scripts/visualization/serve_pointcloud_viser.py \
  --dataset-root data/redirect4d_bench/sample \
  --port 8091
```

Forward port `8091` and open it in your browser.

The preview shows:

- animated foreground/background 4D point clouds
- source video when it exists in the selected dataset root
- source video mask
- target camera trajectory
- target mask and target depth for the active trajectory

![Redirect4D-Bench Viser sample interface](assets/viser_interface.png)

## Full Dataset

Download public assets:

```bash
hf download vveicao/redirect4d-bench \
  --repo-type dataset \
  --local-dir data/redirect4d_bench
```

Public layout:

```text
data/redirect4d_bench/
  metadata.json
  tracks.jsonl
  cases.jsonl
  sample/
  tracks/<track>/
    camera.json
    masks/*.png
    mask_video.mp4
    pointcloud/
    redirected/<trajectory>/
      trajectory.json
      mask.mp4
      depth.mp4
      prompt.txt
```

Meaning:

- `mask_video.mp4`, `masks/*.png`: source video mask
- `pointcloud/`: released 4D point clouds
- `trajectory.json`: target camera trajectory
- `redirected/<trajectory>/mask.mp4`: target pseudo-GT mask
- `redirected/<trajectory>/depth.mp4`: target pseudo-GT depth
- `prompt.txt`: frozen prompt used by the generation pipeline
- `metadata.json`, `tracks.jsonl`: YouTube id, crop boxes, frame ids, fps,
  camera intrinsics, and trajectory names for recovering source clips

Original source RGB is not redistributed. The public metadata can be used to
recover source clips from YouTube:

```bash
python scripts/data/download_original_videos.py \
  --metadata data/redirect4d_bench/metadata.json \
  --output-dir data/original_videos

python scripts/data/reconstruct_source_tracks.py \
  --metadata data/redirect4d_bench/metadata.json \
  --video-dir data/original_videos \
  --output-root data/reconstructed_source_tracks
```

The recovered RGB matches the canonical source used to construct the
benchmark at high quality (typical PSNR around 38 dB on bear/elephant
samples).

YouTube access depends on the user's network environment. If YouTube blocks
the download with bot/login verification, request the canonical source RGB
package from `dave.caowei@gmail.com`.

Validate the local data:

```bash
python scripts/data/validate_dataset.py \
  --dataset-root data/redirect4d_bench \
  --restricted-source-root data/reconstructed_source_tracks
```

### Dataset Visualization

The Viser only reads the folder passed to `--dataset-root`. For the public full
dataset, the source video panel stays empty because source RGB is not included.

```bash
python scripts/visualization/serve_pointcloud_viser.py \
  --dataset-root data/redirect4d_bench \
  --port 8091
```

## Evaluation

This public repo evaluates the Redirect4D-Bench object fidelity/localization
and camera-pose accuracy metrics. FID, FVD, CLIP, and VBench are not run here.

Install the evaluation environments once before running the benchmark:

```bash
bash scripts/env/create_sam3_env.sh redirect4d-sam3
bash scripts/env/create_reconstruction_env.sh redirect4d-recon
bash scripts/env/check_all_envs.sh
```

Generated videos should be named by case:

```text
<track>_<trajectory>.mp4
```

Use one folder per method:

```text
my_method/VIDEOS/
  bear_NnAlfavy2us_003_001_seq1_yaw_120_pitch_0_roll_0_scale_1.mp4
  elephant_4F0hzklQejU_010_001_seq1_yaw_-120_pitch_0_roll_0_scale_1.mp4
```

Evaluate that folder directly:

```bash
python scripts/evaluation/evaluate_user_method.py \
  --video-dir my_method/VIDEOS
```

By default this reads `data/redirect4d_bench`, evaluates every `.mp4` in
`my_method/VIDEOS`, and writes each run to a new timestamped folder:

```text
outputs/my_method/YYYYMMDD_HHMMSS/
```

Camera accuracy needs the reconstruction environment. For object fidelity,
methods submit only generated RGB videos; the evaluation script extracts their
target masks with the benchmark SAM3 propagation step and compares them to the
dataset pseudo-GT target mask. The extracted generated-video masks are written
next to the submitted videos:

```text
my_method/mask/<track>_<trajectory>.mp4
```

## Scale Up

Scale-up creates a new track in the same format as the released dataset.
For large-scale video search, downloading, and candidate filtering, please
refer to the [Animal-in-Motion](https://github.com/briannlongzhao/Animal-in-Motion)
collection pipeline and bring the selected source video and source mask into
this benchmark pipeline.

```text
source RGB video + source mask
-> 45-frame source clip and aligned source masks
-> source-scene reconstruction: foreground 4D point clouds + VIPE/LyRA background
-> target trajectory definition
-> pre-Wan target render: rendered_images.mp4, rendered_depths.mp4, rendered_mask.mp4
-> prompt generation / prompt freezing
-> Wan target RGB generation
-> MaskRefine target pseudo-GT mask from rough mask + post-Wan RGB
-> release-style case folder: source mask, point clouds, trajectory, depth, target mask, prompt
```

Concrete sample command:

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

Generic command:

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

The final release-style case is written under
`outputs/scale_up/<case-name>/release/tracks/<case-name>`. Wan generates a
prompt from the source clip during the run and stores it as `prompt.txt` in the
final case. For more options, see [docs/scale_up.md](docs/scale_up.md).

## License

Code in this repository is released under [Apache-2.0](LICENSE). The released
dataset assets are covered separately by [LICENSE-DATA.md](LICENSE-DATA.md).
Third-party components keep their original licenses.

## Acknowledgements

This project builds on and thanks the following open-source projects:
[Animal-in-Motion](https://github.com/briannlongzhao/Animal-in-Motion) for its
video collection pipeline design, [SAM3](https://github.com/facebookresearch/sam3),
[VIPE](https://github.com/nv-tlabs/vipe),
[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio), and
[SV4D](https://github.com/Stability-AI/generative-models).
