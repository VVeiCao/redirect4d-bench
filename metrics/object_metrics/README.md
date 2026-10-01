# Object Metrics

Object evaluation compares a method-generated video against the released
target pseudo-GT mask.

The public evaluation wrapper runs:

1. SAM3 propagation to convert each generated RGB video into a binary mask.
2. Detection: whether both pseudo-GT and prediction contain the object.
3. Localization: mask IoU and box IoU when both masks are present.
4. Optional recognition aggregation if a VLM recognition cache is provided.

Most users should call:

```bash
python scripts/evaluation/evaluate_user_method.py \
  --video-dir my_method/VIDEOS
```

The extracted generated-video masks are written to:

```text
my_method/mask/<track>_<trajectory>.mp4
```
