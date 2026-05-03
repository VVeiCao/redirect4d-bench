"""Qwen-VL as judge for "is the masked content a structurally plausible
instance of <class>".

Feeds masked RGBA crop (transparent outside mask, Alpha-CLIP style) to VLM with
a prompt that asks for a 0-10 plausibility score + a brief reason, independent
of camera viewpoint.

Output per (case, method):
    scores_<case>.json   list of {method, frame, score, label, reason, crop_path}
    per_method_mean.csv  aggregate
    grid_<case>.png      same layout as build_probe_grid.py, VLM score printed
"""
import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CASES = [
    # (track, traj)
    ("camel_IJ4YajWrDcA_027_001_seq1",        "yaw_-120_pitch_-10_roll_0_scale_1p1"),
    ("cat_l-Tzteg9ksM_007_001_seq1",          "yaw_100_pitch_-30_roll_0_scale_1p8"),
    ("bear_NnAlfavy2us_003_001_seq1",         "yaw_-110_pitch_0_roll_0_scale_1"),
    ("dancer_0tFft6QkuhM_016_001_seq2",       "yaw_110_pitch_-10_roll_0_scale_1"),
    ("elephant_4F0hzklQejU_010_001_seq1",     "yaw_-120_pitch_-20_roll_0_scale_1"),
    ("tiger_MIBAT6BGE6U_002_001_seq1",        "yaw_-120_pitch_-10_roll_0_scale_1"),
    ("zebra_qvRTslcIeSk_002_001_seq1",        "yaw_120_pitch_0_roll_0_scale_1"),
]
REPO_ROOT = Path(__file__).resolve().parents[3]
METHODS: list[str] = []
PREDICTION_VIDEOS = REPO_ROOT / "predictions"
MASK_ROOT = REPO_ROOT / "outputs" / "object_metric_masks" / "seeded_from_pt_box"


# ---------------------- video / mask / crop helpers ----------------------

def read_video(path):
    cap = cv2.VideoCapture(str(path))
    fs = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fs.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return fs

def read_mask(path):
    cap = cv2.VideoCapture(str(path))
    ms = []
    while True:
        ok, f = cap.read()
        if not ok: break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        ms.append((g > 127).astype(np.uint8))
    cap.release()
    return ms

def bbox_of(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0: return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

def build_crop(rgb, mask, mode="bbox", pad=0.10):
    """Return HxWx3 uint8 RGB — exactly what the VLM will see.

    mode:
      "bbox"         = full bbox with natural background (mask only used for localisation)
      "masked_black" = bbox, but all pixels outside mask set to 0,0,0
      "masked_gray"  = bbox, but all pixels outside mask set to 128,128,128
      "masked_white" = bbox, but all pixels outside mask set to 255,255,255
                       (best for dark objects; won't blend like masked_black)
    """
    bb = bbox_of(mask)
    if bb is None: return None
    H, W = rgb.shape[:2]
    x0, y0, x1, y1 = bb
    bw, bh = x1 - x0, y1 - y0
    px, py = int(bw * pad), int(bh * pad)
    x0 = max(0, x0 - px); y0 = max(0, y0 - py)
    x1 = min(W, x1 + px); y1 = min(H, y1 + py)
    rgb_crop = rgb[y0:y1, x0:x1].copy()
    m = mask[y0:y1, x0:x1]
    if m.shape != rgb_crop.shape[:2]:
        m = cv2.resize(m, (rgb_crop.shape[1], rgb_crop.shape[0]), interpolation=cv2.INTER_NEAREST)
    if mode == "bbox":
        return rgb_crop
    if mode == "masked_black":
        rgb_crop[m == 0] = (0, 0, 0)
        return rgb_crop
    if mode == "masked_gray":
        rgb_crop[m == 0] = (128, 128, 128)
        return rgb_crop
    if mode == "masked_white":
        rgb_crop[m == 0] = (255, 255, 255)
        return rgb_crop
    raise ValueError(f"unknown mode {mode!r}")


# ---------------------- prompt + parser ----------------------

PROMPT_TEMPLATE = """You are a QA reviewer inspecting a single {cls} in an image. The image is a crop from an AI-generated video; it may or may not have anatomical defects.

DEFECT categories (flag "yes" if ANY of these is clearly present):
- Extra tail (two distinct tails)
- Extra leg (clearly 5+ legs; count between legs and under the belly)
- Missing leg that SHOULD be visible from this viewpoint
- Limb bent IMPOSSIBLY: knee/hock bent BACKWARDS (opposite to the natural anatomy), or joint rotated so the foot points wrong, or an S-shape with multiple opposing bends
- Fused / melted limbs (two legs merged into one blob with no separation)
- Multiple distinct heads or duplicated face
- A limb attached at an anatomically impossible location on the body
- Grossly wrong proportions a real {cls} could not have

NOT defects (do not flag these):
- A leg bent at the knee/hock/elbow in a natural walking, running, trotting, charging, jumping, or landing pose. Motion always bends limbs; that is normal.
- Parts cropped out or occluded by camera angle.
- Motion blur.
- Species-specific anatomy: the elephant's trunk is NOT an extra tail; the camel's hump is NOT a deformity; any species' normal features are fine.

Reason step by step, then answer:
SCAN: <one short sentence: what do you see — body, limbs count, tail count, head, pose>
DEFECT: yes | no
WHAT: <if yes, 3-8 words naming the defect; if no, leave blank>"""


def build_fewshot(prompt_text, fewshot_dir: Path):
    """Return list of (pil_image, role, text) triples that form the few-shot chat."""
    shots = [
        # A: clean cat (static)
        (fewshot_dir / "A_ok_cat_static.png",
         "SCAN: A black cat from behind, 4 legs visible, 1 tail, standing on ground.\n"
         "DEFECT: no\nWHAT:"),
        # B: walking elephant (motion + species-specific trunk)
        (fewshot_dir / "B_ok_elephant_motion.png",
         "SCAN: An elephant walking, 4 legs with one lifted in stride, trunk from head, 1 tail.\n"
         "DEFECT: no\nWHAT:"),
        # C: cat with extra tail (subtle duplication)
        (fewshot_dir / "C_defect_cat_extratail.png",
         "SCAN: A black cat; two long tail-like appendages are visible, both of similar length and color.\n"
         "DEFECT: yes\nWHAT: two tails"),
        # D: camel with fused/missing body structure
        (fewshot_dir / "D_defect_camel_missing.png",
         "SCAN: A camel body with a thin stretched head; multiple leg-like stubs fused under the belly.\n"
         "DEFECT: yes\nWHAT: fused limbs, deformed body"),
    ]
    out = []
    for img_path, answer in shots:
        if not img_path.exists(): continue
        pil = Image.open(img_path).convert("RGB")
        out.append((pil, answer))
    return out


def parse_verdict(text):
    """Parse binary verdict. Returns (defect_flag in {True, False, None}, what_str)."""
    t = text.strip()
    m = re.search(r"DEFECT\s*:\s*(yes|no)\b", t, re.I)
    what = re.search(r"WHAT\s*:\s*(.*)", t, re.I)
    defect = None
    if m:
        defect = (m.group(1).lower() == "yes")
    what_s = (what.group(1).strip() if what else "")[:80]
    return defect, what_s


# ---------------------- class name from track ----------------------

def class_from_track(track):
    head = track.split("_")[0].lower()
    return {
        "camel": "camel", "cat": "cat", "dog": "dog", "horse": "horse",
        "bear": "bear", "elephant": "elephant", "tiger": "tiger",
        "zebra": "zebra", "cow": "cow", "deer": "deer", "pig": "pig",
        "wolf": "wolf", "gear": "gear",
        "dancer": "human dancer", "person": "person",
        "robot": "humanoid robot",
    }.get(head, head)


# ---------------------- VLM ----------------------

def load_vlm(model_id="Qwen/Qwen3-VL-32B-Instruct", device="cuda", load_in_4bit=False):
    import torch
    from transformers import AutoProcessor
    print(f"[vlm] loading {model_id}  4bit={load_in_4bit}...", flush=True)
    try:
        from transformers import AutoModelForImageTextToText
        ModelCls = AutoModelForImageTextToText
    except ImportError:
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelCls
    device_map = "auto" if device == "auto" else device
    kwargs = {"device_map": device_map}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["dtype"] = torch.bfloat16
    model = ModelCls.from_pretrained(model_id, **kwargs)
    model.eval()
    proc = AutoProcessor.from_pretrained(model_id)
    return model, proc


def ask(model, proc, pil_img: Image.Image, prompt: str, max_new=96, fewshot=None):
    """Supports optional few-shot (list of (pil, answer_str))."""
    import torch
    msgs = []
    imgs = []
    if fewshot:
        for ex_pil, ex_ans in fewshot:
            msgs.append({"role": "user",
                         "content": [{"type": "image", "image": ex_pil},
                                     {"type": "text", "text": prompt}]})
            msgs.append({"role": "assistant",
                         "content": [{"type": "text", "text": ex_ans}]})
            imgs.append(ex_pil)
    msgs.append({"role": "user",
                 "content": [{"type": "image", "image": pil_img},
                             {"type": "text", "text": prompt}]})
    imgs.append(pil_img)
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=imgs, padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    new = out[:, inputs.input_ids.shape[1]:]
    return proc.batch_decode(new, skip_special_tokens=True)[0]


# ---------------------- main ----------------------

def run_case(track, traj, n_frames, model, proc, save_dir, cls_override=None, mode="bbox", fewshot=None):
    key = f"{track}__{traj}"
    cls = cls_override or class_from_track(track)
    prompt = PROMPT_TEMPLATE.format(cls=cls)
    print(f"\n=== {key}  class={cls!r}  mode={mode} ===")

    out = {"case": key, "class": cls, "mode": mode, "methods": {}}
    crops_dir = save_dir / "_crops" / key
    crops_dir.mkdir(parents=True, exist_ok=True)

    for m in METHODS:
        vid_p = PREDICTION_VIDEOS / m / "videos" / f"{key}.mp4"
        msk_p = MASK_ROOT / m / f"{key}.mp4"
        if not vid_p.exists() or not msk_p.exists():
            print(f"  [skip] {m}: missing files")
            continue
        frames = read_video(vid_p)
        masks = read_mask(msk_p)
        n = min(len(frames), len(masks))
        if n < 2:
            continue
        idxs = np.linspace(0, n - 1, n_frames).astype(int).tolist()
        results = []
        for i in idxs:
            f, mk = frames[i], masks[i]
            if f.shape[:2] != mk.shape:
                mk = cv2.resize(mk, (f.shape[1], f.shape[0]), interpolation=cv2.INTER_NEAREST)
            rgb_crop = build_crop(f, mk, mode=mode)
            if rgb_crop is None or rgb_crop.size == 0:
                continue
            pil = Image.fromarray(rgb_crop, mode="RGB")  # what the VLM sees = what we save
            crop_path = crops_dir / f"{m}_f{i:03d}.png"
            pil.save(crop_path)
            try:
                resp = ask(model, proc, pil, prompt, max_new=96, fewshot=fewshot)
                defect, what = parse_verdict(resp)
            except Exception as e:
                resp = f"[ERR] {e}"
                defect, what = None, str(e)[:80]
            results.append({"frame": int(i), "defect": defect, "what": what,
                            "raw": resp, "crop": str(crop_path)})
            verdict_s = "DEFECT" if defect is True else ("OK    " if defect is False else "NA    ")
            print(f"  {m:>16} f{i:03d}  {verdict_s}  {what[:60]}")
        if results:
            defined = [r for r in results if r["defect"] is not None]
            n = len(defined)
            n_ok = sum(1 for r in defined if r["defect"] is False)
            n_def = sum(1 for r in defined if r["defect"] is True)
            plausible_rate = (n_ok / n) if n else None
            out["methods"][m] = {
                "frames": results,
                "n_sampled": len(results),
                "n_parsed": n,
                "n_ok": n_ok,
                "n_defect": n_def,
                "plausible_rate": plausible_rate,
            }

    # save json
    save_path = save_dir / f"scores_{key}.json"
    save_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {save_path}")
    return out


def get_font(size):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if Path(p).exists(): return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def fit_onto(img_bgr, size, pad=(28, 28, 32)):
    h, w = img_bgr.shape[:2]
    s = min(size / w, size / h)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    r = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    c = np.full((size, size, 3), pad, np.uint8)
    y0 = (size - nh) // 2; x0 = (size - nw) // 2
    c[y0:y0 + nh, x0:x0 + nw] = r
    return c


def build_grid(case_result, out_path: Path, cell=240):
    """Render a 6 x N grid: rows = methods, cols = frames. Each cell = crop + score + reason."""
    key = case_result["case"]
    mres = case_result["methods"]
    if not mres: return
    frames_present = sorted({r["frame"] for mr in mres.values() for r in mr["frames"]})
    cols = len(frames_present)
    methods = list(mres.keys())
    rows = len(methods)

    label_w = 210
    header_h = 44
    score_h = 60
    gap = 4
    W = label_w + cols * (cell + gap) + gap
    H = header_h + rows * (cell + score_h + gap) + 20

    img = Image.new("RGB", (W, H), (20, 20, 22))
    dr = ImageDraw.Draw(img)
    f_hdr = get_font(15)
    f_sc  = get_font(11)
    f_title = get_font(16)
    model_name = case_result.get("model", "Qwen-VL")
    dr.text((10, 6),  f"VLM binary verdict ({model_name}) · case: {key}  class: {case_result['class']}", fill=(220, 220, 230), font=f_title)
    dr.text((10, 26), "cell: crop + OK (green) / DEFECT (red) + <=8-word reason.  Per-method: plausible_rate = #OK / #sampled.",
            fill=(150, 155, 165), font=f_sc)

    for j, fi in enumerate(frames_present):
        x = label_w + j * (cell + gap) + gap
        dr.text((x + cell // 2 - 24, header_h - 18), f"frame {fi}", fill=(180, 185, 195), font=f_hdr)

    for i, m in enumerate(methods):
        y = header_h + i * (cell + score_h + gap)
        dr.text((8, y + cell // 2 - 18), m, fill=(230, 230, 240), font=f_hdr)
        mr = mres.get(m)
        if not mr:
            dr.text((8, y + cell // 2 + 2), "(missing)", fill=(200, 100, 100), font=f_sc)
            continue
        rate = mr["plausible_rate"]
        rate_s = f"{rate*100:.0f}%" if rate is not None else "NA"
        rate_col = (134, 200, 155) if rate is not None and rate >= 0.9 else \
                   (244, 105, 94) if rate is not None and rate < 0.5 else (210, 215, 225)
        dr.text((8, y + cell // 2 + 0), f"plausible: {rate_s}", fill=rate_col, font=f_sc)
        dr.text((8, y + cell // 2 + 14), f"{mr['n_ok']}/{mr['n_sampled']} OK",
                fill=(160, 165, 175), font=f_sc)
        dr.text((8, y + cell // 2 + 28), f"{mr['n_defect']} defect",
                fill=(200, 150, 120) if mr['n_defect'] > 0 else (160, 165, 175), font=f_sc)

        by_frame = {r["frame"]: r for r in mr["frames"]}
        for j, fi in enumerate(frames_present):
            x = label_w + j * (cell + gap) + gap
            r = by_frame.get(fi)
            if r is None: continue
            # paste crop
            rgba = cv2.imread(r["crop"], cv2.IMREAD_UNCHANGED)
            if rgba is not None:
                if rgba.shape[2] == 4:
                    bgr = cv2.cvtColor(rgba, cv2.COLOR_BGRA2BGR)
                else:
                    bgr = rgba
                cell_img = fit_onto(bgr, cell)
                pil_c = Image.fromarray(cv2.cvtColor(cell_img, cv2.COLOR_BGR2RGB))
                img.paste(pil_c, (x, y))
            d = r.get("defect")
            if d is True:
                verdict, vc = "DEFECT", (244, 105, 94)
            elif d is False:
                verdict, vc = "OK", (134, 200, 155)
            else:
                verdict, vc = "NA", (200, 200, 200)
            dr.text((x + 4, y + cell + 2), verdict, fill=vc, font=f_sc)
            # wrap the "what"
            what = (r.get("what") or "")[:80]
            def wrap(s, n):
                words = s.split()
                lines = [""]
                for w in words:
                    if len(lines[-1]) + len(w) + 1 <= n:
                        lines[-1] = (lines[-1] + " " + w).strip()
                    else:
                        lines.append(w)
                return lines[:3]
            wc = (244, 170, 100) if d is True else (150, 155, 165)
            for k, line in enumerate(wrap(what, 30)[:3]):
                dr.text((x + 4, y + cell + 18 + k * 14), line, fill=wc, font=f_sc)

    img.save(out_path)
    print(f"[grid] {out_path}  ({W}x{H})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_frames", type=int, default=8)
    ap.add_argument("--model_id", default="Qwen/Qwen3-VL-32B-Instruct")
    ap.add_argument("--modes", nargs="+",
                    default=["bbox"],
                    choices=["bbox", "masked_black", "masked_gray", "masked_white"])
    ap.add_argument("--no_fewshot", action="store_true",
                    help="zero-shot: only send the prompt, no example images")
    ap.add_argument("--out_subroot", default=None,
                    help="save under _vlm/<subroot>/<mode>/; default: _vlm/<mode>/")
    args = ap.parse_args()
    root = Path(__file__).parent / "_vlm"
    root.mkdir(exist_ok=True)

    model, proc = load_vlm(args.model_id)

    if args.no_fewshot:
        fewshot = None
        print("[fewshot] ZERO-SHOT — no examples, prompt only")
    else:
        fewshot_dir = Path(__file__).parent / "fewshot"
        prompt_text = PROMPT_TEMPLATE.format(cls="animal")
        fewshot = build_fewshot(prompt_text, fewshot_dir)
        print(f"[fewshot] loaded {len(fewshot)} examples")

    if args.out_subroot:
        root = root / args.out_subroot
        root.mkdir(parents=True, exist_ok=True)

    # results[mode][case] = case_result
    all_results = {mode: [] for mode in args.modes}
    for mode in args.modes:
        save_dir = root / mode
        save_dir.mkdir(exist_ok=True)
        for track, traj in CASES:
            res = run_case(track, traj, args.n_frames, model, proc, save_dir, mode=mode, fewshot=fewshot)
            grid = save_dir / f"grid_{res['case']}.png"
            build_grid(res, grid)
            all_results[mode].append(res)

    # per-mode summary csv
    import csv
    for mode in args.modes:
        s = root / mode / "summary.csv"
        with open(s, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["case", "method", "n_sampled", "n_parsed", "n_ok", "n_defect", "plausible_rate"])
            for res in all_results[mode]:
                for m, mr in res["methods"].items():
                    pr = mr["plausible_rate"]
                    w.writerow([res["case"], m, mr["n_sampled"], mr["n_parsed"],
                                mr["n_ok"], mr["n_defect"],
                                f"{pr:.3f}" if pr is not None else ""])
        print(f"[summary] {s}")

    # cross-mode comparison table: for each (case, method), show plausible_rate across modes
    combo = root / "compare_modes.csv"
    with open(combo, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "method"] + [f"rate_{mode}" for mode in args.modes])
        # build an index
        idx = {(res["case"], m): {} for mode in args.modes for res in all_results[mode] for m in res["methods"]}
        for mode in args.modes:
            for res in all_results[mode]:
                for m, mr in res["methods"].items():
                    idx[(res["case"], m)][mode] = mr["plausible_rate"]
        for (case, m), rates in idx.items():
            row = [case, m]
            for mode in args.modes:
                v = rates.get(mode)
                row.append(f"{v:.3f}" if v is not None else "")
            w.writerow(row)
    print(f"[compare] {combo}")

    # print pretty table to stdout
    print("\n=== COMPARE MODES (plausible_rate) ===")
    hdr = f"{'case':<55} {'method':<16}  " + "  ".join(f"{m:>14}" for m in args.modes)
    print(hdr); print("-" * len(hdr))
    # sort by case
    by_case = {}
    for (case, m), rates in idx.items():
        by_case.setdefault(case, []).append((m, rates))
    for case in sorted(by_case):
        short = case.split("__")[0][:50]
        for m, rates in sorted(by_case[case]):
            cells = []
            for mode in args.modes:
                v = rates.get(mode)
                cells.append(f"{v*100:>13.1f}%" if v is not None else "            NA")
            print(f"{short:<55} {m:<16}  " + "  ".join(cells))
        print()


if __name__ == "__main__":
    main()
