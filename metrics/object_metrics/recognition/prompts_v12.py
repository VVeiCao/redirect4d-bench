"""Shared prompt definitions + case registry for the V12 judge.

Kept as the common module because the Qwen adapter imports PROMPTS,
CASES, class_label, and parse_answer from here. Older prompt versions
and the GPT-4o runner have been archived under _archive/.
"""
import os
import re
from pathlib import Path


# ============== V12 PROMPTS (class-dispatched) ==============
# Scoped to Redirect4D-Bench categories:
#   BIO  (11): bear, camel, cat, cow, deer, dancer, elephant, pig, tiger, wolf, zebra
#   MECH (2):  gear, robot


PROMPT_V12_BIO = """You are shown a single frame cropped from an AI-generated video. The crop contains a {cls}. The camera angle may be unusual.

Question: is the {cls} catastrophically broken — i.e., so badly deformed that it is clearly not a real {cls}?

Only answer 'yes' if ONE of these is undeniably visible:

  a. DUPLICATION — literally two heads, two tails, two torsos, or two complete instances of a body part that should appear only once.
  b. WRONG LIMB COUNT — a clearly visible extra limb (5+ legs on a quadruped, 3+ arms on a human), or a main limb entirely missing where anatomy demands it.
  c. MELTED / BLOBBED — the {cls}'s body or limbs have dissolved into a continuous amorphous blob with no recognizable structure.
  d. REVERSED JOINT — a knee, elbow, hock, wrist, or ankle bends in the direction opposite to this species' natural anatomy, visible clearly.

If the {cls} looks like a plausible albeit imperfect rendering of a real {cls}, answer 'no'. Imperfect details, blur, odd poses, long or S-curved necks (camel), humps (camel), trunks (elephant), stripes (tiger, zebra), fur / skin texture, dance poses (dancer), unusual camera angles, cropped parts, self-occlusion, and other natural class-specific features are NEVER defects.

When in doubt, always answer 'no'. Flag 'yes' only for defects that would stop someone scrolling past on social media to say "wait, that's wrong."

Respond with exactly one line. Start with "no" if the {cls} is plausible; otherwise start with "yes" followed by a brief description of which body part is catastrophically broken."""


PROMPT_V12_MECH = """You are shown a single frame cropped from an AI-generated video. The crop contains a {cls}. The camera angle may be unusual.

Question: is the {cls} catastrophically broken — i.e., so badly deformed that it is clearly not a real {cls}?

Only answer 'yes' if ONE of these is undeniably visible:

  a. DUPLICATION — literally two of a main component that should appear only once (two gear hubs, two robot torsos / heads, or two complete instances of a structural piece).
  b. MELTED / BLOBBED — the structural parts have dissolved into a continuous amorphous blob with no recognizable mechanical form.
  c. IMPOSSIBLE STRUCTURE — a rigid component fluidly bent like rubber, a load-bearing hinge twisted beyond its physical range, or a main structural piece snapped with two disconnected halves floating apart.
  d. MIS-ATTACHED PART — a main component sprouting from a clearly impossible location (a gear tooth growing from the hub center instead of the rim, a robot limb attached at the head, etc.).

If the {cls} looks like a plausible albeit imperfect rendering of a real {cls}, answer 'no'. Unusual camera angles, cropped parts, 3D-printed or stylized aesthetics, cutaway views, visible internal mechanisms, unusual colors, minor jitter, and class-specific features (gear teeth on the rim, planetary-gear configurations, robot articulated joints, glowing accents, exposed wiring, etc.) are NEVER defects.

When in doubt, always answer 'no'. Flag 'yes' only for defects that would stop someone scrolling past on social media to say "wait, that's wrong."

Respond with exactly one line. Start with "no" if the {cls} is plausible; otherwise start with "yes" followed by a brief description of which part is catastrophically broken."""


_MECH_KEYWORDS = ("gear", "robot", "robotic", "mechanical", "machine")


class _V12Dispatcher:
    """Acts like a str template for .format(cls=...); at call time it
    routes to the BIO or MECH V12 variant based on the class string."""
    def format(self, cls=None, **kwargs):
        c = (cls or kwargs.get("cls") or "").lower()
        is_mech = any(kw in c for kw in _MECH_KEYWORDS)
        tpl = PROMPT_V12_MECH if is_mech else PROMPT_V12_BIO
        return tpl.format(cls=cls)


PROMPT_V12 = _V12Dispatcher()


PROMPTS = {
    "v12": PROMPT_V12,
}


# ============== CASES ==============

_METADATA_PATH = Path(
    os.environ.get(
        "REDIRECT4D_METADATA",
        Path(__file__).resolve().parents[3] / "data" / "redirect4d_bench" / "metadata.json",
    )
)


CASES_TEST = [
    ("camel_IJ4YajWrDcA_027_001_seq1",    "yaw_-120_pitch_-10_roll_0_scale_1p1"),
    ("cat_l-Tzteg9ksM_007_001_seq1",      "yaw_100_pitch_-30_roll_0_scale_1p8"),
    ("bear_NnAlfavy2us_003_001_seq1",     "yaw_-110_pitch_0_roll_0_scale_1"),
    ("dancer_0tFft6QkuhM_016_001_seq2",   "yaw_110_pitch_-10_roll_0_scale_1"),
    ("elephant_4F0hzklQejU_010_001_seq1", "yaw_-120_pitch_-20_roll_0_scale_1"),
    ("tiger_MIBAT6BGE6U_002_001_seq1",    "yaw_-120_pitch_-10_roll_0_scale_1"),
    ("zebra_qvRTslcIeSk_002_001_seq1",    "yaw_120_pitch_0_roll_0_scale_1"),
    ("gear_TzJkD87eQNI_004_001_seq2",     "yaw_-90_pitch_-20_roll_0_scale_1"),
    ("robot_6zPvT0ig1VM_014_001_seq2",    "yaw_-120_pitch_0_roll_0_scale_1"),
]


def load_full_cases():
    """Return all (track_id, trajectory) pairs from benchmark metadata."""
    import json
    data = json.loads(_METADATA_PATH.read_text())
    cases = []
    for track_id, info in sorted(data["tracks"].items()):
        for traj in info["trajectories"]:
            cases.append((track_id, traj))
    return cases


if os.environ.get("USE_FULL_CASES") == "1":
    CASES = load_full_cases()
else:
    CASES = CASES_TEST


# ============== CLASS LABEL ==============


def class_label(track):
    """Return coarse class label."""
    head = track.split("_")[0].lower()
    return {
        "bear": "bear", "camel": "camel", "cat": "cat", "cow": "cow",
        "deer": "deer", "elephant": "elephant", "pig": "pig",
        "tiger": "tiger", "wolf": "wolf", "zebra": "zebra",
        "dancer": "human dancer",
        "gear": "mechanical gear (toothed wheel)",
        "robot": "robot",
    }.get(head, head)


# ============== ANSWER PARSING ==============


def parse_answer(text):
    t = text.strip().strip("'\"").strip()
    first = re.match(r"\s*(yes|no)\b", t, re.I)
    if not first:
        return None, text.strip()[:80]
    is_defect = first.group(1).lower() == "yes"
    rest = t[first.end():].strip()
    rest = re.sub(r"^\s*[—\-–:,\.\s]+\s*", "", rest)
    return is_defect, rest.strip()[:120]
