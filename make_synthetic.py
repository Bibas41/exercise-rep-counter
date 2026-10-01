"""
make_synthetic.py - creates synthetic landmark sequences with a KNOWN number of reps.

They test the counting logic under controlled conditions (clean, fast, slow, pauses,
partial reps, noise, missing landmarks, tracking glitches, no person) without any
personal video data. Output: data/synthetic/*.csv (same format as counter.py saves).
Run:  python make_synthetic.py
"""
import math
import random
from pathlib import Path

from rep_counter import save_landmarks_csv

FPS = 30
OUT = Path("data/synthetic")


def rep(rest, bottom, duration, pause_top=0.3, pause_bottom=0.0):
    """Angles for one rep: rest -> bottom -> rest (cosine shape) with optional pauses."""
    angles = [rest] * int(pause_top * FPS)
    half = int(duration / 2 * FPS)
    angles += [rest - (rest - bottom) * (1 - math.cos(math.pi * i / half)) / 2 for i in range(half)]
    angles += [bottom] * int(pause_bottom * FPS)
    angles += [bottom + (rest - bottom) * (1 - math.cos(math.pi * i / half)) / 2 for i in range(half)]
    return angles


def arm_frame(angle, noise=0.0):
    """Left arm seen from the side: upper arm vertical, forearm rotates around the elbow."""
    sh, el = (300.0, 200.0), (300.0, 350.0)
    th = math.radians(angle)
    wr = (el[0] + 140 * math.sin(th), el[1] - 140 * math.cos(th))
    pts = {"left_shoulder": sh, "left_elbow": el, "left_wrist": wr,
           "left_hip": (300.0, 520.0), "left_knee": (300.0, 700.0), "left_ankle": (300.0, 880.0)}
    return {k: (x + random.gauss(0, noise), y + random.gauss(0, noise), 0.0, 0.95)
            for k, (x, y) in pts.items()}


def leg_frame(angle, noise=0.0):
    """Left leg seen from the side: shin fixed, thigh rotates around the knee."""
    an, kn = (320.0, 880.0), (360.0, 700.0)
    shin = math.atan2(an[1] - kn[1], an[0] - kn[0])
    d = shin - math.radians(angle)
    hip = (kn[0] + 180 * math.cos(d), kn[1] + 180 * math.sin(d))
    sh = (hip[0], hip[1] - 250)
    pts = {"left_shoulder": sh, "left_elbow": (sh[0] + 20, sh[1] + 140), "left_wrist": (sh[0] + 40, sh[1] + 270),
           "left_hip": hip, "left_knee": kn, "left_ankle": an}
    return {k: (x + random.gauss(0, noise), y + random.gauss(0, noise), 0.0, 0.95)
            for k, (x, y) in pts.items()}


def build(name, angles, maker, noise=0.0, drop=0.0, gaps=(), glitches=()):
    rows = []
    for i, a in enumerate(angles):
        t = i / FPS
        frame = maker(a, noise)
        if random.random() < drop or any(s <= t < e for s, e in gaps):
            frame = None
        elif any(s <= t < e for s, e in glitches):  # tracker jumps: wrist lands on the shoulder
            frame["left_wrist"] = frame["left_shoulder"][:2] + (0.0, 0.9)
        rows.append((t, frame))
    save_landmarks_csv(OUT / f"{name}.csv", rows)
    print(f"Created {OUT / name}.csv ({len(rows)} frames)")


def main():
    random.seed(42)
    full = lambda d=2.0, **k: rep(165, 40, d, **k)
    build("curl_clean", sum((full() for _ in range(10)), []), arm_frame)
    build("curl_fast", sum((full(0.8, pause_top=0.05) for _ in range(10)), []), arm_frame)
    build("curl_slow_pauses", sum((full(4.0, pause_top=2.0, pause_bottom=1.0) for _ in range(6)), []), arm_frame)
    partial = lambda b: rep(165, b, 1.6)
    build("curl_partial", full() + full() + partial(100) + full() + partial(95) + full() + partial(110) + full(),
          arm_frame)
    build("curl_noisy", sum((full() for _ in range(8)), []), arm_frame, noise=6.0)
    build("curl_missing", sum((full() for _ in range(8)), []), arm_frame, drop=0.15, gaps=[(7.5, 8.7)])
    build("curl_glitches", sum((full() for _ in range(6)), []) + [165] * 60, arm_frame,
          glitches=[(0.9, 1.05), (5.2, 5.35), (13.0, 13.15)])
    build("squat_clean", sum((rep(172, 80, 2.5) for _ in range(8)), []), leg_frame)
    shallow = rep(172, 125, 2.0)
    build("squat_shallow", sum((rep(172, 80, 2.5) + shallow for _ in range(3)), []), leg_frame)
    build("pushup_clean", sum((rep(168, 75, 1.8) for _ in range(6)), []), arm_frame)
    build("no_person", [165] * (4 * FPS), arm_frame, drop=1.0)


if __name__ == "__main__":
    main()
