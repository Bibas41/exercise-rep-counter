"""
rep_counter.py
Core logic of the exercise rep counter (no camera or model code, so it can be
tested with landmark sequences):
    joint angle -> smoothing -> repetition state machine -> counts, partial reps, log

Used by counter.py (video / webcam / landmark files) and evaluate.py (tests).
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

# MediaPipe Pose landmark indices (33 landmarks in total; these are the ones we track)
INDEX = {
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
}
TRACKED = list(INDEX)

# A frame is a dict: landmark name -> (x, y, z, visibility), or None when no person was found.
Frame = dict | None


# ----------------------------------------------------------------- exercises
@dataclass(frozen=True)
class Exercise:
    key: str
    name: str
    signal: str                   # human-readable name of the movement signal
    joints: tuple[str, str, str]  # first point, middle joint (angle), third point
    extended: float  # angle >= this: start/end position of a rep
    flexed: float    # angle <= this: bottom of the rep reached
    attempt: float   # angle <  this: a rep was attempted (used to detect partial reps)
    min_rep_s: float = 0.5  # completed reps faster than this are rejected as glitches


EXERCISES = {
    "curl": Exercise("curl", "Bicep curl", "elbow angle (shoulder-elbow-wrist)",
                     ("shoulder", "elbow", "wrist"), extended=150, flexed=60, attempt=120),
    "squat": Exercise("squat", "Squat", "knee angle (hip-knee-ankle)",
                      ("hip", "knee", "ankle"), extended=160, flexed=100, attempt=140, min_rep_s=0.6),
    "pushup": Exercise("pushup", "Push-up", "elbow angle (shoulder-elbow-wrist)",
                       ("shoulder", "elbow", "wrist"), extended=150, flexed=90, attempt=130),
}


def get_exercise(key: str, extended: float | None = None, flexed: float | None = None) -> Exercise:
    """Return an exercise definition, optionally with custom thresholds."""
    if key not in EXERCISES:
        raise ValueError(f"Unknown exercise '{key}'. Choose one of: {', '.join(EXERCISES)}.")
    ex = EXERCISES[key]
    if extended is not None:
        ex = replace(ex, extended=extended)
    if flexed is not None:
        ex = replace(ex, flexed=flexed)
    if ex.flexed >= ex.extended:
        raise ValueError("The flexed threshold must be smaller than the extended threshold.")
    if not ex.flexed < ex.attempt < ex.extended:
        ex = replace(ex, attempt=(ex.flexed + ex.extended) / 2)
    return ex


# ----------------------------------------------------------------- angles
def joint_angle(a, b, c) -> float | None:
    """Angle at point b (degrees, 0-180) between the vectors b->a and b->c."""
    ba = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    bc = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    norm = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norm == 0:
        return None
    cos = np.clip(np.dot(ba, bc) / norm, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def side_visibility(frame: Frame, exercise: Exercise, side: str) -> float:
    """Lowest visibility of the three landmarks on one side (0 if any is missing)."""
    if not frame:
        return 0.0
    vis = []
    for joint in exercise.joints:
        lm = frame.get(f"{side}_{joint}")
        if lm is None:
            return 0.0
        vis.append(lm[3])
    return float(min(vis))


def get_angle(frame: Frame, exercise: Exercise, side: str,
              min_visibility: float = 0.3, use_3d: bool = False) -> tuple[float | None, float]:
    """Joint angle for one side, or None if the landmarks are missing or uncertain."""
    visibility = side_visibility(frame, exercise, side)
    if visibility < min_visibility:
        return None, visibility
    dims = 3 if use_3d else 2
    points = [frame[f"{side}_{j}"][:dims] for j in exercise.joints]
    return joint_angle(*points), visibility


class SideSelector:
    """'auto' mode: track the better visible body side, switching only when clearly better."""

    def __init__(self, margin: float = 0.15):
        self.margin = margin
        self.current: str | None = None

    def choose(self, frame: Frame, exercise: Exercise) -> str:
        vis = {s: side_visibility(frame, exercise, s) for s in ("left", "right")}
        best = max(vis, key=vis.get)
        if self.current is None:
            self.current = best
        elif best != self.current and vis[best] > vis[self.current] + self.margin:
            self.current = best
        return self.current


# ----------------------------------------------------------------- smoothing
class EMASmoother:
    """Exponential moving average. alpha=1 means no smoothing."""

    def __init__(self, alpha: float = 0.4):
        if not 0 < alpha <= 1:
            raise ValueError("Smoothing alpha must be between 0 (exclusive) and 1.")
        self.alpha = alpha
        self.value: float | None = None

    def update(self, x: float) -> float:
        self.value = x if self.value is None else self.alpha * x + (1 - self.alpha) * self.value
        return self.value

    def reset(self) -> None:
        self.value = None


# ----------------------------------------------------------------- counting
@dataclass
class Rep:
    number: int
    start: float
    end: float
    min_angle: float


class RepCounter:
    """
    State machine for one body side.

      waiting  --angle >= extended-->  extended
      extended --angle <= flexed---->  flexed
      flexed   --angle >= extended-->  extended   (+1 rep, if it took >= min_rep_s)

    Two thresholds (hysteresis) mean noise around one value cannot create extra reps.
    If the angle goes below 'attempt' but comes back up without reaching 'flexed',
    the movement is recorded as a partial rep and not counted.
    """

    def __init__(self, exercise: Exercise, smoothing: float = 0.4, max_gap_s: float = 1.0):
        self.ex = exercise
        self.smoother = EMASmoother(smoothing)
        self.max_gap_s = max_gap_s
        self.state = "waiting"
        self.count = 0
        self.partials = 0
        self.rejected = 0
        self.reps: list[Rep] = []
        self.events: list[tuple[float, str]] = []  # (time, "rep" / "partial" / "rejected")
        self.frames = 0
        self.missing = 0
        self._last_valid_t: float | None = None
        self._rep_start: float | None = None
        self._attempting = False
        self._min_angle = math.inf
        self._last_raw: float | None = None
        self._last_t: float | None = None

    def _reset_attempt(self, t: float) -> None:
        self._attempting = False
        self._min_angle = math.inf
        self._rep_start = t

    def update(self, t: float, angle: float | None) -> float | None:
        """Feed one frame. Returns the smoothed angle (None for a missing frame)."""
        self.frames += 1
        if angle is None:
            self.missing += 1
            return None
        if self._last_valid_t is not None and t - self._last_valid_t > self.max_gap_s:
            self.smoother.reset()  # long gap: don't blend old and new values
        self._last_valid_t = t
        self._last_raw, self._last_t = angle, t
        a = self.smoother.update(angle)
        ex = self.ex

        if self.state == "waiting":
            if a >= ex.extended:
                self.state = "extended"
                self._reset_attempt(t)
        elif self.state == "extended":
            if a >= ex.extended:
                if self._attempting:  # went part of the way down and came back
                    self.partials += 1
                    self.events.append((t, "partial"))
                self._reset_attempt(t)
            else:
                self._min_angle = min(self._min_angle, a)
                if a < ex.attempt:
                    self._attempting = True
                if a <= ex.flexed:
                    self.state = "flexed"
        elif self.state == "flexed":
            self._min_angle = min(self._min_angle, a)
            if a >= ex.extended:
                duration = t - (self._rep_start if self._rep_start is not None else t)
                if duration >= ex.min_rep_s:
                    self.count += 1
                    self.reps.append(Rep(self.count, self._rep_start, t, self._min_angle))
                    self.events.append((t, "rep"))
                else:
                    self.rejected += 1
                    self.events.append((t, "rejected"))
                self.state = "extended"
                self._reset_attempt(t)
        return a

    def finish(self) -> None:
        """End of the stream: if the last raw angle was already back at the start position but the
        smoothed value had not caught up yet (smoothing lag), complete that last rep."""
        if (self.state == "flexed" and self._last_raw is not None
                and self._last_raw >= self.ex.extended and self._rep_start is not None
                and self._last_t - self._rep_start >= self.ex.min_rep_s):
            self.count += 1
            self.reps.append(Rep(self.count, self._rep_start, self._last_t, self._min_angle))
            self.events.append((self._last_t, "rep"))
            self.state = "extended"


class NaiveCounter:
    """Baseline for comparison: +1 every time the raw angle drops below one threshold."""

    def __init__(self, exercise: Exercise):
        self.threshold = (exercise.extended + exercise.flexed) / 2
        self.count = 0
        self._above = None

    def update(self, angle: float | None) -> None:
        if angle is None:
            return
        above = angle >= self.threshold
        if self._above is True and not above:
            self.count += 1
        self._above = above


# ----------------------------------------------------------------- session
class Session:
    """Runs one or two counters (side = left / right / auto / both) over a stream of frames."""

    def __init__(self, exercise: Exercise, side: str = "auto", smoothing: float = 0.4,
                 min_visibility: float = 0.3, use_3d: bool = False, max_gap_s: float = 1.0):
        if side not in ("left", "right", "auto", "both"):
            raise ValueError("side must be left, right, auto or both.")
        self.ex = exercise
        self.side = side
        self.min_visibility = min_visibility
        self.use_3d = use_3d
        keys = ["left", "right"] if side == "both" else [side]
        self.counters = {k: RepCounter(exercise, smoothing, max_gap_s) for k in keys}
        self.naive = {k: NaiveCounter(exercise) for k in keys}
        self.selector = SideSelector() if side == "auto" else None
        self.log: list[dict] = []
        self.duration = 0.0

    def update(self, t: float, frame: Frame) -> dict:
        self.duration = max(self.duration, t)
        out = {}
        for key, counter in self.counters.items():
            body_side = self.selector.choose(frame, self.ex) if key == "auto" else key
            raw, vis = get_angle(frame, self.ex, body_side, self.min_visibility, self.use_3d)
            smooth = counter.update(t, raw)
            self.naive[key].update(raw)
            row = {"time": round(t, 4), "counter": key, "side": body_side,
                   "raw_angle": None if raw is None else round(raw, 2),
                   "smooth_angle": None if smooth is None else round(smooth, 2),
                   "visibility": round(vis, 3), "state": counter.state,
                   "count": counter.count, "partials": counter.partials}
            self.log.append(row)
            out[key] = row
        return out

    def finish(self) -> None:
        for counter in self.counters.values():
            counter.finish()

    @property
    def count(self) -> int:
        return sum(c.count for c in self.counters.values())

    def summary(self) -> dict:
        frames = sum(c.frames for c in self.counters.values())
        missing = sum(c.missing for c in self.counters.values())
        return {
            "exercise": self.ex.name,
            "signal": self.ex.signal,
            "thresholds": {"extended": self.ex.extended, "flexed": self.ex.flexed,
                           "partial_attempt": self.ex.attempt, "min_rep_seconds": self.ex.min_rep_s},
            "side": self.side,
            "counts": {k: c.count for k, c in self.counters.items()},
            "total_reps": self.count,
            "partial_reps": sum(c.partials for c in self.counters.values()),
            "rejected_too_fast": sum(c.rejected for c in self.counters.values()),
            "naive_baseline_reps": sum(n.count for n in self.naive.values()),
            "frames": frames // max(len(self.counters), 1),
            "missing_or_low_confidence_pct": round(100 * missing / frames, 1) if frames else 0.0,
            "duration_s": round(self.duration, 2),
            "reps": [{"side": k, "number": r.number, "start_s": round(r.start, 2),
                      "end_s": round(r.end, 2), "min_angle": round(r.min_angle, 1)}
                     for k, c in self.counters.items() for r in c.reps],
        }


# ----------------------------------------------------------------- files
def save_landmarks_csv(path: Path, rows: list[tuple[float, Frame]]) -> None:
    """Save a landmark sequence: time + x, y, z, visibility for each tracked landmark."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["time"] + [f"{n}_{k}" for n in TRACKED for k in ("x", "y", "z", "v")]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for t, frame in rows:
            values = [round(t, 4)]
            for name in TRACKED:
                lm = frame.get(name) if frame else None
                values += ["", "", "", ""] if lm is None else [round(v, 4) for v in lm]
            w.writerow(values)


def load_landmarks_csv(path: Path) -> list[tuple[float, Frame]]:
    """Load a landmark sequence saved by save_landmarks_csv (or made by make_synthetic.py)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Landmark file '{path}' was not found.")
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "time" not in reader.fieldnames:
            raise ValueError(f"'{path}' is not a landmark file (missing 'time' column).")
        for line_no, rec in enumerate(reader, start=2):
            try:
                t = float(rec["time"])
            except (TypeError, ValueError):
                raise ValueError(f"{path}: invalid time value on line {line_no}.")
            frame = {}
            for name in TRACKED:
                vals = [rec.get(f"{name}_{k}", "") for k in ("x", "y", "z", "v")]
                if all(v not in ("", None) for v in vals):
                    frame[name] = tuple(float(v) for v in vals)
            rows.append((t, frame or None))
    if not rows:
        raise ValueError(f"'{path}' contains no frames.")
    return rows


def save_signal_csv(path: Path, log: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not log:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0]))
        w.writeheader()
        w.writerows(log)


def plot_signal(session: Session, path: Path, title: str = "") -> None:
    """Time series of the raw and smoothed angle, thresholds and counted reps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ex = session.ex
    fig, axes = plt.subplots(len(session.counters), 1, figsize=(11, 3.6 * len(session.counters)),
                             squeeze=False)
    for ax, (key, counter) in zip(axes[:, 0], session.counters.items()):
        rows = [r for r in session.log if r["counter"] == key]
        t = [r["time"] for r in rows]
        raw = [np.nan if r["raw_angle"] is None else r["raw_angle"] for r in rows]
        smooth = [np.nan if r["smooth_angle"] is None else r["smooth_angle"] for r in rows]
        missing = [r["time"] for r in rows if r["raw_angle"] is None]
        ax.plot(t, raw, color="#9aa5b1", lw=1, label="raw angle")
        ax.plot(t, smooth, color="#1f6feb", lw=2, label="smoothed angle")
        ax.axhline(ex.extended, color="#2da44e", ls="--", lw=1, label=f"extended ({ex.extended:g} deg)")
        ax.axhline(ex.flexed, color="#cf222e", ls="--", lw=1, label=f"flexed ({ex.flexed:g} deg)")
        ax.axhline(ex.attempt, color="#bf8700", ls=":", lw=1, label=f"partial attempt ({ex.attempt:g} deg)")
        if missing:
            ax.plot(missing, [5] * len(missing), "|", color="#6e7781", ms=8, label="missing / low confidence")
        for ev_t, kind in counter.events:
            color = {"rep": "#2da44e", "partial": "#bf8700", "rejected": "#cf222e"}[kind]
            ax.axvline(ev_t, color=color, alpha=0.5, lw=1.5)
        ax.set_ylim(0, 185)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("angle (deg)")
        ax.set_title(f"{title or ex.name} - {key}: {counter.count} reps, {counter.partials} partial"
                     + (f", {counter.rejected} rejected" if counter.rejected else ""))
        ax.legend(loc="lower right", fontsize=7, ncol=3)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
