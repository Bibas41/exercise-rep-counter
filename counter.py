"""
counter.py - count exercise repetitions in a video file, a webcam stream or a landmark sequence.

Examples:
    python counter.py --input videos/curl_normal.mp4 --exercise curl --show
    python counter.py --webcam --exercise squat
    python counter.py --landmarks data/synthetic/curl_clean.csv --exercise curl
Press q or Esc to stop the video window.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from rep_counter import (EXERCISES, INDEX, Session, get_exercise, load_landmarks_csv,
                         plot_signal, save_landmarks_csv, save_signal_csv)

MODEL_URLS = {
    v: f"https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_{v}/float16/latest/pose_landmarker_{v}.task"
    for v in ("lite", "full", "heavy")
}
MODELS_DIR = Path("models")
OUTPUT_DIR = Path("outputs")

# body connections for drawing (MediaPipe landmark indices, face left out)
CONNECTIONS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
               (23, 25), (25, 27), (24, 26), (26, 28), (27, 29), (29, 31), (27, 31), (28, 30),
               (30, 32), (28, 32), (15, 17), (15, 19), (17, 19), (16, 18), (16, 20), (18, 20)]
STATE_TEXT = {"waiting": "Get into the start position", "extended": "EXTENDED", "flexed": "FLEXED"}


# ----------------------------------------------------------------- model
def ensure_model(variant: str = "full") -> Path:
    """Download the MediaPipe Pose Landmarker model once (about 9 MB for 'full')."""
    path = MODELS_DIR / f"pose_landmarker_{variant}.task"
    if path.is_file() and path.stat().st_size > 0:
        return path
    MODELS_DIR.mkdir(exist_ok=True)
    print(f"Downloading the MediaPipe pose model ({variant}) ...")
    try:
        urllib.request.urlretrieve(MODEL_URLS[variant], path)
    except Exception as exc:
        path.unlink(missing_ok=True)
        sys.exit(f"Error: could not download the model ({exc}).\n"
                 f"Download it manually from {MODEL_URLS[variant]} and save it as {path}.")
    return path


class PoseDetector:
    """MediaPipe Pose Landmarker in VIDEO mode (runs locally)."""

    def __init__(self, model_path: Path, min_confidence: float = 0.5):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        self.mp = mp
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_confidence,
            min_pose_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self._last_ts = -1

    def detect(self, bgr, t_ms: float):
        """Returns (frame_2d_pixels, frame_3d_world, all_pixel_points) or (None, None, None)."""
        import cv2
        h, w = bgr.shape[:2]
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                              data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        ts = max(int(t_ms), self._last_ts + 1)  # timestamps must increase
        self._last_ts = ts
        result = self.landmarker.detect_for_video(image, ts)
        if not result.pose_landmarks:
            return None, None, None
        lms = result.pose_landmarks[0]
        vis = [lm.visibility if lm.visibility is not None else 1.0 for lm in lms]
        pixels = [(lm.x * w, lm.y * h, lm.z * w, v) for lm, v in zip(lms, vis)]
        frame_2d = {name: pixels[i] for name, i in INDEX.items()}
        frame_3d = None
        if result.pose_world_landmarks:
            world = result.pose_world_landmarks[0]
            frame_3d = {name: (world[i].x, world[i].y, world[i].z, vis[i]) for name, i in INDEX.items()}
        return frame_2d, frame_3d, pixels

    def close(self) -> None:
        self.landmarker.close()


# ----------------------------------------------------------------- drawing
def blur_face(img, pixels) -> None:
    """Blur the face area (landmarks 0-10) for privacy."""
    import cv2
    face = [(x, y) for x, y, _, v in pixels[:11] if v > 0.3]
    if len(face) < 3:
        return
    h, w = img.shape[:2]
    xs, ys = [p[0] for p in face], [p[1] for p in face]
    size = max(max(xs) - min(xs), max(ys) - min(ys), 20)
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    x0, x1 = int(max(cx - size, 0)), int(min(cx + size, w))
    y0, y1 = int(max(cy - size * 1.2, 0)), int(min(cy + size, h))
    if x1 > x0 and y1 > y0:
        k = max(31, (min(x1 - x0, y1 - y0) // 2) * 2 + 1)
        img[y0:y1, x0:x1] = cv2.GaussianBlur(img[y0:y1, x0:x1], (k, k), 0)


def draw_pose(img, pixels, exercise, info: dict) -> None:
    import cv2
    for a, b in CONNECTIONS:
        if pixels[a][3] > 0.5 and pixels[b][3] > 0.5:
            cv2.line(img, (int(pixels[a][0]), int(pixels[a][1])),
                     (int(pixels[b][0]), int(pixels[b][1])), (230, 230, 230), 2)
    for x, y, _, v in pixels[11:]:
        if v > 0.5:
            cv2.circle(img, (int(x), int(y)), 4, (255, 180, 60), -1)
    for row in info.values():  # highlight the joints used for counting
        pts = [pixels[INDEX[f"{row['side']}_{j}"]] for j in exercise.joints]
        if row["raw_angle"] is None:
            continue
        p = [(int(x), int(y)) for x, y, _, _ in pts]
        cv2.line(img, p[0], p[1], (0, 215, 255), 5)
        cv2.line(img, p[1], p[2], (0, 215, 255), 5)
        for q in p:
            cv2.circle(img, q, 8, (0, 140, 255), -1)
        s = max(img.shape[:2]) / 1000
        org = (p[1][0] + int(12 * s), p[1][1] - int(12 * s))
        cv2.putText(img, f"{row['smooth_angle']:.0f} deg", org, cv2.FONT_HERSHEY_SIMPLEX, 0.9 * s,
                    (0, 0, 0), max(3, int(5 * s)), cv2.LINE_AA)
        cv2.putText(img, f"{row['smooth_angle']:.0f} deg", org, cv2.FONT_HERSHEY_SIMPLEX, 0.9 * s,
                    (0, 215, 255), max(1, int(2 * s)), cv2.LINE_AA)


def draw_hud(img, exercise, info: dict, warning: str, fps: float) -> None:
    import cv2
    s = max(img.shape[:2]) / 1000  # scale text with the video size
    font = cv2.FONT_HERSHEY_SIMPLEX
    row_h = int(95 * s)
    overlay = img.copy()
    cv2.rectangle(overlay, (int(10 * s), int(10 * s)), (int(430 * s), int((70 + 95 * len(info)) * s)),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
    cv2.putText(img, f"{exercise.name}  |  {fps:4.1f} fps", (int(22 * s), int(42 * s)),
                font, 0.7 * s, (255, 255, 255), max(1, int(2 * s)), cv2.LINE_AA)
    y = int(95 * s)
    for key, row in info.items():
        label = f"{key.upper()} REPS: {row['count']}" if key in ("left", "right") else f"REPS: {row['count']}"
        cv2.putText(img, label, (int(22 * s), y), font, 1.3 * s, (80, 255, 120), max(2, int(3 * s)), cv2.LINE_AA)
        state = STATE_TEXT.get(row["state"], row["state"])
        cv2.putText(img, f"partial: {row['partials']}   {state}", (int(22 * s), y + int(35 * s)),
                    font, 0.65 * s, (200, 200, 200), max(1, int(2 * s)), cv2.LINE_AA)
        y += row_h
    if warning:
        org = (int(22 * s), img.shape[0] - int(25 * s))
        cv2.putText(img, warning, org, font, 0.9 * s, (0, 0, 0), max(3, int(5 * s)), cv2.LINE_AA)
        cv2.putText(img, warning, org, font, 0.9 * s, (60, 60, 255), max(1, int(2 * s)), cv2.LINE_AA)


# ----------------------------------------------------------------- processing
def output_dir_for(name: str, exercise_key: str) -> Path:
    return OUTPUT_DIR / f"{name}_{exercise_key}"


def process_landmarks(path, exercise, side="auto", smoothing=0.4, min_visibility=0.3,
                      use_3d=False, out_dir: Path | None = None) -> tuple[Session, Path]:
    """Count reps in a saved landmark sequence (CSV)."""
    rows = load_landmarks_csv(path)
    session = Session(exercise, side, smoothing, min_visibility, use_3d)
    for t, frame in rows:
        session.update(t, frame)
    session.finish()
    out_dir = out_dir or output_dir_for(Path(path).stem, exercise.key)
    write_outputs(session, out_dir, Path(path).stem)
    return session, out_dir


def process_video(source, exercise, side="auto", smoothing=0.4, min_visibility=0.3,
                  use_3d=False, model="full", show=False, save_video=True, blur=True,
                  rotate=0, out_dir: Path | None = None,
                  landmarks_csv: Path | None = None) -> tuple[Session, Path]:
    """Count reps in a video file (path) or webcam (int index)."""
    import cv2
    webcam = isinstance(source, int)
    if not webcam and not Path(source).is_file():
        raise FileNotFoundError(f"Video file '{source}' was not found.")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError("Could not open the webcam." if webcam else f"Could not open video '{source}'.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if 1 <= fps <= 240 else 30.0
    name = datetime.now().strftime("webcam_%Y%m%d_%H%M%S") if webcam else Path(source).stem
    out_dir = out_dir or output_dir_for(name, exercise.key)
    out_dir.mkdir(parents=True, exist_ok=True)
    rotations = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}

    detector = PoseDetector(ensure_model(model))
    session = Session(exercise, side, smoothing, min_visibility, use_3d)
    landmark_rows, writer = [], None
    index, t0, tick, shown_fps = 0, time.monotonic(), time.monotonic(), 0.0
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if rotate in rotations:
                img = cv2.rotate(img, rotations[rotate])
            t = time.monotonic() - t0 if webcam else index / fps
            index += 1
            frame_2d, frame_3d, pixels = detector.detect(img, t * 1000)
            info = session.update(t, frame_3d if use_3d else frame_2d)
            landmark_rows.append((t, frame_2d))

            if save_video or show:
                if pixels is None:
                    warning = "No person detected"
                elif all(r["raw_angle"] is None for r in info.values()):
                    warning = "Joints not visible - frame skipped"
                else:
                    warning = ""
                if pixels is not None:
                    if blur:
                        blur_face(img, pixels)
                    draw_pose(img, pixels, exercise, info)
                now = time.monotonic()
                shown_fps = 0.9 * shown_fps + 0.1 * (1 / max(now - tick, 1e-6)) if shown_fps else 1 / max(now - tick, 1e-6)
                tick = now
                draw_hud(img, exercise, info, warning, shown_fps)
            if save_video:
                if writer is None:
                    h, w = img.shape[:2]
                    writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"),
                                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                writer.write(img)
            if show:  # shrink large videos so the whole frame fits on the screen
                h0, w0 = img.shape[:2]
                scale = min(1.0, 1280 / w0, 720 / h0)
                view = cv2.resize(img, (int(w0 * scale), int(h0 * scale))) if scale < 1 else img
                cv2.imshow("Exercise Rep Counter (q = quit)", view)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        cap.release()
        detector.close()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()
    if index == 0:
        raise RuntimeError("No frames could be read from the video.")
    session.finish()
    save_landmarks_csv(landmarks_csv or out_dir / "landmarks.csv", landmark_rows)
    write_outputs(session, out_dir, name)
    return session, out_dir


def write_outputs(session: Session, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_signal_csv(out_dir / "signal.csv", session.log)
    plot_signal(session, out_dir / "plot.png", title=name)
    (out_dir / "summary.json").write_text(json.dumps(session.summary(), indent=2), encoding="utf-8")


def print_summary(session: Session, out_dir: Path) -> None:
    s = session.summary()
    print()
    print(f"Exercise:            {s['exercise']} - {s['signal']}")
    print(f"Thresholds:          extended >= {s['thresholds']['extended']} deg, "
          f"flexed <= {s['thresholds']['flexed']} deg")
    for side, n in s["counts"].items():
        print(f"Completed reps ({side}): {n}")
    print(f"Partial reps:        {s['partial_reps']} (not counted)")
    print(f"Rejected (too fast): {s['rejected_too_fast']}")
    print(f"Frames:              {s['frames']} ({s['duration_s']} s), "
          f"{s['missing_or_low_confidence_pct']}% missing or low confidence")
    print(f"Results saved in:    {out_dir}")


# ----------------------------------------------------------------- CLI
def main() -> None:
    p = argparse.ArgumentParser(description="Count exercise repetitions with MediaPipe pose landmarks.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="video file (mp4, mov, avi ...)")
    src.add_argument("--webcam", nargs="?", const=0, type=int, help="webcam index (default 0)")
    src.add_argument("--landmarks", help="landmark sequence CSV")
    p.add_argument("--exercise", choices=list(EXERCISES), default="curl")
    p.add_argument("--side", choices=["auto", "left", "right", "both"], default="auto",
                   help="body side to track; 'both' counts left and right separately")
    p.add_argument("--extended", type=float, help="custom extended threshold (degrees)")
    p.add_argument("--flexed", type=float, help="custom flexed threshold (degrees)")
    p.add_argument("--smoothing", type=float, default=0.4,
                   help="EMA smoothing factor 0-1 (1 = no smoothing, default 0.4)")
    p.add_argument("--min-visibility", type=float, default=0.3,
                   help="skip frames where joint visibility is below this (default 0.3)")
    p.add_argument("--world", action="store_true", help="use 3D world landmarks for the angle")
    p.add_argument("--model", choices=list(MODEL_URLS), default="full")
    p.add_argument("--show", action="store_true", help="show the annotated video while processing")
    p.add_argument("--no-video", action="store_true", help="do not save the annotated video")
    p.add_argument("--no-blur", action="store_true", help="do not blur the face in the output")
    p.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0,
                   help="rotate frames (for sideways phone videos)")
    args = p.parse_args()

    try:
        exercise = get_exercise(args.exercise, args.extended, args.flexed)
        common = dict(side=args.side, smoothing=args.smoothing,
                      min_visibility=args.min_visibility, use_3d=args.world)
        if args.landmarks:
            session, out_dir = process_landmarks(args.landmarks, exercise, **common)
        else:
            source = args.webcam if args.webcam is not None else args.input
            session, out_dir = process_video(
                source, exercise, model=args.model, show=args.show or args.webcam is not None,
                save_video=not args.no_video, blur=not args.no_blur, rotate=args.rotate, **common)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        sys.exit(f"Error: {exc}")
    except KeyboardInterrupt:
        sys.exit("Stopped by user.")
    print_summary(session, out_dir)


if __name__ == "__main__":
    main()
