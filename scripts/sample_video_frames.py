from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio_ffmpeg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample frames from the florr.io guide video")
    parser.add_argument("video", help="Path to mp4/video file")
    parser.add_argument("--out", default="assets/debug/video_frames", help="Output folder")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds")
    parser.add_argument("--end", type=float, default=None, help="End time in seconds")
    parser.add_argument("--every", type=float, default=5.0, help="Seconds between sampled frames")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    video = Path(args.video)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"imageio ffmpeg: {imageio_ffmpeg.get_ffmpeg_exe()}")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = frame_count / fps if frame_count else None
    end = args.end if args.end is not None else duration
    if end is None:
        raise SystemExit("Could not determine video duration; pass --end")

    t = args.start
    written = 0
    while t <= end:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            break
        out = out_dir / f"frame_{int(t):05d}s.png"
        cv2.imwrite(str(out), frame)
        written += 1
        t += args.every
    cap.release()
    print(f"Wrote {written} frame(s) to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
