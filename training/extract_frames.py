"""Extrai frames dos videos para montar o dataset de fine-tune.

Uso:
    uv run python training/extract_frames.py --step 10
"""
import argparse
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parent.parent
VIDEOS_DIR = REPO / "videos"
OUT_DIR = REPO / "training" / "dataset" / "images"


def extract(video_path, out_dir, step):
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir o video: {video_path}")

    stem = video_path.stem[:40]
    saved = index = 0
    try:
        while True:
            ok, frame = video.read()
            if not ok:
                break
            if index % step == 0:
                cv2.imwrite(str(out_dir / f"{stem}_{index:06d}.jpg"), frame)
                saved += 1
            index += 1
    finally:
        video.release()
    return saved, index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=10, help="salva 1 a cada N frames")
    parser.add_argument("--videos-dir", type=Path, default=VIDEOS_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    videos = sorted(
        p for p in args.videos_dir.iterdir()
        if p.suffix.lower() in {".mp4", ".avi", ".mov"}
    )
    if not videos:
        raise SystemExit(f"Nenhum video encontrado em {args.videos_dir}")

    total = 0
    for video_path in videos:
        saved, read = extract(video_path, args.out_dir, args.step)
        print(f"{video_path.name}: {saved} frames salvos de {read}")
        total += saved
    print(f"Total: {total} frames em {args.out_dir}")


if __name__ == "__main__":
    main()
