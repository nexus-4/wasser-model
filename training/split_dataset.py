"""Divide o dataset plano em train/val no layout que o Ultralytics espera.

Uso:
    uv run python training/split_dataset.py --val-ratio 0.2
"""
import argparse
import random
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATASET = REPO / "training" / "dataset"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    args = parser.parse_args()

    images = sorted((args.dataset / "images").glob("*.jpg"))
    if not images:
        raise SystemExit("Nenhuma imagem. Rode extract_frames.py antes.")

    # Frames vizinhos sao quase identicos: embaralhar por frame vaza informacao
    # do treino para a validacao. Cortamos em bloco para manter os splits
    # temporalmente separados.
    corte = int(len(images) * (1 - args.val_ratio))
    splits = {"train": images[:corte], "val": images[corte:]}

    for split, arquivos in splits.items():
        img_dir = args.dataset / "images" / split
        lbl_dir = args.dataset / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for image_path in arquivos:
            shutil.copy2(image_path, img_dir / image_path.name)
            label = args.dataset / "labels" / f"{image_path.stem}.txt"
            if label.exists():
                shutil.copy2(label, lbl_dir / label.name)
        print(f"{split}: {len(arquivos)} imagens")


if __name__ == "__main__":
    main()
