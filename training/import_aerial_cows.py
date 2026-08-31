"""Importa o dataset publico aerial-cows e converte para o formato YOLO.

Fonte: https://huggingface.co/datasets/Francesco/aerial-cows
Espelho do aerial-cows do Roboflow 100. Licenca CC BY 4.0 -- uso comercial
permitido, exige atribuicao. Creditos: Omar Kapur, wwblodge, Ricardo Jenez,
Justin Jeng e Jeffrey Day.

Escreve em training/dataset_aerial_cows/, SEPARADO de training/dataset/, que
guarda os frames do nosso proprio video. Os dois convivem de proposito: um
serve para treinar agora, o outro para comparar depois com filmagem real.

Uso:
    uv run --extra training python training/import_aerial_cows.py
"""
import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEST = REPO / "training" / "dataset_aerial_cows"

# No HF os splits sao train/validation/test; o Ultralytics espera train/val.
SPLIT_MAP = {"train": "train", "validation": "val"}


def coco_to_yolo(bbox, width, height):
    """COCO [x, y, w, h] absoluto -> YOLO [cx, cy, w, h] normalizado."""
    x, y, w, h = bbox
    return (
        (x + w / 2) / width,
        (y + h / 2) / height,
        w / width,
        h / height,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=Path, default=DEST)
    parser.add_argument("--repo-id", default="Francesco/aerial-cows")
    args = parser.parse_args()

    from datasets import load_dataset

    ds = load_dataset(args.repo_id)
    print(f"Splits na origem: { {k: len(v) for k, v in ds.items()} }")

    total_imgs = total_boxes = descartadas = 0
    for origem, destino in SPLIT_MAP.items():
        img_dir = args.dest / "images" / destino
        lbl_dir = args.dest / "labels" / destino
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for ex in ds[origem]:
            stem = f"aerialcows_{ex['image_id']:06d}"
            width, height = ex["width"], ex["height"]

            linhas = []
            for bbox in ex["objects"]["bbox"]:
                cx, cy, bw, bh = coco_to_yolo(bbox, width, height)
                # Caixa degenerada ou fora do quadro quebra o treino.
                if bw <= 0 or bh <= 0 or not (0 <= cx <= 1 and 0 <= cy <= 1):
                    descartadas += 1
                    continue
                # Classe unica: 0 = cattle, igual ao training/dataset.yaml.
                linhas.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            ex["image"].convert("RGB").save(img_dir / f"{stem}.jpg", quality=95)
            (lbl_dir / f"{stem}.txt").write_text("\n".join(linhas), encoding="utf-8")
            total_imgs += 1
            total_boxes += len(linhas)

        print(f"  {origem} -> {destino}: {len(ds[origem])} imagens")

    yaml_path = args.dest.parent / "dataset_aerial_cows.yaml"
    yaml_path.write_text(
        "# Gerado por training/import_aerial_cows.py\n"
        "# Fonte: Francesco/aerial-cows (Hugging Face), CC BY 4.0\n"
        "path: dataset_aerial_cows\n"
        "train: images/train\n"
        "val: images/val\n"
        "\n"
        "names:\n"
        "  0: cattle\n",
        encoding="utf-8",
    )

    print(f"\nImagens : {total_imgs}")
    print(f"Caixas  : {total_boxes} (media {total_boxes / max(total_imgs,1):.1f}/imagem)")
    if descartadas:
        print(f"Descartadas (degeneradas): {descartadas}")
    print(f"\nDestino : {args.dest}")
    print(f"Config  : {yaml_path}")
    print(f"\nTreinar : uv run python training/train.py --data {yaml_path.name}")


if __name__ == "__main__":
    main()
