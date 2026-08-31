"""Gera rotulos-rascunho no formato YOLO para revisao humana.

Aproveita que o modelo generico DETECTA o gado em vista aerea (com caixas
corretas) mas o ROTULA errado -- normalmente como 'bird'. Ignoramos o rotulo
e ficamos so com as caixas, atribuindo a classe unica 0 = cattle.

Isto NAO substitui a revisao humana: as caixas trazem falsos positivos
(sombras, pedras, aves de verdade) e perdem animais. Abra o resultado em uma
ferramenta de rotulagem (Label Studio, CVAT, Roboflow) e corrija antes de
treinar.

Uso:
    uv run python training/pre_annotate.py --imgsz 1280 --conf 0.25
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

REPO = Path(__file__).resolve().parent.parent
DATASET = REPO / "training" / "dataset"

# Classes COCO plausiveis para gado visto de cima. O modelo generico erra o
# rotulo de forma sistematica nesse angulo, entao aceitamos varias.
ANIMAL_LIKE_COCO = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]


def to_yolo_line(box_xyxy, width, height, class_id=0):
    x1, y1, x2, y2 = box_xyxy
    cx = ((x1 + x2) / 2) / width
    cy = ((y1 + y2) / 2) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo26x.pt")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--images-dir", type=Path, default=DATASET / "images")
    parser.add_argument("--labels-dir", type=Path, default=DATASET / "labels")
    args = parser.parse_args()

    images = sorted(
        p for p in args.images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        raise SystemExit(
            f"Nenhuma imagem em {args.images_dir}. Rode extract_frames.py antes."
        )

    args.labels_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)

    total_boxes = 0
    vazias = 0
    for image_path in images:
        result = model.predict(
            str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            classes=ANIMAL_LIKE_COCO,
            verbose=False,
        )[0]

        height, width = result.orig_shape
        boxes = [] if result.boxes is None else result.boxes.xyxy.cpu().numpy()
        linhas = [to_yolo_line(b, width, height) for b in boxes]

        (args.labels_dir / f"{image_path.stem}.txt").write_text(
            "\n".join(linhas), encoding="utf-8"
        )
        total_boxes += len(linhas)
        vazias += 1 if not linhas else 0

    print(f"Imagens processadas : {len(images)}")
    print(f"Caixas-rascunho     : {total_boxes}")
    print(f"Media por imagem    : {total_boxes / len(images):.1f}")
    print(f"Imagens sem caixa   : {vazias}")
    print(f"\nRotulos em {args.labels_dir}")
    print("REVISE antes de treinar -- ha falsos positivos e animais perdidos.")


if __name__ == "__main__":
    main()
