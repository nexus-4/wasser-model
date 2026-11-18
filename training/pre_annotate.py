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
import sys
from pathlib import Path

from torchvision.ops import nms
from ultralytics import YOLO

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from processor import resolve_class_ids, resolve_device

DATASET = REPO / "training" / "dataset"

# Classes COCO plausiveis para gado visto de cima. O modelo generico erra o
# rotulo de forma sistematica nesse angulo, entao aceitamos varias.
ANIMAL_LIKE_COCO = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]


def resolve_pre_annotate_classes(model, explicit_classes=None):
    """Usa o conjunto amplo de animais se o modelo for COCO; senao resolve gado."""
    if explicit_classes is not None:
        return resolve_class_ids(model, explicit_classes)
    names = getattr(model, "names", {}) or {}
    if isinstance(names, dict) and names.get(14) == "bird" and names.get(19) == "cow":
        return list(ANIMAL_LIKE_COCO)
    return resolve_class_ids(model)


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
    parser.add_argument("--classes", default=None,
                        help="IDs de classe separados por virgula (padrao: auto para COCO ou fine-tune)")
    parser.add_argument("--device", default=None,
                        help="cuda / mps / cpu (padrao: detecta automaticamente)")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45,
                        help="limiar de NMS agnostico a classe para evitar caixas duplicadas "
                             "(essencial em modelos end2end como YOLO26 ou multi-classe COCO)")
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
    device = resolve_device(args.device)
    model = YOLO(args.model)
    class_ids = resolve_pre_annotate_classes(model, args.classes)

    total_boxes = 0
    suprimidas = 0
    vazias = 0
    for image_path in images:
        result = model.predict(
            str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            classes=class_ids,
            device=device,
            verbose=False,
        )[0]

        height, width = result.orig_shape
        if result.boxes is not None and len(result.boxes):
            raw_count = len(result.boxes)
            keep = nms(result.boxes.xyxy, result.boxes.conf, args.iou)
            boxes = result.boxes.xyxy[keep].cpu().numpy()
            suprimidas += raw_count - len(boxes)
        else:
            boxes = []
        linhas = [to_yolo_line(b, width, height) for b in boxes]

        (args.labels_dir / f"{image_path.stem}.txt").write_text(
            "\n".join(linhas), encoding="utf-8"
        )
        total_boxes += len(linhas)
        vazias += 1 if not linhas else 0

    print(f"Imagens processadas : {len(images)}")
    print(f"Caixas-rascunho     : {total_boxes} ({suprimidas} duplicatas suprimidas por NMS)")
    print(f"Media por imagem    : {total_boxes / len(images):.1f}")
    print(f"Imagens sem caixa   : {vazias}")
    print(f"\nRotulos em {args.labels_dir}")
    print("REVISE antes de treinar -- ha falsos positivos e animais perdidos.")


if __name__ == "__main__":
    main()
