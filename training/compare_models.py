"""Compara o modelo generico (COCO) com o fine-tunado na nossa filmagem.

Roda os dois no mesmo frame e salva uma imagem lado a lado, alem de imprimir
as contagens. Serve para responder objetivamente se o fine-tune valeu.

Uso:
    uv run python training/compare_models.py --finetuned runs/aerialcows-v2/weights/best.pt
"""
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

REPO = Path(__file__).resolve().parent.parent
VIDEO = REPO / "videos" / "YTDown.com_Shorts_Video-por-drone-mostra-como-o-gado-se-mo_Media_0dqOtU8HJqg_001_720p.mp4"


def desenhar(frame, result, titulo, cor):
    img = frame.copy()
    n = 0 if result.boxes is None else len(result.boxes)
    if n:
        for b, c, cf in zip(result.boxes.xyxy.cpu().numpy(),
                            result.boxes.cls.int().tolist(),
                            result.boxes.conf.tolist()):
            x1, y1, x2, y2 = map(int, b)
            cv2.rectangle(img, (x1, y1), (x2, y2), cor, 2)
            cv2.putText(img, f"{result.names[c]} {cf:.2f}", (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, cor, 1)
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(img, f"{titulo}: {n} deteccoes", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, cor, 2)
    return img, n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetuned", type=Path, required=True)
    parser.add_argument("--generico", default="yolo26x.pt")
    parser.add_argument("--video", type=Path, default=VIDEO)
    parser.add_argument("--frame", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--out", type=Path, default=REPO / "runs" / "comparacao.jpg")
    args = parser.parse_args()

    if not args.finetuned.exists():
        raise SystemExit(f"Pesos nao encontrados: {args.finetuned}")

    cap = cv2.VideoCapture(str(args.video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Nao consegui ler o frame {args.frame}")

    # Generico: classe 19 = "cow" no COCO, que e o que o projeto usa hoje.
    gen = YOLO(args.generico)
    r_gen = gen.predict(frame, imgsz=args.imgsz, conf=args.conf, classes=[19], verbose=False)[0]
    img_gen, n_gen = desenhar(frame, r_gen, "COCO classes=[19]", (0, 165, 255))

    ft = YOLO(str(args.finetuned))
    r_ft = ft.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
    img_ft, n_ft = desenhar(frame, r_ft, "fine-tune aerial-cows", (0, 255, 0))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), cv2.hconcat([img_gen, img_ft]))

    print(f"frame {args.frame}, imgsz={args.imgsz}, conf={args.conf}")
    print(f"  generico (COCO cow) : {n_gen} deteccoes")
    print(f"  fine-tune           : {n_ft} deteccoes")
    print(f"\nComparacao salva em {args.out}")


if __name__ == "__main__":
    main()
