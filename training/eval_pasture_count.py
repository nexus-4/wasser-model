"""Compara caixas com centros de animais marcados no vídeo de exemplo (frames base zero)."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys

import cv2
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from processor import (DEFAULT_VIDEO_PATH, PASTURE_INFERENCE_SIZE, PASTURE_MODEL_PATH,
                       pasture_detections, resolve_class_ids, resolve_device)


def erros(caixas, pontos):
    """Associa no máximo uma caixa por centro e um centro por caixa."""
    candidatos = [[i for i, (x, y) in enumerate(pontos) if x1 <= x <= x2 and y1 <= y <= y2]
                  for x1, y1, x2, y2 in caixas]
    dono = {}

    def associar(caixa, visitados):
        for alvo in candidatos[caixa]:
            if alvo in visitados:
                continue
            visitados.add(alvo)
            if alvo not in dono or associar(dono[alvo], visitados):
                dono[alvo] = caixa
                return True
        return False

    for caixa in range(len(caixas)):
        associar(caixa, set())
    sobras = set(range(len(caixas))) - set(dono.values())
    duplicadas = sum(bool(candidatos[i]) for i in sobras)
    return len(pontos) - len(dono), duplicadas, len(sobras) - duplicadas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(REPO / PASTURE_MODEL_PATH))
    args = parser.parse_args()
    pontos = defaultdict(list)
    with Path(__file__).with_name("contagem_pasto_referencia.csv").open() as fh:
        for row in csv.DictReader(fh):
            pontos[int(row["frame"])].append((int(row["x"]), int(row["y"])))
    modelo = YOLO(args.model)
    classes = resolve_class_ids(modelo)
    device = resolve_device()
    cap = cv2.VideoCapture(str(REPO / DEFAULT_VIDEO_PATH))
    try:
        print("frame,modo,referencia,contagem,perdidos,duplicatas,sem_centro")
        for indice, marcados in sorted(pontos.items()):
            cap.set(cv2.CAP_PROP_POS_FRAMES, indice)
            ok, frame = cap.read()
            if not ok or frame.shape[:2] != (720, 720):
                raise ValueError(f"Frame de referência indisponível: {indice}")
            bruto = modelo.predict(frame, classes=classes, imgsz=640, conf=0.25,
                                   device=device, verbose=False)[0].boxes.xyxy.cpu().tolist()
            pasto = pasture_detections(modelo, frame, classes=classes, imgsz=PASTURE_INFERENCE_SIZE, device=device)
            for nome, caixas in (("bruto", bruto), ("pasto", [box for box, _, _ in pasto])):
                area = [b for b in caixas if 30 <= (b[0]+b[2])/2 <= 690 and 80 <= (b[1]+b[3])/2 <= 565]
                print(indice, nome, len(marcados), len(area), *erros(area, marcados), sep=",")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
