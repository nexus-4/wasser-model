"""Agrega medidas por animal para uma sessao de pesagem.

Cada animal recebe um track ID e aparece em varios frames. Aqui as medidas
de um mesmo animal viram uma linha so, com a mediana -- mais estavel que
qualquer frame isolado.

Gera CSV com a coluna peso_kg vazia. Preencha com o valor da balanca e use
fit_weight.py para ajustar a curva.

Uso:
    uv run python training/weigh_session.py VIDEO.MP4 --marker-cm 50 -o sessao.csv
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_cattle import escala_do_frame, medida_orientada, toca_borda, isolado

REPO = Path(__file__).resolve().parent.parent

COW_CLASS_ID = 19


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--marker-cm", type=float, required=True)
    parser.add_argument("--model", default="yolo26x.pt")
    parser.add_argument("--tracker", type=Path,
                        default=REPO / "wasser_tracker.yaml")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--min-razao", type=float, default=2.5,
                        help="descarta medidas com proporcao comprimento/largura abaixo "
                             "disto: indicam contorno mal segmentado")
    parser.add_argument("--min-amostras", type=int, default=5,
                        help="animal precisa aparecer em pelo menos N frames validos")
    parser.add_argument("-o", "--out", type=Path, default=Path("runs/sessao_pesagem.csv"))
    args = parser.parse_args()

    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
        cv2.aruco.DetectorParameters(),
    )
    modelo = YOLO(args.model)
    video = cv2.VideoCapture(str(args.video))
    if not video.isOpened():
        raise SystemExit(f"Nao consegui abrir: {args.video}")

    por_animal = defaultdict(lambda: {"comp": [], "larg": [], "frames": []})
    frame_no = 0
    try:
        while True:
            ok, frame = video.read()
            if not ok:
                break
            frame_no += 1

            cm_px = escala_do_frame(frame, detector, args.marker_cm)
            if cm_px is None:
                continue

            r = modelo.track(frame, persist=True, tracker=str(args.tracker),
                             classes=[COW_CLASS_ID], imgsz=args.imgsz,
                             conf=args.conf, verbose=False)[0]
            if r.boxes is None or r.boxes.id is None:
                continue

            alt, larg = frame.shape[:2]
            caixas = [tuple(b) for b in r.boxes.xyxy.cpu().numpy()]
            ids = r.boxes.id.int().cpu().tolist()

            for caixa, tid in zip(caixas, ids):
                if toca_borda(caixa, larg, alt) or not isolado(caixa, caixas):
                    continue
                x1, y1, x2, y2 = (int(v) for v in caixa)
                orient = medida_orientada(frame[y1:y2, x1:x2])
                if orient is None:
                    continue
                comp, largura_px = orient[0] * cm_px, orient[1] * cm_px
                if largura_px <= 0 or comp / largura_px < args.min_razao:
                    continue
                por_animal[tid]["comp"].append(comp)
                por_animal[tid]["larg"].append(largura_px)
                por_animal[tid]["frames"].append(frame_no)
    finally:
        video.release()

    linhas = []
    for tid, d in sorted(por_animal.items()):
        if len(d["comp"]) < args.min_amostras:
            continue
        comp = float(np.median(d["comp"]))
        larg = float(np.median(d["larg"]))
        linhas.append({
            "track_id": tid,
            "amostras": len(d["comp"]),
            "primeiro_frame": min(d["frames"]),
            "ultimo_frame": max(d["frames"]),
            "comprimento_cm": round(comp, 1),
            "largura_cm": round(larg, 1),
            "area_cm2": round(comp * larg, 0),
            "desvio_comp": round(float(np.std(d["comp"])), 1),
            "peso_kg": "",
            "brinco": "",
        })

    if not linhas:
        raise SystemExit("Nenhum animal com amostras suficientes. "
                         "Baixe --min-amostras ou --min-razao.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)

    print(f"Frames processados : {frame_no}")
    print(f"Animais no CSV     : {len(linhas)}")
    print(f"Amostras por animal: mediana {np.median([l['amostras'] for l in linhas]):.0f}")
    print(f"Comprimento        : mediana {np.median([l['comprimento_cm'] for l in linhas]):.0f} cm")
    print(f"Largura            : mediana {np.median([l['largura_cm'] for l in linhas]):.0f} cm")
    print(f"\nCSV: {args.out}")
    print("Preencha peso_kg (e brinco, se houver) e rode fit_weight.py")


if __name__ == "__main__":
    main()
