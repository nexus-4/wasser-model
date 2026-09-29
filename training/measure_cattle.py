"""Mede gado em centimetros usando marcadores ArUco como referencia de escala.

A escala vem do marcador visivel no proprio frame, entao variacao de altitude
nao contamina a medida. Exporta CSV para calibrar contra balanca depois.

So mede animais ISOLADOS: com vizinho colado a caixa do detector engloba os
dois e a medida vira lixo. Use --all para exportar todos com a marcacao.

Uso:
    uv run python training/measure_cattle.py VIDEO.MP4 --marker-cm 50 -o medidas.csv
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

COW_CLASS_ID = 19


def escala_do_frame(frame, detector, marker_cm):
    cantos, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    if ids is None or not len(ids):
        return None
    lados = [
        np.mean([np.linalg.norm(c[0][k] - c[0][(k + 1) % 4]) for k in range(4)])
        for c in cantos
    ]
    return marker_cm / float(np.median(lados))


def medida_orientada(roi):
    """Comprimento, largura e angulo do animal, via retangulo de area minima.

    A caixa do detector e alinhada aos eixos: um animal na diagonal vira quase
    quadrado (proporcao 1:1 em vez de 3.7:1) e a largura fica inutil.
    """
    if roi.size == 0:
        return None
    cinza = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(cinza, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    maior = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(maior) < 0.15 * roi.shape[0] * roi.shape[1]:
        return None
    (_, _), (w, h), ang = cv2.minAreaRect(maior)
    if min(w, h) <= 0:
        return None
    return max(w, h), min(w, h), ang


def toca_borda(caixa, largura, altura, margem=8):
    x1, y1, x2, y2 = caixa
    return x1 <= margem or y1 <= margem or x2 >= largura - margem or y2 >= altura - margem


def isolado(caixa, outras, folga=0.15):
    """True se nenhuma outra caixa encosta nesta, com uma folga proporcional."""
    x1, y1, x2, y2 = caixa
    mx, my = (x2 - x1) * folga, (y2 - y1) * folga
    for o in outras:
        if o is caixa:
            continue
        if not (o[0] > x2 + mx or o[2] < x1 - mx or o[1] > y2 + my or o[3] < y1 - my):
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--marker-cm", type=float, required=True,
                        help="lado do quadrado preto do marcador, em cm")
    parser.add_argument("--model", default="yolo26x.pt")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--step", type=int, default=30, help="1 frame a cada N")
    parser.add_argument("--all", action="store_true",
                        help="exporta tambem os aglomerados (coluna isolado=False)")
    parser.add_argument("-o", "--out", type=Path, default=Path("runs/medidas.csv"))
    args = parser.parse_args()

    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
        cv2.aruco.DetectorParameters(),
    )
    modelo = YOLO(args.model)
    video = cv2.VideoCapture(str(args.video))
    if not video.isOpened():
        raise SystemExit(f"Nao consegui abrir: {args.video}")
    total = int(video.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    linhas = []
    sem_escala = 0
    cortados = [0]
    try:
        for frame_no in range(0, total, args.step):
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ok, frame = video.read()
            if not ok:
                break

            cm_px = escala_do_frame(frame, detector, args.marker_cm)
            if cm_px is None:
                sem_escala += 1
                continue

            r = modelo.predict(frame, imgsz=args.imgsz, conf=args.conf,
                               classes=[COW_CLASS_ID], verbose=False)[0]
            if r.boxes is None or not len(r.boxes):
                continue

            alt, larg = frame.shape[:2]
            caixas = [tuple(b) for b in r.boxes.xyxy.cpu().numpy()]
            for caixa, cf in zip(caixas, r.boxes.conf.tolist()):
                x1, y1, x2, y2 = caixa
                if toca_borda(caixa, larg, alt):
                    cortados[0] += 1
                    continue
                so = isolado(caixa, caixas)
                if not so and not args.all:
                    continue
                w, h = (x2 - x1) * cm_px, (y2 - y1) * cm_px
                orient = medida_orientada(frame[int(y1):int(y2), int(x1):int(x2)])
                comp_o = larg_o = ang_o = razao = None
                if orient is not None:
                    comp_o = round(orient[0] * cm_px, 1)
                    larg_o = round(orient[1] * cm_px, 1)
                    ang_o = round(orient[2], 1)
                    razao = round(comp_o / larg_o, 2) if larg_o else None
                linhas.append({
                    "frame": frame_no,
                    "conf": round(cf, 3),
                    "isolado": so,
                    "cm_por_px": round(cm_px, 5),
                    "comprimento_cm": comp_o,
                    "largura_cm": larg_o,
                    "angulo": ang_o,
                    "razao": razao,
                    "bbox_comp_cm": round(max(w, h), 1),
                    "bbox_larg_cm": round(min(w, h), 1),
                    "cx_px": round((x1 + x2) / 2, 1),
                    "cy_px": round((y1 + y2) / 2, 1),
                })
    finally:
        video.release()

    if not linhas:
        raise SystemExit("Nenhuma medida. Sem marcador visivel ou sem deteccao.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)

    isolados = [x for x in linhas if x["isolado"] and x["comprimento_cm"]]
    print(f"Frames sem marcador visivel : {sem_escala}")
    print(f"Descartados na borda        : {cortados[0]}")
    print(f"Medidas exportadas          : {len(linhas)} ({len(isolados)} isoladas)")
    if isolados:
        comp = [x["comprimento_cm"] for x in isolados]
        larg = [x["largura_cm"] for x in isolados]
        print(f"Comprimento (isolados)      : mediana {np.median(comp):.0f} cm "
              f"(p10 {np.percentile(comp,10):.0f} / p90 {np.percentile(comp,90):.0f})")
        print(f"Largura (isolados)          : mediana {np.median(larg):.0f} cm "
              f"(p10 {np.percentile(larg,10):.0f} / p90 {np.percentile(larg,90):.0f})")
    print(f"\nCSV: {args.out}")
    print("\nA caixa vai da cauda ao focinho, entao NAO e comprimento corporal")
    print("no sentido zootecnico. Para calibrar contra balanca, use o mesmo")
    print("criterio nas duas pontas.")


if __name__ == "__main__":
    main()
