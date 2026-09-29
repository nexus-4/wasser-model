"""Deriva escala metrica de um video de drone usando marcadores ArUco.

O OpenCV detecta o QUADRADO PRETO do marcador, nao a placa branca em volta.
Entao --marker-cm deve ser o lado do quadrado preto, nao o da placa.

Uso:
    uv run python training/aruco_scale.py VIDEO.MP4 --marker-cm 46
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

DICIONARIOS = {
    "4x4_50": cv2.aruco.DICT_4X4_50,
    "4x4_250": cv2.aruco.DICT_4X4_250,
    "5x5_250": cv2.aruco.DICT_5X5_250,
    "6x6_250": cv2.aruco.DICT_6X6_250,
    "apriltag_36h11": cv2.aruco.DICT_APRILTAG_36h11,
}


def lado_medio(canto):
    p = canto[0]
    return float(np.mean([np.linalg.norm(p[k] - p[(k + 1) % 4]) for k in range(4)]))


def medir(video_path, dicionario, passo):
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(DICIONARIOS[dicionario]),
        cv2.aruco.DetectorParameters(),
    )
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise SystemExit(f"Nao consegui abrir: {video_path}")

    total = int(video.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    largura = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    lados, amostras = [], 0
    try:
        for frame_no in range(0, total, passo):
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ok, frame = video.read()
            if not ok:
                break
            amostras += 1
            cantos, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            if ids is None:
                continue
            lados.extend(lado_medio(c) for c in cantos)
    finally:
        video.release()

    return lados, amostras, largura, altura


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--marker-cm", type=float, required=True,
                        help="lado do QUADRADO PRETO em cm (nao o da placa branca)")
    parser.add_argument("--dict", default="4x4_50", choices=sorted(DICIONARIOS))
    parser.add_argument("--step", type=int, default=60, help="1 amostra a cada N frames")
    args = parser.parse_args()

    lados, amostras, largura, altura = medir(args.video, args.dict, args.step)
    if not lados:
        raise SystemExit("Nenhum marcador detectado. Confira o dicionario com --dict.")

    mediana = float(np.median(lados))
    cm_por_px = args.marker_cm / mediana
    variacao = (max(lados) - min(lados)) / mediana * 100

    print(f"Video   : {args.video.name}")
    print(f"Resolucao: {largura}x{altura} | {amostras} amostras | {len(lados)} marcadores medidos")
    print()
    print(f"Marcador : {mediana:.1f} px  =  {args.marker_cm:.1f} cm")
    print(f"Escala   : {cm_por_px:.4f} cm/px   ({1/cm_por_px:.2f} px/cm)")
    print(f"Cena     : {largura*cm_por_px/100:.2f} m x {altura*cm_por_px/100:.2f} m")
    print(f"Variacao do marcador: {variacao:.1f}%  "
          f"({'altitude estavel' if variacao < 10 else 'ATENCAO: altitude mudou muito'})")
    print()
    print("Para converter qualquer medida do video:")
    print(f"  centimetros = pixels * {cm_por_px:.4f}")


if __name__ == "__main__":
    main()
