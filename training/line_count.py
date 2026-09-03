"""Conta gado que cruza uma linha, com sentido.

Alternativa a contagem por acumulo de track IDs, que infla quando o tracker
perde e reencontra um animal com ID novo. Aqui um animal so conta quando
atravessa a linha, e o sentido separa entrada de saida.

A linha e dada em fracao do quadro (0..1), entao independe da resolucao.

Uso:
    uv run python training/line_count.py VIDEO.MP4 --line 0.5,0,0.5,1
    uv run python training/line_count.py VIDEO.MP4 --line 0,0.6,1,0.6 --save-video saida.mp4
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

COW_CLASS_ID = 19


def distancia_com_sinal(ponto, a, b):
    """Distancia do ponto a reta AB, com sinal indicando o lado."""
    ab = np.array(b, dtype=float) - np.array(a, dtype=float)
    n = np.linalg.norm(ab)
    cross = ab[0] * (ponto[1] - a[1]) - ab[1] * (ponto[0] - a[0])
    return cross / (n + 1e-9)


def dentro_do_segmento(ponto, a, b, folga=60):
    """Evita contar cruzamento na extensao infinita da reta, fora do segmento."""
    ab = np.array(b) - np.array(a)
    t = np.dot(np.array(ponto) - np.array(a), ab) / (np.dot(ab, ab) + 1e-9)
    margem = folga / (np.linalg.norm(ab) + 1e-9)
    return -margem <= t <= 1 + margem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--line", default="0.5,0,0.5,1",
                        help="x1,y1,x2,y2 em fracao do quadro (0..1)")
    parser.add_argument("--model", default="yolo26x.pt")
    parser.add_argument("--tracker", default="wasser_tracker.yaml")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--deadzone", type=float, default=0.04,
                        help="zona morta em volta da linha, em fracao da diagonal do "
                             "quadro. Animal parado sobre a linha oscila entre os dois "
                             "lados e conta varias vezes sem isto.")
    parser.add_argument("--save-video", type=Path)
    args = parser.parse_args()

    fx1, fy1, fx2, fy2 = (float(v) for v in args.line.split(","))
    modelo = YOLO(args.model)
    video = cv2.VideoCapture(str(args.video))
    if not video.isOpened():
        raise SystemExit(f"Nao consegui abrir: {args.video}")

    largura = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS) or 30
    a = (fx1 * largura, fy1 * altura)
    b = (fx2 * largura, fy2 * altura)

    saida = None
    if args.save_video:
        saida = cv2.VideoWriter(str(args.save_video), cv2.VideoWriter_fourcc(*"mp4v"),
                                fps, (largura, altura))

    zona = args.deadzone * float(np.hypot(largura, altura))
    lado_confirmado = {}
    entradas = saidas = 0
    eventos = []

    try:
        frame_no = 0
        while True:
            ok, frame = video.read()
            if not ok:
                break
            frame_no += 1

            r = modelo.track(frame, persist=True, tracker=args.tracker,
                             classes=[COW_CLASS_ID], imgsz=args.imgsz,
                             conf=args.conf, verbose=False)[0]

            if r.boxes is not None and r.boxes.id is not None:
                for caixa, tid in zip(r.boxes.xyxy.cpu().numpy(),
                                      r.boxes.id.int().cpu().tolist()):
                    x1, y1, x2, y2 = caixa
                    centro = ((x1 + x2) / 2, (y1 + y2) / 2)
                    d = distancia_com_sinal(centro, a, b)
                    if abs(d) < zona:
                        continue
                    lado = 1 if d > 0 else -1
                    anterior = lado_confirmado.get(tid)
                    if anterior is not None and anterior != lado:
                        if dentro_do_segmento(centro, a, b):
                            if lado > 0:
                                entradas += 1
                                eventos.append((frame_no, tid, "entrada"))
                            else:
                                saidas += 1
                                eventos.append((frame_no, tid, "saida"))
                    lado_confirmado[tid] = lado

            if saida is not None:
                vis = frame.copy()
                if r.boxes is not None and r.boxes.id is not None:
                    for caixa in r.boxes.xyxy.cpu().numpy():
                        x1, y1, x2, y2 = map(int, caixa)
                        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
                d = np.array(b, dtype=float) - np.array(a, dtype=float)
                nrm = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9) * zona
                for s_ in (1, -1):
                    p1 = (np.array(a) + nrm * s_).astype(int)
                    p2 = (np.array(b) + nrm * s_).astype(int)
                    cv2.line(vis, tuple(p1), tuple(p2), (80, 80, 80), 2)
                cv2.line(vis, tuple(map(int, a)), tuple(map(int, b)), (0, 200, 255), 5)
                cv2.rectangle(vis, (0, 0), (760, 90), (0, 0, 0), -1)
                cv2.putText(vis, f"Entradas: {entradas}   Saidas: {saidas}   Saldo: {entradas-saidas}",
                            (16, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
                saida.write(vis)
    finally:
        video.release()
        if saida is not None:
            saida.release()

    print(f"Frames processados: {frame_no}")
    print(f"Entradas : {entradas}")
    print(f"Saidas   : {saidas}")
    print(f"Saldo    : {entradas - saidas}")
    if eventos:
        print("\nprimeiros cruzamentos:")
        for f, tid, sentido in eventos[:10]:
            print(f"  frame {f:>5}  ID {tid:>3}  {sentido}")
    if args.save_video:
        print(f"\nVideo: {args.save_video}")


if __name__ == "__main__":
    main()
