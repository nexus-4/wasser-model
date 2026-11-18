"""Pesagem por passagem com câmera RGB fixa acima de um corredor."""
import argparse
import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from processor import resolve_class_ids, resolve_device
from training.measure_cattle import escala_do_frame, medir_deteccao, toca_borda


def observar(video, referencia, area, modelo, escala, opcoes):
    """Uma observação por frame: quantidade detectada e dimensões quando válidas."""
    altura, largura = referencia.shape[:2]
    x1, y1, x2, y2 = area or (0, 0, largura, altura)
    if not (0 <= x1 < x2 <= largura and 0 <= y1 < y2 <= altura):
        raise ValueError("A área deve estar dentro da foto de referência.")
    while True:
        ok, frame = video.read()
        if not ok:
            break
        if frame.shape[:2] != (altura, largura):
            raise ValueError("Vídeo e foto de referência devem ter a mesma resolução.")
        roi = frame[y1:y2, x1:x2]
        resultado = modelo.predict(roi, **opcoes)[0]
        caixas = resultado.boxes
        quantidade = len(caixas) if caixas is not None else 0
        medida = None
        if quantidade == 1:
            caixa = caixas.xyxy[0].cpu().numpy()
            if not toca_borda(caixa, x2 - x1, y2 - y1):
                masks = resultado.masks.data if resultado.masks is not None else None
                orientada = medir_deteccao(roi, caixa, masks, 0)
                if orientada is not None:
                    medida = (orientada[0] * escala, orientada[1] * escala)
        yield quantidade, medida


def resumir(inicio, fim, amostras, alertas, minimo):
    if len(amostras) < minimo:
        alertas.setdefault("poucas_amostras", inicio)
    comp = larg = ""
    revisao = inicio
    if amostras:
        frames, comprimentos, larguras = np.asarray(amostras).T
        comp, larg = np.median(comprimentos), np.median(larguras)
        distancia = abs(comprimentos / comp - 1) + abs(larguras / larg - 1)
        revisao = int(frames[np.argmin(distancia)])
        comp, larg = round(float(comp), 1), round(float(larg), 1)
    return {
        "status": "revisar" if alertas else "ok",
        "motivo": ", ".join(alertas),
        "primeiro_frame": inicio,
        "ultimo_frame": fim,
        "frame_revisao": next(iter(alertas.values()), revisao),
        "amostras": len(amostras),
        "comprimento_cm": comp,
        "largura_cm": larg,
        "peso_kg": "",
        "brinco": "",
    }


def separar_passagens(observacoes, intervalo, minimo, min_razao):
    inicio = fim = None
    amostras = []
    alertas = {}
    for frame, (quantidade, medida) in enumerate(observacoes, start=1):
        if quantidade == 0:
            if inicio is not None and frame - fim >= intervalo:
                yield resumir(inicio, fim, amostras, alertas, minimo)
                inicio = None
                amostras, alertas = [], {}
            continue
        if inicio is None:
            inicio = frame
            if frame == 1:
                alertas["inicio_ocupado"] = frame
        elif frame > fim + 1:
            alertas.setdefault("deteccao_interrompida", frame)
        fim = frame
        if quantidade > 1:
            alertas.setdefault("animais_juntos", frame)
        elif medida is not None and medida[0] / medida[1] >= min_razao:
            amostras.append((frame, *medida))
    if inicio is not None:
        alertas["fim_sem_intervalo_livre"] = fim
        yield resumir(inicio, fim, amostras, alertas, minimo)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--referencia", required=True, type=Path,
                        help="foto do corredor vazio com o marcador de escala")
    parser.add_argument("--marker-cm", type=float, required=True)
    parser.add_argument("--area", type=int, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                        help="retângulo do corredor em pixels; padrão: quadro inteiro")
    parser.add_argument("--intervalo-vazio", type=int, default=5,
                        help="frames consecutivos sem detecção que separam passagens")
    parser.add_argument("--min-amostras", type=int, default=5)
    parser.add_argument("--min-razao", type=float, default=2.5)
    parser.add_argument("--model", default="yolo26x.pt")
    parser.add_argument("--device")
    parser.add_argument("--classes")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("-o", "--out", type=Path, default=Path("runs/passagens.csv"))
    args = parser.parse_args()
    if not math.isfinite(args.marker_cm) or args.marker_cm <= 0:
        parser.error("--marker-cm precisa ser positivo e finito")
    if min(args.intervalo_vazio, args.min_amostras) < 1 or not 0 < args.min_razao < math.inf:
        parser.error("intervalo, amostras e razão mínima precisam ser positivos")
    if args.out.exists():
        parser.error("CSV de saída já existe. Escolha outro nome para preservar os pesos preenchidos.")
    referencia = cv2.imread(str(args.referencia))
    if referencia is None:
        parser.error("Foto de referência ilegível.")
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    escala = escala_do_frame(referencia, detector, args.marker_cm)
    if escala is None:
        parser.error("Nenhum marcador ArUco DICT_4X4_50 na foto de referência.")
    modelo = YOLO(args.model)
    opcoes = dict(classes=resolve_class_ids(modelo, args.classes), device=resolve_device(args.device),
                  imgsz=args.imgsz, conf=args.conf, retina_masks=True, verbose=False)
    video = cv2.VideoCapture(str(args.video))
    try:
        if not video.isOpened():
            parser.error("Vídeo ilegível.")
        observacoes = observar(video, referencia, args.area, modelo, escala, opcoes)
        linhas = list(separar_passagens(observacoes, args.intervalo_vazio, args.min_amostras, args.min_razao))
    finally:
        video.release()
    if not linhas:
        raise SystemExit("Nenhuma passagem detectada.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["passagem", *linhas[0]])
        writer.writeheader()
        for numero, linha in enumerate(linhas, start=1):
            writer.writerow({"passagem": numero, **linha})
    revisar = sum(linha["status"] == "revisar" for linha in linhas)
    print(f"{len(linhas)} passagens, {revisar} para revisão. CSV: {args.out}")


if __name__ == "__main__":
    main()
