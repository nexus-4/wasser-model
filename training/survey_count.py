"""Estima trajetórias de gado por trecho contínuo de um voo e identifica prováveis replays."""
import argparse
import csv
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import YAML
from ultralytics.utils.checks import check_yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from processor import (PASTURE_INFERENCE_SIZE, PASTURE_MODEL_PATH, pasture_boxes,
                       prepare_browser_video, resolve_class_ids, resolve_device)

from training.track_recovery import recuperar_ids


def assinatura(frame):
    h, w = frame.shape[:2]
    # Evita textos e marcas d'água perto da borda.
    centro = frame[h//12:h-h//8, w//20:w-w//20]
    mini = cv2.resize(centro, (32, 24), interpolation=cv2.INTER_AREA)
    pixels = cv2.cvtColor(mini, cv2.COLOR_BGR2GRAY).astype(np.float32).ravel()
    pixels -= pixels.mean()
    norma = np.linalg.norm(pixels)
    return pixels / norma if norma > 0 else pixels


def dividir_trechos(video, passo):
    trechos = []
    inicio = 0
    amostras = []
    anterior = None
    indice = 0
    while True:
        ok, frame = video.read()
        if not ok:
            break
        atual = assinatura(frame)
        if anterior is not None and np.any(atual) and np.any(anterior) and atual @ anterior < 0.65:
            trechos.append((inicio, indice, np.array(amostras)))
            inicio, amostras = indice, []
        if (indice - inicio) % passo == 0:
            amostras.append(atual)
        anterior = atual
        indice += 1
    if amostras:
        trechos.append((inicio, indice, np.array(amostras)))
    return trechos


def provavel_replay(atual, original):
    if min(len(atual), len(original)) < 6:
        return False
    marcos = atual[np.linspace(0, len(atual)-1, min(64, len(atual)), dtype=int)]
    if np.min(marcos @ marcos[0]) > 0.85:
        return False  # Pouca variação para distinguir uma cena parada de um replay.
    similaridade = marcos @ original.T
    posicoes = similaridade.argmax(axis=1)
    return (np.all(similaridade.max(axis=1) >= 0.98)
            and np.all(np.diff(posicoes) >= 0) and len(np.unique(posicoes)) >= 6)


def rastrear(video, inicio, fim, modelo, opcoes, minimo, saida, fps=30, auditoria=None):
    config = SimpleNamespace(**YAML.load(check_yaml("bytetrack.yaml")))
    # A posição é mais estável que a confiança do detector neste vídeo aéreo.
    config.fuse_score = False
    tracker = BYTETracker(config)
    registros = {}
    caixas = {}
    por_frame = []
    if not video.set(cv2.CAP_PROP_POS_FRAMES, inicio):
        raise ValueError(f"Falha ao posicionar vídeo no frame {inicio}.")
    for indice in range(inicio, fim):
        ok, frame = video.read()
        if not ok:
            raise ValueError(f"Falha ao ler frame {indice}.")
        boxes = pasture_boxes(modelo, frame, **opcoes)
        deteccoes = (boxes.cpu().numpy() if boxes is not None else
                     Boxes(np.empty((0, 6), np.float32), frame.shape[:2]))
        centros = []
        for x1, y1, x2, y2, tid, *_ in tracker.update(deteccoes):
            tid = int(tid)
            x, y = int((x1+x2)/2), int((y1+y2)/2)
            caixa = np.array([x1, y1, x2, y2])
            centros.append((tid, x, y))
            if tid not in registros:
                caixas[tid] = [caixa, caixa]
                registros[tid] = dict(id=tid, primeiro_frame=indice, x_inicial=x, y_inicial=y, deteccoes=0)
            caixas[tid][1] = caixa
            r = registros[tid]
            r.update(ultimo_frame=indice, x_final=x, y_final=y, deteccoes=r["deteccoes"]+1)
        por_frame.append(np.asarray(centros, dtype=np.int32))

    anteriores = recuperar_ids(video, registros, caixas, minimo, fps)
    canonicos, consolidados = consolidar(registros, anteriores, auditoria)
    desenhar_trajetorias(video, inicio, por_frame, canonicos, minimo, saida)
    return [{**r, "confirmada": int(r["deteccoes"] >= minimo)} for r in consolidados.values()]


def consolidar(registros, anteriores, auditoria):
    canonicos = {}
    consolidados = {}
    for tid, r in registros.items():  # ordem da primeira observação
        raiz = canonicos[anteriores[tid]] if tid in anteriores else tid
        canonicos[tid] = raiz
        if tid == raiz:
            consolidados[raiz] = {**r, "ids_origem": str(tid)}
            continue
        anterior = anteriores[tid]
        if auditoria is not None:
            auditoria.append(dict(id_final=raiz, id_anterior=anterior, id_novo=tid,
                                  frame_antes=registros[anterior]["ultimo_frame"], frame_depois=r["primeiro_frame"]))
        grupo = consolidados[raiz]
        grupo.update(ultimo_frame=r["ultimo_frame"], x_final=r["x_final"], y_final=r["y_final"],
                     deteccoes=grupo["deteccoes"] + r["deteccoes"], ids_origem=f"{grupo['ids_origem']},{tid}")
    return canonicos, consolidados


def desenhar_trajetorias(video, inicio, por_frame, canonicos, minimo, saida):
    if not video.set(cv2.CAP_PROP_POS_FRAMES, inicio):
        raise ValueError(f"Falha ao posicionar vídeo no frame {inicio}.")
    contagens = {}
    confirmadas = 0
    for indice, centros in enumerate(por_frame, start=inicio):
        ok, frame = video.read()
        if not ok:
            raise ValueError(f"Falha ao ler frame {indice}.")
        for bruto, x, y in centros:
            tid = canonicos[bruto]
            contagens[tid] = contagens.get(tid, 0) + 1
            if contagens[tid] == minimo:
                confirmadas += 1
            if contagens[tid] >= minimo:
                cv2.circle(frame, (x, y), 3, (70, 245, 150), -1, cv2.LINE_AA)
                for cor, largura in [((0, 0, 0), 3), ((255, 255, 255), 1)]:
                    cv2.putText(frame, str(tid), (x+3, y-3), 0, .43, cor, largura, cv2.LINE_AA)
        titulo = f"Trajetorias: {confirmadas} | Frame: {indice}"
        for cor, largura in [((0, 0, 0), 4), ((255, 255, 255), 1)]:
            cv2.putText(frame, titulo, (8, 26), 0, .7, cor, largura, cv2.LINE_AA)
        saida.write(frame)


def escrever_csv(path, colunas, linhas):
    with path.open("w", newline="", encoding="utf-8") as fh:
        escritor = csv.DictWriter(fh, fieldnames=colunas.split())
        escritor.writeheader()
        escritor.writerows(linhas)


def analisar(args, pasta):
    video = cv2.VideoCapture(str(args.video))
    try:
        if not video.isOpened():
            raise ValueError(f"Falha ao abrir {args.video}")
        fps = video.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Vídeo sem taxa de quadros válida.")
        tamanho = tuple(int(video.get(k)) for k in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
        trechos = dividir_trechos(video, max(1, round(fps / 5)))
        if not trechos:
            raise ValueError("Nenhum frame legível.")
        modelo = YOLO(args.model)
        opcoes = dict(classes=resolve_class_ids(modelo), imgsz=PASTURE_INFERENCE_SIZE, device=resolve_device(args.device))
        originais, resumo, trajetorias, reassociacoes = [], [], [], []
        for numero, (inicio, fim, amostras) in enumerate(trechos, start=1):
            repetido = next((n for n, ref in originais if provavel_replay(amostras, ref)), None)
            linha = dict(trecho=numero, inicio=inicio, fim_exclusivo=fim,
                         replay_provavel_de=repetido or "", trajetorias_confirmadas="")
            if repetido is None:
                originais.append((numero, amostras))
                path = pasta / f"trecho_{numero}.mp4"
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, tamanho)
                try:
                    if not writer.isOpened():
                        raise ValueError(f"Falha ao abrir saída {path}")
                    pares = []
                    tracks = rastrear(video, inicio, fim, modelo, opcoes, args.min_amostras, writer, fps, pares)
                    reassociacoes.extend(dict(trecho=numero, **p) for p in pares)
                finally:
                    writer.release()
                prepare_browser_video(path)
                linha["trajetorias_confirmadas"] = sum(r["confirmada"] for r in tracks)
                trajetorias.extend(dict(trecho=numero, **r) for r in tracks)
            resumo.append(linha)
        escrever_csv(pasta / "trechos.csv", "trecho inicio fim_exclusivo replay_provavel_de trajetorias_confirmadas", resumo)
        escrever_csv(pasta / "trajetorias.csv", "trecho id primeiro_frame ultimo_frame x_inicial y_inicial x_final y_final deteccoes confirmada ids_origem", trajetorias)
        escrever_csv(pasta / "reassociacoes.csv", "trecho id_final id_anterior id_novo frame_antes frame_depois", reassociacoes)
    finally:
        video.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--model", default=str(REPO / PASTURE_MODEL_PATH))
    parser.add_argument("--device")
    parser.add_argument("--min-amostras", type=int, default=5, help="detecções necessárias para contar uma trajetória")
    parser.add_argument("-o", "--out", type=Path, required=True, help="nova pasta de saída")
    args = parser.parse_args()
    if args.min_amostras < 1 or args.out.exists():
        parser.error("Use um mínimo positivo e uma pasta de saída nova.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent) as tmp:
        pasta = Path(tmp) / "resultado"
        pasta.mkdir()
        analisar(args, pasta)
        pasta.rename(args.out)
    print(f"Revisão dos trechos e trajetórias: {args.out}")


if __name__ == "__main__":
    main()
