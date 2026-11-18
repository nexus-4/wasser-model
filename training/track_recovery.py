"""Recupera IDs após falhas curtas usando a aparência observada nos dois extremos."""
from bisect import bisect_left
from collections import Counter

import cv2
import numpy as np


def seguir_recorte(frames, indices, caixa):
    centro = (caixa[:2] + caixa[2:]) / 2
    w, h = np.rint((caixa[2:] - caixa[:2]) * 0.65).astype(int)
    if min(w, h) < 8:
        return None
    raio = max(4, round(min(w, h) * 0.6))
    molde = cv2.getRectSubPix(frames[indices[0]], (w, h), tuple(centro))
    if molde.std() < 6:
        return None
    for indice in indices[1:]:
        frame = frames[indice]
        busca_w, busca_h = w + 2 * raio, h + 2 * raio
        metade = np.array([busca_w, busca_h]) / 2
        if np.any(centro < metade) or np.any(centro + metade >= frame.shape[::-1]):
            return None
        busca = cv2.getRectSubPix(frame, (busca_w, busca_h), tuple(centro))
        scores = cv2.matchTemplate(busca, molde, cv2.TM_CCOEFF_NORMED)
        _, score, _, (dx, dy) = cv2.minMaxLoc(scores)
        if score < 0.65:
            return None
        centro += [dx - raio, dy - raio]
    return centro


def chega_ao_centro(ponto, caixa):
    return ponto is not None and np.all(abs(ponto - (caixa[:2] + caixa[2:]) / 2)
                                        <= (caixa[2:] - caixa[:2]) * 0.15)


def mesma_aparencia(video, fim, inicio, anterior, nova, passo):
    # Só mantém na memória o corredor entre as duas caixas durante a falha.
    tamanho = np.maximum(anterior[2:] - anterior[:2], nova[2:] - nova[:2])
    origem = np.maximum(0, np.floor(np.minimum(anterior[:2], nova[:2]) - tamanho)).astype(int)
    limite = np.ceil(np.maximum(anterior[2:], nova[2:]) + tamanho).astype(int)
    ida = [*range(fim, inicio, passo), inicio]
    volta = [*range(inicio, fim, -passo), fim]
    amostrar = set(ida + volta)
    frames = {}
    if not video.set(cv2.CAP_PROP_POS_FRAMES, fim):
        raise ValueError(f"Falha ao posicionar vídeo no frame {fim}.")
    for i in range(fim, inicio + 1):
        if not video.grab():
            raise ValueError(f"Falha ao avançar ao frame {i}.")
        if i in amostrar:
            ok, frame = video.retrieve()
            if not ok:
                raise ValueError(f"Falha ao ler frame {i}.")
            roi = frame[origem[1]:limite[1], origem[0]:limite[0]]
            frames[i] = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    deslocamento = np.tile(origem, 2)
    anterior, nova = anterior - deslocamento, nova - deslocamento
    return (chega_ao_centro(seguir_recorte(frames, ida, anterior), nova)
            and chega_ao_centro(seguir_recorte(frames, volta, nova), anterior))


def recuperar_ids(video, registros, caixas, minimo, fps):
    """Novo ID -> predecessor, limitado a pares confirmados, disjuntos e sem empate."""
    confirmados = [r for r in registros.values() if r["deteccoes"] >= minimo]
    finais = sorted((r["ultimo_frame"], r["id"]) for r in confirmados)
    tempos = [f for f, _ in finais]
    candidatos = []
    for r in confirmados:
        inicio, novo = r["primeiro_frame"], r["id"]
        nova = caixas[novo][0]
        wh = nova[2:] - nova[:2]
        for fim, antigo in finais[bisect_left(tempos, inicio - round(3 * fps)):bisect_left(tempos, inicio)]:
            anterior = caixas[antigo][1]
            wh_antiga = anterior[2:] - anterior[:2]
            if np.any(np.maximum(wh, wh_antiga) > 1.7 * np.minimum(wh, wh_antiga)):
                continue
            distancia = np.linalg.norm((anterior[:2] + anterior[2:] - nova[:2] - nova[2:]) / 2)
            if distancia > 2 * max(wh.max(), wh_antiga.max()):
                continue
            if mesma_aparencia(video, fim, inicio, anterior, nova, max(1, round(fps / 8))):
                candidatos.append((antigo, novo))
    saidas = Counter(antigo for antigo, _ in candidatos)
    entradas = Counter(novo for _, novo in candidatos)
    return {novo: antigo for antigo, novo in candidatos if saidas[antigo] == entradas[novo] == 1}
