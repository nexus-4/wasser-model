"""Ajusta e valida uma curva peso ~ medidas do drone.

Sem calibracao contra balanca nao existe estimativa de peso: a relacao entre
dimensao e massa depende de raca, idade e condicao corporal.

Uso:
    uv run python training/fit_weight.py sessao.csv
"""
import argparse
import csv
import math
from pathlib import Path

import numpy as np

MIN_AMOSTRAS = 8


def carregar(caminho):
    linhas = []
    with Path(caminho).open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if not r.get("peso_kg"):
                continue
            try:
                linhas.append({
                    "peso": float(r["peso_kg"]),
                    "comp": float(r["comprimento_cm"]),
                    "larg": float(r["largura_cm"]),
                })
            except (ValueError, KeyError):
                continue
    return linhas


def ajustar(x, y):
    """Regressao linear simples, devolvendo coeficientes e R2."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    a, b = np.polyfit(x, y, 1)
    pred = a * x + b
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
    erro = np.abs(pred - y)
    return a, b, r2, float(np.mean(erro)), float(np.mean(erro / y) * 100)


def validacao_cruzada(x, y):
    """Leave-one-out: erro em animal que nao entrou no ajuste."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    erros = []
    for i in range(len(x)):
        m = np.ones(len(x), bool)
        m[i] = False
        a, b = np.polyfit(x[m], y[m], 1)
        erros.append(abs(a * x[i] + b - y[i]))
    return float(np.mean(erros)), float(np.mean(np.array(erros) / y) * 100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()

    dados = carregar(args.csv)
    if len(dados) < MIN_AMOSTRAS:
        raise SystemExit(
            f"So {len(dados)} animais com peso preenchido. "
            f"Precisa de pelo menos {MIN_AMOSTRAS} para um ajuste que signifique algo "
            f"(o ideal e 25 ou mais, cobrindo a faixa de peso do rebanho)."
        )

    peso = [d["peso"] for d in dados]
    preditores = {
        "comprimento": [d["comp"] for d in dados],
        "largura": [d["larg"] for d in dados],
        "area (comp x larg)": [d["comp"] * d["larg"] for d in dados],
        "comp x larg^2": [d["comp"] * d["larg"] ** 2 for d in dados],
    }

    print(f"Animais com peso: {len(dados)}")
    print(f"Faixa de peso   : {min(peso):.0f} a {max(peso):.0f} kg\n")
    print(f"{'preditor':22} {'R2':>6} {'erro medio':>12} {'erro %':>8} {'LOO %':>8}")

    melhor = None
    for nome, x in preditores.items():
        a, b, r2, erro, pct = ajustar(x, peso)
        _, loo_pct = validacao_cruzada(x, peso)
        print(f"{nome:22} {r2:>6.3f} {erro:>9.1f} kg {pct:>7.1f}% {loo_pct:>7.1f}%")
        if melhor is None or r2 > melhor[1]:
            melhor = (nome, r2, a, b)

    nome, r2, a, b = melhor
    print(f"\nMelhor preditor: {nome}")
    print(f"  peso_kg = {a:.6f} * {nome} + {b:.2f}   (R2 = {r2:.3f})")

    if r2 < 0.5:
        print("\nR2 baixo. Provaveis causas: poucas amostras, faixa de peso estreita,")
        print("medidas ruins (confira o CSV) ou associacao errada entre animal e peso.")
    if len(dados) < 25:
        print(f"\nCom {len(dados)} animais o ajuste e preliminar. Use o erro LOO, nao o R2,")
        print("para julgar: ele mede o erro em animal que nao entrou no ajuste.")


if __name__ == "__main__":
    main()
