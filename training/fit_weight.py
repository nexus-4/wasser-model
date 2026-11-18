"""Ajusta e valida uma curva peso ~ medidas do drone.

Sem calibracao contra balanca nao existe estimativa de peso: a relacao entre
dimensao e massa depende de raca, idade e condicao corporal.

Uso:
    uv run python training/fit_weight.py sessao.csv
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

MIN_AMOSTRAS = 8


def carregar(caminho):
    por_animal = defaultdict(list)
    with Path(caminho).open(encoding="utf-8-sig", newline="") as fh:
        for linha, r in enumerate(csv.DictReader(fh), start=2):
            if r.get("status", "ok") != "ok" or not r.get("peso_kg"):
                continue
            try:
                valores = [float(r[col]) for col in ("peso_kg", "comprimento_cm", "largura_cm")]
            except (ValueError, TypeError, KeyError):
                raise ValueError(f"Linha {linha}: peso ou dimensoes invalidas.") from None
            if not all(np.isfinite(v) and v > 0 for v in valores):
                raise ValueError(f"Linha {linha}: peso e dimensoes devem ser positivos e finitos.")
            # Um tracker pode atribuir varios IDs ao mesmo animal na sessao.
            brinco = (r.get("brinco") or "").strip()
            por_animal[brinco or linha].append(valores)
    return [
        dict(zip(("peso", "comp", "larg"), np.median(amostras, axis=0)))
        for amostras in por_animal.values()
    ]


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
        if np.ptp(x[m]) == 0:
            return float("inf"), float("inf")
        a, b = np.polyfit(x[m], y[m], 1)
        erros.append(abs(a * x[i] + b - y[i]))
    return float(np.mean(erros)), float(np.mean(np.array(erros) / y) * 100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()

    try:
        dados = carregar(args.csv)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
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
    print(f"{'preditor':22} {'R2':>6} {'erro medio':>12} {'erro %':>8} {'LOO kg':>8} {'LOO %':>8}")

    melhor = None
    for nome, x in preditores.items():
        loo_kg, loo_pct = validacao_cruzada(x, peso)
        if not np.isfinite(loo_kg):
            print(f"{nome:22} variacao insuficiente para validacao")
            continue
        a, b, r2, erro, pct = ajustar(x, peso)
        print(f"{nome:22} {r2:>6.3f} {erro:>9.1f} kg {pct:>7.1f}% {loo_kg:>8.1f} {loo_pct:>7.1f}%")
        if melhor is None or loo_kg < melhor[0]:
            melhor = (loo_kg, nome, r2, a, b)

    if melhor is None:
        raise SystemExit("Nenhum preditor com variacao suficiente entre animais.")
    loo_kg, nome, r2, a, b = melhor
    print(f"\nMelhor preditor: {nome} (erro absoluto medio LOO: {loo_kg:.1f} kg)")
    print(f"  peso_kg = {a:.6f} * {nome} + {b:.2f}   (R2 = {r2:.3f})")

    if r2 < 0.5:
        print("\nR2 baixo. Provaveis causas: poucas amostras, faixa de peso estreita,")
        print("medidas ruins (confira o CSV) ou associacao errada entre animal e peso.")
    if len(dados) < 25:
        print(f"\nCom {len(dados)} animais o ajuste e preliminar. Use o erro LOO, nao o R2,")
        print("para julgar: ele mede o erro em animal que nao entrou no ajuste.")


if __name__ == "__main__":
    main()
