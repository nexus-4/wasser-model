# Sessao de pesagem: protocolo de campo

Objetivo: coletar pares (medida do drone, peso da balanca) para ajustar uma
curva de estimativa. **Sem esses pares nao existe estimativa de peso** -- a
relacao entre dimensao e massa depende de raca, idade e condicao corporal, e
nao ha constante universal para copiar.

## O que levar

- Drone com camera 4K
- Marcador ArUco `DICT_4X4_50`, quadrado preto de tamanho **medido com regua**
  (anote o valor: a escala inteira depende dele)
- Balanca funcionando, com o peso legivel
- Forma de identificar o animal: brinco, ou a ordem de passagem

## Como filmar

**Um animal por vez, sozinho no quadro.** Este e o ponto que mais importa.
Animal colado em outro faz a caixa do detector englobar os dois e a medida
vira lixo -- foi o que inviabilizou a medicao no curral cheio.

1. Marcador no chao, no mesmo plano do animal, dentro do quadro e sem ser
   pisado. Se o animal sobe numa plataforma, o marcador sobe junto: marcador
   no chao e animal elevado dao escalas diferentes.
2. Drone parado, camera apontada para baixo (nadir), o mais perpendicular
   possivel.
3. Altura que deixe o animal inteiro no quadro com folga. Animal cortado na
   borda mede menos que o real e e descartado.
4. Filme uns 10 segundos por animal. Varios frames viram uma mediana, que e
   mais estavel que medida unica.
5. Anote peso e identificacao **na ordem de passagem**.

## Cuidados que mudam o resultado

**Sombra.** A medida do contorno usa contraste, e sombra dura gruda no animal
e infla a largura. Filme com ceu nublado, ou perto do meio-dia, quando a
sombra fica embaixo do corpo. Comece por alguns animais de teste e confira o
CSV antes de rodar o lote inteiro.

**Altura constante.** A escala e lida do marcador em cada frame, entao mudanca
de altura nao invalida a medida -- mas manter estavel reduz ruido.

**Faixa de peso.** Escolha animais cobrindo a faixa toda do rebanho, do mais
leve ao mais pesado. Calibrar so com animais de peso parecido produz uma curva
que nao extrapola.

**Quantidade.** Abaixo de 8 animais o ajuste nao significa nada. 25 ou mais e
o razoavel para uma primeira curva.

## Depois de filmar

```bash
# 1. Uma linha por animal, com a mediana das medidas dele
uv run python training/weigh_session.py VIDEO.MP4 --marker-cm 50 -o sessao.csv

# 2. Abra sessao.csv e preencha peso_kg (e brinco) usando a ordem de passagem.
#    As colunas primeiro_frame/ultimo_frame ajudam a localizar cada animal no video.

# 3. Ajusta e valida a curva
uv run python training/fit_weight.py sessao.csv
```

O `fit_weight.py` testa comprimento, largura, area e `comp x larg^2` como
preditores, e reporta R2, erro medio e erro por validacao cruzada
(leave-one-out). **Use o erro LOO para julgar**, nao o R2: ele mede o erro em
animal que nao entrou no ajuste, que e o caso de uso real.

## O que esperar

A medida e da envoltoria do animal visto de cima -- da cauda ao focinho, nao
comprimento corporal zootecnico. Isso nao impede a calibracao, desde que o
criterio seja o mesmo sempre, mas significa que os coeficientes nao sao
comparaveis com tabelas de literatura.

Vista de cima nao captura altura nem profundidade de costela, que sao boas
preditoras de massa. Espere erro maior que uma balanca. O valor esta em
estimar sem passar o animal no tronco, nao em substituir a pesagem oficial.
