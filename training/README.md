# Fine-tune para gado em vista aerea

## Por que isto existe

Os pesos genericos (COCO) **nao funcionam** neste caso de uso. Medido no
video de teste do projeto (`videos/`, drone, 720x720), com `yolo26x.pt`:

| imgsz | conf | deteccoes |
| --- | --- | --- |
| 640 (padrao do projeto) | 0.25 | **0** |
| 640 | 0.10 | 1, rotulada `bird` |
| 1280 | 0.25 | **37, todas rotuladas `bird`** |

As caixas caem em cima do gado, mas o modelo o classifica como **passaro**:
de cima, um boi nao se parece com as fotos de nivel do chao do COCO. Como o
pipeline filtra `classes=[19]` ("cow"), o contador fica em zero para sempre.

Somam-se dois problemas: em `imgsz=640` quase nada e detectado (o animal e
pequeno demais), e o que e detectado sai com o rotulo errado.

Fine-tune com dados aereos rotulados resolve os dois de uma vez.

## Fluxo

```bash
# 1. Extrai frames dos videos (1 a cada 10)
uv run python training/extract_frames.py --step 10

# 2. Gera rotulos-rascunho reaproveitando as caixas do modelo generico
uv run python training/pre_annotate.py --imgsz 1280 --conf 0.25

# 3. >>> REVISE OS ROTULOS A MAO <<<  (ver secao abaixo)

# 4. Divide em train/val
uv run python training/split_dataset.py --val-ratio 0.2

# 5. Treina (o device e detectado sozinho: cuda -> mps -> cpu)
uv run python training/train.py --epochs 100 --imgsz 1280
```

## O passo 3 nao e opcional

O `pre_annotate.py` ignora o rotulo do modelo e fica so com as caixas,
marcando tudo como classe 0 (`cattle`). Isso adianta a maior parte do
trabalho, mas o rascunho **tem erro**: falsos positivos (sombras, pedras,
aves de verdade) e animais perdidos, principalmente os sobrepostos.

Treinar em cima do rascunho sem revisar so ensina o modelo a repetir os
erros do modelo generico. Abra `training/dataset/` em uma ferramenta de
rotulagem (Label Studio, CVAT, Roboflow) e corrija.

## Sobre o tamanho do dataset

O repositorio tem **um** video. Extraindo 1 a cada 10 frames saem ~109
imagens, todas da mesma cena, mesmo rebanho, mesma altitude e mesma luz.
Frames vizinhos sao quase identicos.

Isso da para validar o fluxo, **nao** para treinar um modelo que generalize.
Para producao, junte filmagens de varios dias, altitudes, horarios e pastos.
O `split_dataset.py` corta em bloco (nao embaralha) justamente para os
frames vizinhos nao vazarem do treino para a validacao e inflarem a metrica.

## Depois de treinar

Os pesos saem em `runs/<name>/weights/best.pt`. Para usar:

1. Aponte `DEFAULT_MODEL_PATH` em `processor.py` para esse arquivo.
2. Passe `class_ids=[CATTLE_CLASS_ID]` em `process_video()` — o modelo
   fine-tunado tem uma classe so, indice 0, nao a 19 do COCO.
3. Considere `imgsz=1280`; o padrao 640 perde animal pequeno.

O ReID do tracker (`model:` em `wasser_tracker.yaml`) continua usando os
pesos genericos de classificacao — e independente do detector.

## Dataset publico: aerial-cows

Para nao depender so da nossa filmagem, `import_aerial_cows.py` traz um
dataset publico de gado em vista aerea:

```bash
uv run --extra training python training/import_aerial_cows.py
uv run python training/train.py --data training/dataset_aerial_cows.yaml
```

- **1.383 imagens** (1.084 treino / 299 validacao), 640x640, classe unica
- **13.035 caixas**, media de 9,4 por imagem
- 383 imagens sem gado, uteis como negativos
- **Licenca CC BY 4.0**: uso comercial permitido, exige atribuicao

> Creditos: Omar Kapur, wwblodge, Ricardo Jenez, Justin Jeng e Jeffrey Day.
> Via Roboflow 100, espelhado em `Francesco/aerial-cows` no Hugging Face.
> A atribuicao e obrigatoria pela licenca -- mantenha este credito.

Ele vai para `training/dataset_aerial_cows/`, **separado** de
`training/dataset/`, que guarda os frames do nosso video. Os dois convivem de
proposito: um treina agora, o outro serve para comparar depois em fazenda
real.

### Os dois nao sao equivalentes

Medindo o tamanho do gado nos dois conjuntos:

| conjunto | lado equivalente | fracao da imagem |
| --- | --- | --- |
| aerial-cows | 9,4 px | 1,47% |
| nosso video | 43,5 px | 6,03% |

O dataset publico foi filmado de **muito mais alto**: o gado aparece ~4,6x
menor. Por isso o `train.py` usa `--scale 0.9` por padrao, bem acima do
0.5 usual -- sem escala agressiva o modelo aprende so o objeto minusculo e
nao generaliza para a nossa altitude.

### Limite de dominio: raca

O aerial-cows e majoritariamente gado de clima temperado, de pelagem
malhada. No Piaui a maior parte e **nelore** (pelagem clara, cupim) e
eventualmente **pe-duro**. Vistos de cima, contorno e cor sao diferentes.

Isso nao invalida o dataset -- ele ensina o modelo a reconhecer "animal de
quatro patas visto de cima", que e exatamente o que o COCO nao sabe fazer.
Mas espere queda de precisao em nelore, e planeje a segunda etapa:

1. Treinar no aerial-cows para ter um detector funcionando
2. Usar esse modelo (nao o COCO) no `pre_annotate.py`, o que torna a
   revisao humana muito mais rapida
3. Somar frames rotulados da filmagem real e retreinar

E um ciclo: cada rodada melhora o rascunho da rodada seguinte.

## GPU no treino

O `train.py` detecta o acelerador sozinho e imprime qual escolheu. Para
forcar, use `--device cuda`, `--device mps` ou `--device cpu`.

CUDA depende do extra instalado -- com `uv sync --extra cpu` o torch nao tem
CUDA e a deteccao cai para CPU, por mais que a GPU exista na maquina. Para
treinar na NVIDIA:

```bash
uv sync --extra cu130
uv run python training/train.py --epochs 100 --imgsz 1280
```

No MacBook use `--extra cpu`; o Apple Silicon acelera via MPS, que a
deteccao encontra automaticamente. Treinar em CPU pura e lento -- na medicao
do README principal a inferencia foi ~10x mais lenta que CUDA, e treino
sofre na mesma proporcao.
