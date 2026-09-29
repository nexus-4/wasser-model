# Wasser Model

Projeto de visao computacional para deteccao, rastreamento e contagem unica de gado em video, usando YOLO (Ultralytics) + BoT-SORT.

## O que este projeto faz

- Detecta gado em cada frame do video.
- Rastreia cada animal com um ID persistente ao longo do tempo.
- Conta cada animal apenas uma vez (contagem unica) ou por frame.
- Desenha caixas, nomes, confianca e legenda no frame.
- Gera um mapa de calor das areas de maior atividade.
- Entrega tudo por linha de comando ou por interface web (Streamlit).

## Estrutura

- `processor.py`: nucleo do pipeline (deteccao + tracking + contagem + heatmap). Nao depende de Streamlit.
- `app.py`: interface web Streamlit ("Vigilio Monitor") — upload, preview ao vivo, heatmap e download.
- `main.py`: execucao via terminal, usando o mesmo `processor.py`.
- `wasser_tracker.yaml`: hiperparametros do BoT-SORT (limiares, ReID, buffer etc.).
- `videos/`: video de teste versionado.
- `casos de teste/`: relatorio de validacao (CT01 a CT28).

## Pesos do modelo (obrigatorio)

Os arquivos `.pt` **nao sao versionados** (ver `.gitignore`). Sao necessarios dois:

- `yolo26x.pt` — modelo de deteccao (`DEFAULT_MODEL_PATH` em `processor.py`)
- `yolo26x-cls.pt` — modelo de aparencia para ReID (`model:` em `wasser_tracker.yaml`)

Coloque-os na raiz do projeto antes de rodar. Se informado apenas pelo nome, o Ultralytics tenta baixar automaticamente na primeira execucao (requer rede). Se voce apontar um caminho explicito que nao existe, o `processor.py` falha com erro claro em vez de silenciosamente nao processar.

## Requisitos

- Python 3.13+
- macOS, Linux ou Windows

## Setup via uv (recomendado)

Para instalacoes rapidas, use o [uv](https://github.com/astral-sh/uv):

1. Instale o `uv` (caso ainda nao tenha):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # Ou via Homebrew no Mac: brew install uv
   ```

2. Sincronize escolhendo **um** extra, conforme a sua maquina:

   ```bash
   # CPU. Serve em qualquer lugar; e o que o MacBook usa
   # (no macOS este wheel ja traz aceleracao MPS/Metal).
   uv sync --extra cpu

   # GPU NVIDIA (Windows/Linux). CUDA 13.0.
   uv sync --extra cu130
   ```

   Os dois extras sao mutuamente exclusivos -- o `uv` recusa instalar ambos.
   Se voce trocar de extra, rode o `uv sync` de novo: ele substitui o torch.

   > **Mac nao tem CUDA.** Apple Silicon acelera via MPS (Metal), que ja vem
   > no wheel do extra `cpu`. Use `--extra cpu` no Mac.

3. Rode os comandos com `uv run` (dispensa ativar venv):
   ```bash
   uv run streamlit run app.py
   uv run python main.py
   ```

## Setup via pip (alternativa)

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install --upgrade pip

# Escolha a variante do torch (o requirements.txt nao a define):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# ou, com GPU NVIDIA:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

pip install -r requirements.txt
```

> `requirements.txt` e um espelho do `pyproject.toml`. A fonte de verdade das versoes e o `pyproject.toml` / `uv.lock`.

## Como rodar

### Interface web

```bash
uv run streamlit run app.py
```

Abre em `http://localhost:8501`. Faca upload de um `.mp4`/`.avi`/`.mov`, clique em **Export Video**, acompanhe o preview e baixe o resultado.

A barra lateral permite ajustar modelo, tracker, tamanho de inferencia (`imgsz`) e modo do contador:

- **Accumulated total**: total de animais distintos vistos no video inteiro.
- **Current frame**: quantidade presente no frame atual.

### Linha de comando

```bash
uv run python main.py
```

Usa o video de `videos/` por padrao, abre uma janela de acompanhamento (`q` encerra) e salva `resultado_wasser.mp4` + `resultado_wasser_heatmap.jpg`.

## Docker

```bash
docker build -t wasser-model .
docker run --rm -p 8501:8501 \
  -v "$PWD/yolo26x.pt:/app/yolo26x.pt" \
  -v "$PWD/yolo26x-cls.pt:/app/yolo26x-cls.pt" \
  wasser-model
```

A imagem sobe o Streamlit em `0.0.0.0:8501`. Como os `.pt` sao excluidos pelo `.dockerignore`, eles precisam ser montados em runtime — os dois caminhos acima sao os que `processor.py` e `wasser_tracker.yaml` esperam dentro do container.

## Ajustes rapidos

- Trocar video/modelo/tracker padrao: constantes `DEFAULT_*` no topo de `processor.py`.
- Ajustar comportamento do tracking: `wasser_tracker.yaml`.
- Aumentar `imgsz` (960/1280) ajuda na deteccao de animais pequenos em filmagem de drone, ao custo de velocidade.
- Escolher o acelerador: `device=` em `process_video()`, ou o seletor
  "Processing device" na barra lateral da interface. O padrao detecta
  sozinho, na ordem CUDA -> MPS -> CPU.

## Desempenho medido

20 frames do video de teste, `yolo26x.pt` em `imgsz=1280`, nesta maquina
(RTX 4050 Laptop, 6GB):

| device | tempo | por frame |
| --- | --- | --- |
| cpu | 130.0s | 6.50s |
| cuda | 12.4s | **0.62s** |

Cerca de 10x. Para processar video de verdade, ou para treinar, use GPU.

## Limitacoes conhecidas

- **A deteccao nao funciona com os pesos genericos neste caso de uso.**
  Medido no video de teste com `yolo26x.pt`: em `imgsz=640` (o padrao) saem
  **zero** deteccoes; em `imgsz=1280` saem 37, todas rotuladas `bird`. As
  caixas caem em cima do gado, mas o COCO classifica boi visto de cima como
  passaro -- e o pipeline filtra `classes=[19]` ("cow"), entao o contador
  fica em zero. Ver `training/README.md` para o fine-tune que resolve.
- A contagem acumulada cresce a cada novo ID do tracker: se um animal e perdido e reencontrado com ID novo, ele conta duas vezes. O `track_buffer: 3000` do `wasser_tracker.yaml` mitiga, mas nao elimina.
- O filtro de tempo da interface ainda nao altera o processamento (ver CT18/CT19/CT25 no relatorio de testes).
- O projeto nao faz estimativa de peso.

## Solucao de problemas

- **Erro de arquivo nao encontrado**: confira video, tracker e pesos `.pt`.
- **Erro de dependencia**: rode `uv sync --extra cpu` (ou `--extra cu130`) novamente.
- **GPU NVIDIA existe mas nao e usada**: voce esta com o extra CPU. Rode
  `uv sync --extra cu130`. Confira com
  `uv run python -c "import torch; print(torch.cuda.is_available())"`.
- **No Mac o device sai como `cpu`**: confirme que e Apple Silicon e que o
  torch enxerga o Metal:
  `uv run python -c "import torch; print(torch.backends.mps.is_available())"`.
- **Janela de video nao abre** (modo terminal): verifique suporte grafico/OpenCV. A interface web nao precisa de display.

## Medicao com marcadores ArUco

Filmagem com marcadores ArUco no chao permite converter pixels em
centimetros, o que abre medicao dimensional e contagem com sentido.

Os marcadores usados sao `DICT_4X4_50`. **O detector enxerga o quadrado
preto**, nao a placa branca em volta -- `--marker-cm` e o lado do quadrado.

### Escala

```bash
uv run python training/aruco_scale.py VIDEO.MP4 --marker-cm 50
```

Devolve a escala (cm/px), a area coberta pelo quadro e a variacao do
marcador ao longo do voo, que indica se a altitude ficou estavel.

Medido nos voos de 14/08/2026, com marcador de 50 cm:

| video | escala | cena | variacao |
| --- | --- | --- | --- |
| ..._0018_D | 0,2506 cm/px | 9,62 x 5,41 m | 2,5% |
| ..._0019_D | 0,2882 cm/px | 11,07 x 6,22 m | 3,2% |

### Medidas em centimetros

```bash
uv run python training/measure_cattle.py VIDEO.MP4 --marker-cm 50 -o medidas.csv
```

A escala vem do marcador visivel no proprio frame, entao variacao de
altitude nao contamina a medida. Descarta animal encostado em outro (a
caixa engloba os dois) e animal tocando a borda do quadro (esta cortado).

**A medida e da caixa envolvente, da cauda ao focinho** -- nao e
comprimento corporal no sentido zootecnico. Para estimar peso, calibre
contra balanca usando o mesmo criterio nas duas pontas.

### Contagem por cruzamento de linha

```bash
uv run python training/line_count.py VIDEO.MP4 --line 0.5,0,0.5,1 --save-video saida.mp4
```

Conta animal que atravessa a linha, com sentido (entrada/saida). E mais
confiavel que o acumulo de track IDs do `processor.py`, que infla quando o
tracker perde e reencontra um animal com ID novo. A linha e dada em fracao
do quadro, entao independe da resolucao.
