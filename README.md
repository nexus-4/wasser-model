# Wasser Model

Projeto de visao computacional para deteccao, rastreamento e contagem unica de gado em video, usando YOLO (Ultralytics) + BoT-SORT.

## O que este projeto faz

- Detecta gado em cada frame do video.
- Rastreia cada animal com um ID persistente ao longo do tempo.
- Conta cada animal apenas uma vez (contagem unica).
- Desenha caixas, nomes e IDs no frame.
- Gera um video final anotado: `resultado_wasser.mp4`.

## Como funciona

Fluxo principal em `main.py`:

1. Carrega o modelo de deteccao `yolo26x.pt`.
2. Abre o video de entrada (`media/video-teste-wasser.mp4`).
3. Executa `model.track(...)` com tracker configurado em `wasser_tracker.yaml`.
4. Filtra classe `19` (classe "cow" no COCO).
5. Usa os IDs do tracker para manter contagem unica com um `set`.
6. Salva os frames anotados no arquivo `resultado_wasser.mp4`.

## Estrutura basica

- `main.py`: pipeline principal de deteccao + tracking + contagem.
- `wasser_tracker.yaml`: hiperparametros do BoT-SORT (limiares, ReID, buffer etc.).
- `yolo26x.pt`: pesos do modelo de deteccao.
- `yolo26x-cls.pt`: modelo de aparencia para ReID no tracker.
- `media/`: videos de entrada.
- `database/`: pasta reservada para dados auxiliares.
- `requirements.txt`: dependencias Python.

## Requisitos

- Python 3.10+ (recomendado)
- macOS, Linux ou Windows
- Dependencias em `requirements.txt`:
  - `ultralytics`
  - `opencv-python`

## Setup recomendado (com ambiente virtual)

No diretorio do projeto:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Para desativar o ambiente virtual:

```bash
deactivate
```

## Como rodar

Com o ambiente virtual ativo:

```bash
python main.py
```

Durante a execucao:

- Uma janela com o tracking sera exibida.
- Pressione `q` para encerrar antes do fim do video.

Ao final:

- O video processado sera salvo como `resultado_wasser.mp4`.

## Ajustes rapidos

- Trocar video de entrada: edite `VIDEO_PATH` em `main.py`.
- Trocar modelo: edite `MODEL_PATH` em `main.py`.
- Ajustar comportamento do tracking: edite `wasser_tracker.yaml`.

## Recomendacoes

- Sempre use ambiente virtual para evitar conflito de versoes entre projetos.
- Evite commitar a pasta `venv/` no Git.
- Se quiser reproducibilidade total, fixe versoes no `requirements.txt` (ex.: `ultralytics==8.x.x`).
- Mantenha os arquivos `.pt` no caminho esperado pelo `main.py` e pelo tracker.
- Para videos grandes, valide espaco em disco antes de processar.

## Solucao de problemas

- Erro ao abrir video:
  - Verifique se `media/video-teste-wasser.mp4` existe.
- Erro de dependencia:
  - Reative o `venv` e rode `pip install -r requirements.txt` novamente.
- Janela de video nao abre:
  - Verifique suporte grafico/OpenCV no seu ambiente.
