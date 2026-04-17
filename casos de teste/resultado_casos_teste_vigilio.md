## Relatorio de testes - Sistema Vigilio

Data da analise: 2026-04-16

### Escopo e metodo

A validacao abaixo foi feita com base em:

- analise estatica de `app.py`, `processor.py` e `main.py`
- leitura do video em `videos/YTDown.com_Shorts_Video-por-drone-mostra-como-o-gado-se-mo_Media_0dqOtU8HJqg_001_720p.mp4`
- execucao de verificacoes locais com OpenCV
- execucao parcial de funcoes auxiliares do `processor.py`

### Video utilizado

- Arquivo: `videos/YTDown.com_Shorts_Video-por-drone-mostra-como-o-gado-se-mo_Media_0dqOtU8HJqg_001_720p.mp4`
- Resolucao: `720 x 720`
- FPS: `30`
- Total de frames: `1086`
- Tamanho: `4.861.885 bytes`

### Limitacoes encontradas

1. O projeto nao inclui os pesos `yolo26x.pt`, entao o processamento completo do video nao pode ser executado neste repositorio.
2. Ao tentar carregar um modelo inexistente, a biblioteca `ultralytics` tenta acessar a rede para buscar assets, o que falha no ambiente atual.
3. `main.py` e `processor.py` usam como video padrao `media/video-teste-wasser.mp4`, mas o repositorio atual possui a pasta `videos/`.
4. Os testes de responsividade, rede lenta e precisao de IA dependem de browser, dispositivo ou ground truth, e por isso nao puderam ser medidos de ponta a ponta aqui.

### Achados principais do codigo

- O app implementa upload de video, processamento, preview de video anotado, preview de heatmap e download do video processado.
- As cores de confianca estao implementadas com estes limites:
  - alta: `>= 0.8` -> verde
  - media: `>= 0.5` -> amarelo
  - baixa: `< 0.5` -> vermelho
- A legenda de confianca e desenhada em todos os frames exportados.
- O contador e desenhado com fundo preto e texto branco.
- O botao `Export Video` inicia o processamento, mas o download so aparece depois, em outro botao (`Download processed video`).
- O filtro de tempo existe na interface, mas o valor selecionado nao e usado em nenhum processamento.
- O upload restringe extensoes `mp4`, `avi` e `mov`, mas nao existe tratamento dedicado para "arquivo invalido" com mensagem funcional especifica.

### Resultado dos casos de teste

Legenda:

- `Aprovado`: comportamento implementado e confirmado por codigo ou execucao local
- `Parcial`: ha algo implementado, mas nao atende integralmente o esperado
- `Reprovado`: o comportamento esperado nao esta implementado
- `Bloqueado`: nao foi possivel executar o teste de ponta a ponta no ambiente atual

| ID | Caso | Status | Evidencia / observacao |
| --- | --- | --- | --- |
| CT01 | Carregar video processado | Bloqueado | O fluxo existe no app, mas o processamento completo depende do modelo `yolo26x.pt`, ausente no repositorio. |
| CT02 | Validar contagem | Bloqueado | A logica de contagem existe (`counted_ids` para total unico e `frame_track_ids` para frame atual), mas nao foi possivel gerar saida real sem o modelo. |
| CT03 | Sincronia | Bloqueado | O tracker desenha caixas a cada frame via `model.track(... persist=True ...)`, mas a validacao visual depende da execucao do modelo. |
| CT04 | Confianca alta | Aprovado | `get_color_and_status()` retorna verde para `conf >= 0.8`. |
| CT05 | Confianca media | Aprovado | `get_color_and_status()` retorna amarelo para `0.5 <= conf < 0.8`. |
| CT06 | Confianca baixa | Aprovado | `get_color_and_status()` retorna vermelho para `conf < 0.5`. |
| CT07 | Legenda | Aprovado | `draw_legend()` e chamada em todos os frames antes da exportacao. |
| CT08 | Legibilidade contador | Aprovado | `draw_counter()` desenha fundo preto opaco com texto branco, o que favorece contraste. |
| CT09 | Atualizacao tempo real | Parcial | O preview e atualizado via callback durante o processamento, mas so a cada `preview_interval=5` frames, nao a cada frame. |
| CT10 | Exportacao | Parcial | O clique em `Export Video` nao baixa o arquivo; ele apenas processa o video. O download aparece depois em `Download processed video`. |
| CT11 | Resolucao exportada | Aprovado | O `VideoWriter` usa largura e altura do video original (`width`, `height`). |
| CT12 | Sem animais | Bloqueado | Nao existe video vazio no repositorio e o processamento real nao rodou sem o modelo. |
| CT13 | Alta densidade | Bloqueado | O video possui gado, mas nao foi possivel medir performance real sem o modelo. |
| CT14 | Exportar antes | Aprovado | O botao `Export Video` fica desabilitado quando `uploaded_video is None`. |
| CT15 | Arquivo invalido | Parcial | A UI restringe extensoes no uploader, mas nao ha fluxo robusto para exibir erro funcional claro para conteudo invalido. |
| CT16 | Gerar heatmap | Bloqueado | A funcionalidade existe (`save_activity_heatmap()`), mas o heatmap real depende da execucao do tracking. |
| CT17 | Escala cores | Aprovado | O heatmap usa `cv2.COLORMAP_JET`, compatível com gradiente azul -> amarelo -> vermelho. |
| CT18 | Filtro 1h | Reprovado | Nao existe opcao de `1h`; a UI oferece `15 min`, `30 min` e `2h`. Alem disso, o filtro nao altera processamento nenhum. |
| CT19 | Filtro 2h | Reprovado | A opcao `2h` existe, mas o valor selecionado nao e utilizado em lugar nenhum. |
| CT20 | Sem dados | Reprovado | Nao ha logica de periodo vazio. O placeholder exibido fala apenas que o mapa real sera gerado apos a analise. |
| CT21 | Alta concentracao | Aprovado | Pelo uso do `COLORMAP_JET`, regioes de maior intensidade tendem ao vermelho. |
| CT22 | Baixa concentracao | Aprovado | Pelo uso do `COLORMAP_JET`, regioes de menor intensidade tendem ao azul. |
| CT23 | Troca rapida | Parcial | A troca visual do selectbox nao deve travar, mas o filtro nao tem efeito funcional real. |
| CT24 | Sem coordenadas | Parcial | Sem heatmap gerado, o app mostra area vazia com mensagem; nao existe conceito explicito de "sem coordenadas". |
| CT25 | Persistencia | Reprovado | Nao existe logica de persistencia nem atualizacao baseada no filtro selecionado. |
| CT26 | Responsividade | Bloqueado | Ha CSS customizado, mas sem teste em mobile real e sem media queries dedicadas. |
| CT27 | Rede lenta | Bloqueado | Nao foi executado em ambiente com simulacao de rede 4G. |
| CT28 | Precisao IA | Bloqueado | Nao foi possivel medir recall/precision sem modelo e sem base rotulada. |
| CT29 | Falsos positivos | Bloqueado | Nao foi possivel avaliar erros de deteccao sem executar o modelo sobre o video. |

### Evidencias de implementacao relevantes

- Upload de video com restricao de extensao: `app.py`, bloco do `st.file_uploader(...)`
- Botao de exportacao bloqueado antes do upload: `app.py`, `disabled=uploaded_video is None`
- Botao de download separado do processamento: `app.py`, `st.download_button(...)`
- Filtro de tempo sem uso funcional: `app.py`, `st.selectbox(...)` sem leitura posterior do valor
- Limiares de confianca: `processor.py`, funcao `get_color_and_status()`
- Legenda e contador: `processor.py`, funcoes `draw_legend()` e `draw_counter()`
- Heatmap: `processor.py`, funcoes `add_detection_to_heatmap()`, `render_activity_heatmap()` e `save_activity_heatmap()`

### Conclusao

O sistema implementa boa parte da base visual de `HT01` e da geracao de heatmap de `HT02`, mas varios criterios ainda nao estao fechados funcionalmente. Os principais gaps observados foram:

- ausencia do modelo no repositorio, impedindo validacao fim a fim
- filtro temporal apenas visual, sem impacto real
- fluxo de exportacao diferente do esperado no caso de teste
- tratamento de arquivo invalido incompleto
- falta de evidencia objetiva para testes nao funcionais

Com o modelo correto disponivel no projeto, os proximos testes prioritarios seriam: `CT01`, `CT02`, `CT03`, `CT12`, `CT13`, `CT16`, `CT28` e `CT29`.
