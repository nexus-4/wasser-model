# Adiciona interface com Streamlit

Descrição do PR

 ## Resumo
 Este PR adiciona uma interface com Streamlit ao
projeto para facilitar a execução e visualização do
fluxo da aplicação.

 ## Mudanças principais
 - `app.py`
   - Implementação da interface web com Streamlit.
 - `processor.py`
   - Ajustes para integração com a nova camada de
interface.
 - `main.py`
   - Refatorações para suportar o fluxo via interface.
 - `requirements.txt`
   - Inclusão/ajuste de dependências necessárias para
o Streamlit.

 ## Impacto
 - Torna o uso do projeto mais acessível por interface
 gráfica.
 - Mantém o fluxo principal integrado com o
processamento existente.

 ## Como validar
 1. Instalar dependências (`pip install -r
requirements.txt`)
 2. Executar a interface (`streamlit run app.py`)
 3. Verificar se o fluxo principal funciona
normalmente pela UI