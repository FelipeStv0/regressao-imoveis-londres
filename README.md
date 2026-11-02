# Previsão de preços de imóveis em Londres

Estudo de Machine Learning que compara regressão linear simples, múltipla e polinomial para estimar o preço médio de imóveis por distrito de Londres. As observações representam estatísticas agregadas por distrito e ano, não imóveis individuais.

## Arquivos

- [Regressao_Londres_Codigo.ipynb](Regressao_Londres_Codigo.ipynb) — preparação, exploração, treinamento e avaliação dos modelos.
- [Relatorio_Regressao_Londres.pdf](Relatorio_Regressao_Londres.pdf) — metodologia, resultados, gráficos e limitações.
- [requirements.txt](requirements.txt) — dependências Python.

## Dados

Fonte: [Housing in London (Kaggle)](https://www.kaggle.com/datasets/justinas/housing-in-london), com dados provenientes do London Datastore.

O notebook utiliza os arquivos `housing_in_london_monthly_variables.csv` e `housing_in_london_yearly_variables.csv`, que precisam ser obtidos separadamente na fonte.

## Como reproduzir

1. Instale as dependências: `pip install -r requirements.txt`.
2. Abra `Regressao_Londres_Codigo.ipynb` no Google Colab ou no Jupyter.
3. Coloque os dois arquivos CSV na pasta `/content/` (Colab) ou na pasta de execução local.
4. Execute as células na ordem apresentada.

## Avaliação e limitações

O estudo compara modelos com separação de treino, validação e teste por distrito, usando métricas MAE, RMSE e R². Os resultados completos e as hipóteses estão no relatório.

O projeto é experimental e não deve ser usado como sistema de precificação imobiliária: dados agregados, falta de características individuais dos imóveis e erros expressivos nos distritos de preços mais altos limitam a generalização.
