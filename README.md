# Regressão de imóveis em Londres

Estudo **acadêmico e exploratório** de regressão linear simples, múltipla e polinomial sobre **preços médios agregados por distrito e ano** em Londres. O alvo não é o preço de um imóvel individual. O projeto **não é uma ferramenta confiável de avaliação ou precificação imobiliária**.

## O que o projeto avalia

- Fonte: [Housing in London — Kaggle](https://www.kaggle.com/datasets/justinas/housing-in-london), com indicadores originalmente associados ao London Datastore.
- Alvo: `average_price` (£), observação de dezembro de cada distrito/ano.
- Atributos: `median_salary`; na regressão múltipla, também `number_of_jobs` e `recycling_pct`.
- Modelos: baseline (média de treino), linear simples, linear múltipla e polinomial (graus 2, 3 e 5 avaliados na validação).
- Divisão **por distrito**, não por linhas: no resultado original, 399 observações no treino, 147 na validação e 147 no teste, de 33 distritos. A divisão é reproduzível com semente 42. O teste é usado apenas após a escolha do grau.

O pipeline usa imputação pela mediana e padronização ajustadas apenas sobre treino, para reduzir vazamento de informação. A partição por distrito evita que um mesmo distrito apareça em treino e teste, **mas não comprova generalização para anos futuros**.

## Comparação resumida de resultados

**Resultados reproduzidos em ambiente limpo em 08/11/2026:** o [GitHub Actions (execução #42500276468)](https://github.com/FelipeStv0/regressao-imoveis-londres/actions/runs/42500276468) instalou as dependências em Python 3.11, obteve os CSVs pelo Kaggle, executou todas as células do notebook e validou os resultados. As métricas de teste coincidiram com as saídas históricas do estudo. Consulte o [artefato da execução](https://github.com/FelipeStv0/regressao-imoveis-londres/actions/runs/42500276468/artifacts/13625745262) para obter o notebook executado e os CSVs.

| Modelo | MAE no teste (£) ↓ | RMSE no teste (£) ↓ | R² no teste ↑ |
| --- | ---: | ---: | ---: |
| Baseline (média) | 169.966 | 270.542 | -0,064 |
| Regressão linear simples | 137.579 | 254.067 | 0,062 |
| Regressão linear múltipla | 139.172 | 256.869 | 0,041 |
| Regressão polinomial (grau 2) | 137.459 | 249.094 | 0,098 |

**Interpretação:** o polinômio de grau 2 apresentou menor RMSE no teste registrado, mas todos os R² ficaram próximos de zero. A regressão múltipla foi melhor na validação (RMSE ≈ £107.852 contra ≈ £136.130 do baseline), sem manter a vantagem no teste. Os erros absolutos são elevados; uma posição acima do baseline não implica utilidade prática para precificar imóveis.

- **MAE:** média do erro absoluto, em libras.
- **RMSE:** raiz do erro quadrático médio, em libras; dá maior peso a erros extremos.
- **R²:** proporção de variação explicada em relação à previsão pela média do conjunto avaliado; pode ser negativo.

## Reproduzir em um ambiente novo

Requisitos: **Python 3.11**, acesso à fonte Kaggle quando optar pelo download, Git e ambiente Jupyter.

```bash
git clone https://github.com/FelipeStv0/regressao-imoveis-londres.git
cd regressao-imoveis-londres
python -m venv .venv
# Windows (PowerShell): .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
```

**Dados:** coloque os arquivos `housing_in_london_monthly_variables.csv` e `housing_in_london_yearly_variables.csv` na pasta `data/`; ou **autorize explicitamente** a obtenção por meio da API oficial:

```bash
python scripts/prepare_data.py --download
```

A integração usa `kagglehub` e a identificação `justinas/housing-in-london`. O download pode exigir conexão, autenticação ou aceite dos termos de acesso no Kaggle. Se houver restrição, obtenha manualmente os CSVs conforme autorizado pela fonte. **Os arquivos de dados não são redistribuídos neste repositório.** Não inclua credenciais Kaggle no Git.

Para executar todas as células, na ordem, sem utilizar saídas antigas:

```bash
mkdir -p artifacts
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 --output executed.ipynb --output-dir artifacts Regressao_Londres_Codigo.ipynb
python scripts/check_results.py artifacts/metricas_teste.csv
```

Também é possível abrir o notebook no Jupyter/Colab, **com o repositório completo disponível**. A leitura aceita `data/`, a raiz do projeto ou `/content` no Colab; o caminho pode ser definido por `LONDON_DATA_DIR`. O download automático no notebook, quando desejado, exige `LONDON_DATA_AUTO_DOWNLOAD=1`. Sem essa autorização, a ausência dos CSVs produz uma mensagem de erro com instruções.

## Testes e evidências de execução

O workflow [Reprodutibilidade do notebook](.github/workflows/reproducibility.yml) no GitHub Actions (push, pull request e execução manual) realiza, em Python 3.11 e ambiente limpo:

1. Instalação das dependências e **testes offline** de obtenção e validação das métricas.
2. Download explícito do dataset pela API do Kaggle, respeitando controles de acesso.
3. Execução integral do notebook via `nbconvert` e exportação de `artifacts/metricas_validacao.csv` e `artifacts/metricas_teste.csv`.
4. Verificação de estrutura e sanidade das métricas e publicação do **notebook executado e CSVs** como artefatos do job (retenção: 14 dias).

Consulte a aba **Actions** para verificar se houve sucesso ou falha. **Verificação:** na [execução #42500276468](https://github.com/FelipeStv0/regressao-imoveis-londres/actions/runs/42500276468), os 6 testes offline passaram, os dois arquivos foram obtidos com sucesso (675.287 e 79.913 bytes), e a execução integral produziu os quatro resultados da tabela. **Isso demonstra funcionamento técnico naquela execução, não confiabilidade estatística para produção.** Se a fonte estiver indisponível, o job deverá falhar de forma visível, sem simular métricas reais.

## Arquivos principais

| Caminho | Finalidade |
| --- | --- |
| [Regressao_Londres_Codigo.ipynb](Regressao_Londres_Codigo.ipynb) | Análise, visualizações, treinamento, avaliação e exportação de métricas |
| [Relatorio_Regressao_Londres.pdf](Relatorio_Regressao_Londres.pdf) | Relatório acadêmico original, mantido para referência histórica |
| [scripts/prepare_data.py](scripts/prepare_data.py) | Resolução local dos CSVs e download autorizado |
| [scripts/check_results.py](scripts/check_results.py) | Validação do contrato das métricas de teste |
| [tests/test_pipeline.py](tests/test_pipeline.py) | Testes locais sem rede |
| [requirements.txt](requirements.txt) e [requirements-dev.txt](requirements-dev.txt) | Dependências de execução e teste |

## Limitações metodológicas e uso responsável

Os indicadores são agregados por distrito; **não** contemplam metragem, conservação, localização exata ou outras características de um imóvel. O período histórico utiliza valores nominais não ajustados pela inflação. Há poucas unidades geográficas independentes, dependências temporais dentro de cada distrito e grande sensibilidade a distritos caros. A seleção por validação e o teste em distritos diferentes não substituem validação temporal, análise de incerteza, dados atuais e controle de mudanças de distribuição.

**Conclusão:** material educacional de comparação de modelos. Não usar suas previsões como laudos, preços de anúncio, avaliação de garantia, recomendação de investimento ou decisão financeira.
