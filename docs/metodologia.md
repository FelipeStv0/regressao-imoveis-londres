# Protocolo experimental e critérios de avaliação

## Pergunta de pesquisa

**Até que ponto um conjunto reduzido de atributos socioeconômicos contemporâneos consegue explicar preços médios históricos por distrito de Londres?** O estudo original responde parcialmente a essa pergunta com uma divisão por distrito. A nova etapa busca adicionalmente testar mudança temporal, extrapolação geográfica e erros extremos.

## Unidade amostral, variáveis e origem

- **Uma observação:** distrito × dezembro de um ano (não é um imóvel individual).
- **Alvo:** `average_price`, preço médio em libras. A coluna é mensal, filtrada em dezembro.
- **Entradas:** `median_salary`, `number_of_jobs`, `recycling_pct` anuais do mesmo distrito e ano.
- **Fonte:** [Housing in London (Kaggle)](https://www.kaggle.com/datasets/justinas/housing-in-london); CSVs baixados da fonte com `kagglehub`.
- **Rastreabilidade:** artefato `configuracao.json` contém hashes SHA-256 dos CSVs e versões principais de bibliotecas.
- **Integridade:** `merge(validate="one_to_one")` e checagem de chaves (área, data) evitam duplicação involuntária; alvos inválidos interrompem a execução.

**Atenção à temporalidade:** as variáveis anuais podem só se tornar conhecidas no fim do próprio ano. Portanto, o teste temporal avalia **generalização retrospectiva condicional aos indicadores observados**, e **não** constitui um sistema de previsão ex ante. Seria necessário usar defasagens e calendários reais de publicação para oferecer previsões antes do fechamento do ano.

## Dois experimentos distintos

### A. Divisão original por distrito (notebook acadêmico)

O estudo original reserva distritos inteiros para treino, validação e teste, preservando a semente 42. É adequado como exploração de transferência geográfica para distritos ainda não vistos, mas não avalia observações cronologicamente futuras. Seu conjunto de teste deve permanecer intocado na seleção feita dentro do notebook.

### B. Holdout temporal (benchmark novo)

| Partição | Período | Uso |
| --- | --- | --- |
| Treino | 1999–2013 | Ajuste e imputação inicial |
| Validação | 2014–2016 | Escolha de hiperparâmetros de Ridge/Random Forest **e escolha do modelo** |
| Teste | 2017–2019 | Avaliação final **depois** do congelamento das escolhas |

As séries dos mesmos distritos podem atravessar as partições temporais, mas **nenhum registro de um ano posterior** treina um modelo usado para pontuar anos anteriores. Após escolher modelos e parâmetros pela validação, o ajuste final usa treino+validação. Teste não deve ser usado para refazer decisões.

É possível comparar resultados dos modelos **dentro de cada protocolo**; as métricas de teste geográfico e temporal **não são comparáveis diretamente como uma mesma tarefa**, pois amostras, período e distribuição diferem.

### Modelos e controle da complexidade

- **DummyRegressor** de média: referencial obrigatório.
- **Regressão linear simples:** apenas salário mediano.
- **Regressão linear múltipla:** salário, empregos e reciclagem.
- **Ridge:** mesmos três atributos, regularização L2; alpha avaliado na validação: 0,1; 1; 10; 100; 1000.
- **Random Forest:** mesmos três atributos, 200 árvores, semente 42, `min_samples_leaf` escolhido entre 2, 5 e 10 na validação.

Imputação mediana e, para modelos lineares, padronização estão **dentro de pipelines**; em cada ajuste, essas estatísticas são aprendidas apenas com a parte de treino. A escolha do melhor candidato de cada família é baseada no RMSE de validação; o campeão geral também é definido somente pela validação. Não se troca o campeão em função de seu resultado no teste.

### Avaliação geográfica complementar

`GroupKFold(n_splits=5)` calcula MAE, RMSE e R² em distritos inteiros não observados na dobra de treinamento, usando **apenas os anos do período de desenvolvimento (até 2016)**. O conjunto de teste temporal permanece fora desta avaliação.

**Limitação importante:** os hiperparâmetros já foram escolhidos com o período de desenvolvimento; logo, a validação cruzada é um **diagnóstico exploratório** e pode ter viés de seleção. Uma estimativa de transferência geográfica após seleção exigiria avaliação aninhada (`nested group CV`).

### Estatísticas e explicabilidade

- **MAE (libras):** erro absoluto médio, menos sensível a outliers que RMSE.
- **RMSE (libras):** penaliza mais fortemente grandes diferenças.
- **R²:** pode ser negativo se o desempenho for pior que uma previsão constante de referência para aquele conjunto.
- **Erro por distrito e por ano:** torna falhas localizadas visíveis, sem esconder extremos na média.
- **Importância por permutação:** calculada com o modelo já selecionado **sobre validação** (não teste), medindo aumento do RMSE após embaralhar cada atributo. Não é prova de causalidade; variáveis correlacionadas podem repartir relevância.
- **ICs percentis descritivos de 95%:** bootstrap com 1.000 reamostragens de **distritos inteiros**, pois anos do mesmo distrito não são independentes. Não garantem cobertura de previsão futura nem eliminam viés temporal.

## Riscos e honestidade científica

Os dados são históricos, agregados e nominais; não incluem metragem, tipo, padrão construtivo ou geolocalização precisa. Há poucos distritos independentes e mudanças econômicas no período. A melhoria de uma métrica não equivale à validação comercial. Não usar para laudos, crédito, preço de anúncio ou decisão de investimento.

## Reproduzir as evidências

Na raiz do repositório:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
python scripts/prepare_data.py --download
python scripts/run_benchmark.py --dir data --output artifacts/benchmark
python scripts/check_benchmark.py artifacts/benchmark
```

O GitHub Actions automatiza esses passos, além de executar o notebook original e gerar figuras e relatórios. Os artefatos guardam arquivos derivados, não os CSVs da fonte. Sem conexão com a fonte, `prepare_data.py` admite CSVs locais obtidos legalmente.

## Próximos estudos (fora do escopo e sem promessas de qualidade preditiva)

Comparação temporal com atributos defasados, validação aninhada por distrito, modelos de intervalo e calibração, mudança de distribuição, ajuste pela inflação e fontes adicionais com características de imóveis individuais.
