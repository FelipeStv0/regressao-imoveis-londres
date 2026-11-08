# Model Card — Housing in London (exploratório)

## Identificação
- **Tipo:** regressões tabulares (DummyRegressor, LinearRegression, Ridge, Random Forest e modelos polinomiais no estudo original).
- **Finalidade:** demonstrar experimentação, avaliação, controle de vazamento e reprodutibilidade em Machine Learning.
- **Dados:** médias de preços históricas por distrito e ano, associadas a indicadores socioeconômicos contemporâneos.
- **Artefatos:** notebook acadêmico e benchmark tabular executados pela automação do GitHub.

## Aplicações permitidas
Aprendizado, comparação metodológica de modelos, exploração de diagnósticos e demonstração de competências em portfólio.

## Aplicações desaconselhadas
- Avaliação de imóveis individuais, aluguel, preço de anúncio, aprovação de crédito ou análise de garantias.
- Recomendações financeiras ou de investimento imobiliário.
- Projeções econômicas ou previsões futuras com indicadores do mesmo ano não disponíveis no momento de decisão.

## Limitações e riscos
- **Erro de unidade:** uma linha é a média de um distrito e ano, não uma transação.
- **Tempo:** atributos contemporâneos do próprio ano podem ser conhecidos só retrospectivamente.
- **Distribuição:** regiões de preços extremos podem dominar o RMSE; períodos futuros podem divergir dos históricos.
- **Pequena amostra geográfica:** 33 distritos e observações seriadas dentro de cada área.
- **Vieses:** diferenças socioeconômicas e variáveis omitidas afetam interpretação; importância por permutação não indica causalidade.
- **Incerteza:** bootstrap por distrito ajuda a descrever variabilidade, mas não formaliza cobertura para novos imóveis.

## Avaliações implementadas
- Holdout geográfico no notebook original, com separação por distrito.
- Holdout cronológico no benchmark (2017–2019 retidos para teste, sem seleção sobre esse período).
- GroupKFold por distrito para diagnóstico complementar dentro do período de desenvolvimento.
- MAE, RMSE, R², erros por área/ano, dispersão observado × previsto e importância por permutação em validação.

## Acompanhamento e reprodução
Execute `python scripts/run_benchmark.py --dir data --output artifacts/benchmark` e consulte `artifacts/benchmark/relatorio_benchmark.md`, hashes SHA-256 e versões em `configuracao.json`. Resultados podem mudar se a fonte ou dependências mudarem; não substitua uma medição por outra sem registrar origem e versão.

## Resultados observados na avaliação temporal de 08/11/2026

Na execução [#42533710022](https://github.com/FelipeStv0/regressao-imoveis-londres/actions/runs/42533710022), o Random Forest foi escolhido na validação (RMSE £225.919), e no teste posterior obteve MAE £122.305, RMSE £207.682, R² = −0,014; o baseline apresentou RMSE £301.299 e R² = −1,133. Apesar da redução relativa de RMSE, **R² negativo impede alegar capacidade preditiva satisfatória**. Consulte [análise detalhada](resultados.md).

## Situação
**Protótipo didático; sem validação para implantação ou decisões comerciais.**
