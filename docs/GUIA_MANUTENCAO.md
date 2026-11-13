# Guia de manutenção e integração — Benchmark Londres

## Módulos e contratos

| Arquivo | Responsabilidade | Onde ampliar |
| --- | --- | --- |
| `scripts/prepare_data.py` | Obter CSVs explicitamente autorizados | Fonte de dados e mensagens de reprodução |
| `scripts/data_protocol.py` | Esquema, merge, variáveis e divisão temporal | Novas variáveis/anos **exigem novo protocolo** |
| `scripts/model_registry.py` | Definições de famílias e pipelines sklearn | Novos modelos e grades de hiperparâmetros |
| `scripts/london_benchmark.py` | Validação, teste, CV geográfico, bootstrap e diagnósticos | Novas métricas e análises; preserva APIs legadas |
| `scripts/run_benchmark.py` | CLI, execução completa, exportação de resultados e figuras | Novos artefatos sem alterar seleção por validação |
| `scripts/check_benchmark.py` | Validar formato dos artefatos gerados | Novos contratos de resultados |

## Cuidados para alterar

1. O alvo é o **preço médio por distrito e período**, não o preço de um imóvel individual.
2. Dados de 2017–2019 são teste final; não usá-los para decidir modelos ou hiperparâmetros.
3. Os indicadores do mesmo ano podem só estar disponíveis retrospectivamente. Não chamar o benchmark de previsão ex ante.
4. Qualquer escalonamento ou imputação deve ser ajustado somente no conjunto de treinamento de cada etapa.
5. Para avaliar novos distritos sem viés de seleção, planeje uma avaliação geográfica verdadeiramente independente (ou validação aninhada).
6. As funções de `scripts.london_benchmark` continuam importáveis; para extensões novas, prefira os módulos proprietários `data_protocol` e `model_registry`.

## Comandos de verificação

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
python -m compileall -q scripts tests
python scripts/prepare_data.py --download
python scripts/run_benchmark.py --dir data --output artifacts/benchmark
python scripts/check_benchmark.py artifacts/benchmark
```

A execução com dados reais é dependente da fonte e precisa ser verificada no CI. O projeto é acadêmico, não uma ferramenta confiável de precificação.
