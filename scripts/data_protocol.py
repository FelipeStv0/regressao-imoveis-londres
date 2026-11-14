"""Protocolo de dados da avaliação retrospectiva de preços médios.

INTEGRAÇÃO FUTURA: mudanças em colunas, fonte ou anos mudam o experimento;
versione o protocolo e atualize testes antes de comparar novas métricas.
Não faça download neste módulo; a obtenção é explícita em prepare_data.py.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

# Contrato público de features e intervalos de treino/validação/teste.
TARGET = "average_price"
SIMPLE = ("median_salary",)
MULTIPLE = ("median_salary", "number_of_jobs", "recycling_pct")
RANDOM_SEED = 42
TRAIN_LAST_YEAR, VAL_LAST_YEAR, TEST_LAST_YEAR = 2013, 2016, 2019


def load_housing_data(data_dir: Path) -> pd.DataFrame:
    """Junta as duas tabelas sem multiplicar linhas em merges ambíguos."""
    monthly = pd.read_csv(data_dir / "housing_in_london_monthly_variables.csv", parse_dates=["date"])
    yearly = pd.read_csv(data_dir / "housing_in_london_yearly_variables.csv", parse_dates=["date"])
    for frame, columns in [
        (monthly, {"area", "date", "borough_flag", TARGET}),
        (yearly, {"area", "date", "borough_flag", *MULTIPLE}),
    ]:
        missing = columns - set(frame.columns)
        if missing:
            raise ValueError(f"Colunas necessárias ausentes: {sorted(missing)}")
    monthly = monthly.loc[(monthly["borough_flag"] == 1) & (monthly["date"].dt.month == 12),
                          ["area", "date", TARGET]].copy()
    yearly = yearly.loc[yearly["borough_flag"] == 1, ["area", "date", *MULTIPLE]].copy()
    for label, frame in [("mensal", monthly), ("anual", yearly)]:
        if frame.duplicated(["area", "date"]).any():
            raise ValueError(f"Chaves duplicadas na tabela {label}")
    data = yearly.merge(monthly, on=["area", "date"], validate="one_to_one", how="inner")
    for feature in (*MULTIPLE, TARGET):
        data[feature] = pd.to_numeric(data[feature], errors="coerce")
    if data[TARGET].isna().any() or (data[TARGET] <= 0).any():
        raise ValueError("Alvo ausente ou não positivo; revisar fonte antes de prosseguir")
    data["year"] = data["date"].dt.year.astype(int)
    data = data[["area", "date", "year", TARGET, *MULTIPLE]].sort_values(["area", "date"]).reset_index(drop=True)
    if data.empty or data["area"].isna().any() or data[["area", "date"]].duplicated().any():
        raise ValueError("Dataset vazio, área ausente ou registros duplicados")
    return data


def temporal_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Holdouts cronológicos com anos exclusivos e sem embaralhamento."""
    train = data.loc[data["year"] <= TRAIN_LAST_YEAR].copy()
    val = data.loc[data["year"].between(TRAIN_LAST_YEAR + 1, VAL_LAST_YEAR)].copy()
    test = data.loc[data["year"].between(VAL_LAST_YEAR + 1, TEST_LAST_YEAR)].copy()
    if any(part.empty for part in (train, val, test)):
        raise ValueError("Não há observações suficientes nos três períodos temporais")
    if data["year"].max() > TEST_LAST_YEAR:
        raise ValueError("Novos anos na fonte: reveja e versiona o protocolo antes de continuar")
    if not (train["date"].max() < val["date"].min() < test["date"].min()):
        raise AssertionError("Vazamento temporal entre treino, validação e teste")
    if sum(len(x) for x in (train, val, test)) != len(data):
        raise ValueError("Há anos fora do protocolo temporal definido")
    return train, val, test
