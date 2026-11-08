"""Verifica se o notebook executado emitiu métricas válidas e comparáveis."""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def validate_metrics(csv_path: Path) -> pd.DataFrame:
    result = pd.read_csv(csv_path, index_col="modelo")
    needed_models = {"Baseline", "Linear simples", "Linear múltipla"}
    needed_columns = {"MAE teste", "RMSE teste", "R² teste"}
    if not needed_models.issubset(result.index) or not any(
        name.startswith("Polinomial grau ") for name in result.index
    ) or len(result) != 4:
        raise ValueError("Conjunto de modelos inesperado no CSV de teste")
    if not needed_columns.issubset(result.columns) or not result.index.is_unique:
        raise ValueError("Colunas ou índices de métricas inválidos")
    values = result[list(needed_columns)].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("Há métricas inválidas (NaN/infinito)")
    if (result["MAE teste"] < 0).any() or (result["RMSE teste"] < result["MAE teste"]).any():
        raise ValueError("MAE e RMSE violam propriedades matemáticas")
    if (result["R² teste"] > 1).any():
        raise ValueError("R² acima de 1")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path, help="CSV exportado pelo notebook")
    args = parser.parse_args()
    print(validate_metrics(args.csv).round(3).to_string())


if __name__ == "__main__":
    main()
