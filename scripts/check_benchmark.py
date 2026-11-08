"""Garante presença e consistência dos artefatos do benchmark real em CI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def validate(directory: Path) -> dict:
    filenames = [
        "validacao_temporal.csv", "teste_temporal.csv", "geografico_cv_detalhado.csv",
        "erros_por_distrito.csv", "erros_por_ano.csv", "predicoes_teste.csv",
        "importancia_permutacao.csv", "configuracao.json", "relatorio_benchmark.md",
        "figures/observado_vs_previsto.png", "figures/erros_por_distrito.png",
        "figures/erros_por_ano.png", "figures/importancia_validacao.png",
    ]
    missing = [name for name in filenames if not (directory / name).is_file()
               or (directory / name).stat().st_size == 0]
    if missing:
        raise ValueError(f"Artefatos ausentes ou vazios: {missing}")
    meta = json.loads((directory / "configuracao.json").read_text(encoding="utf-8"))
    val = pd.read_csv(directory / "validacao_temporal.csv")
    test = pd.read_csv(directory / "teste_temporal.csv")
    if len(val) != 5 or len(test) != 5 or set(val["modelo"]) != set(test["modelo"]):
        raise ValueError("Esperados cinco modelos idênticos na validação e no teste")
    for table in [val, test]:
        for column in ["MAE", "RMSE", "R2"]:
            if not np.isfinite(pd.to_numeric(table[column], errors="raise")).all():
                raise ValueError(f"Métrica inválida: {column}")
        if not ((table["MAE"] >= 0) & (table["RMSE"] >= table["MAE"])
                & (table["R2"] <= 1)).all():
            raise ValueError("Propriedades das métricas violadas")
    best_validation = val.loc[val["RMSE"].idxmin(), "modelo"]
    if best_validation != meta["melhor_modelo_por_validacao"]:
        raise ValueError("Seleção de modelo dependeu indevidamente do teste")
    preds = pd.read_csv(directory / "predicoes_teste.csv")
    if len(preds) != meta["n_teste"]:
        raise ValueError("Predições e tamanho de teste divergem")
    cv = pd.read_csv(directory / "geografico_cv_detalhado.csv")
    if len(cv) != 25 or cv.groupby("modelo")["fold"].nunique().ne(5).any():
        raise ValueError("GroupKFold incompleto")
    if meta["n_treino"] + meta["n_validacao"] + meta["n_teste"] != meta["n_total"]:
        raise ValueError("Partição não cobre dataset")
    if meta["n_distritos"] < 5:
        raise ValueError("Dataset possui distritos insuficientes")
    if set(meta["sha256_dados"]) != {
        "housing_in_london_monthly_variables.csv", "housing_in_london_yearly_variables.csv",
    } or any(len(value) != 64 for value in meta["sha256_dados"].values()):
        raise ValueError("Hashes de origem incompletos")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    meta = validate(args.output)
    print("BENCHMARK_VALIDADO:", meta["melhor_modelo_por_validacao"])
    print("Observações:", meta["n_total"], "| Teste:", meta["n_teste"])


if __name__ == "__main__":
    main()
