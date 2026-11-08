"""Executa o benchmark novo usando apenas CSVs originais e gera evidências auditáveis.

Exemplo: python scripts/run_benchmark.py --dir data --output artifacts/benchmark
Não acessa rede; obtenha os CSVs antes com scripts/prepare_data.py --download.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance

from prepare_data import ensure_data, REQUIRED_FILES
from london_benchmark import (
    RANDOM_SEED, TARGET, bootstrap_by_district, choose_from_validation,
    error_diagnostics, evaluate_final, file_sha256, grouped_cv,
    load_housing_data, temporal_split, write_markdown_report,
)


def create_figures(output: Path, test: pd.DataFrame, predictions: np.ndarray,
                   by_area: pd.DataFrame, by_year: pd.DataFrame,
                   feature_names: tuple[str, ...], importances: np.ndarray) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    actual = test[TARGET].to_numpy(dtype=float)
    predicted = np.asarray(predictions, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(actual, predicted, alpha=0.65, s=23)
    bounds = [float(min(actual.min(), predicted.min())), float(max(actual.max(), predicted.max()))]
    ax.plot(bounds, bounds, "--", color="black", label="Ideal (y = ŷ)")
    ax.set(xlabel="Preço médio observado (£)", ylabel="Preço médio previsto (£)",
           title="Teste temporal — observado × previsto")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "observado_vs_previsto.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    top = by_area.nlargest(12, "MAE").sort_values("MAE")
    ax.barh(top["area"], top["MAE"])
    ax.set(xlabel="MAE por distrito (£)", title="Distritos com maiores erros — teste temporal")
    fig.tight_layout()
    fig.savefig(figures / "erros_por_distrito.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(by_year["year"], by_year["MAE"], "o-")
    ax.set(xlabel="Ano de teste", ylabel="MAE (£)", title="Erro absoluto por ano")
    ax.set_xticks(by_year["year"].to_numpy())
    fig.tight_layout()
    fig.savefig(figures / "erros_por_ano.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    order = np.argsort(importances)
    ax.barh(np.array(feature_names)[order], importances[order])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(xlabel="Aumento médio do RMSE após permutação (£)",
           title="Importância na validação (não no teste)")
    fig.tight_layout()
    fig.savefig(figures / "importancia_validacao.png", dpi=160)
    plt.close(fig)


def run(data_dir: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    data = load_housing_data(data_dir)
    train, val, test = temporal_split(data)
    development = pd.concat([train, val], ignore_index=True)
    chosen, validation, champion = choose_from_validation(train, val)
    final, predictions, _ = evaluate_final(chosen, development, test)
    geographical = grouped_cv(chosen, development)
    test_predictions, by_area, by_year = error_diagnostics(test, predictions[champion])
    uncertainty = bootstrap_by_district(test, predictions[champion])
    champion_spec = chosen[champion]
    # Interpretabilidade APENAS na validação, antes do reajuste final.
    model = champion_spec.build().fit(train[list(champion_spec.features)], train[TARGET])
    permutation = permutation_importance(
        model, val[list(champion_spec.features)], val[TARGET],
        scoring="neg_root_mean_squared_error", n_repeats=15,
        random_state=RANDOM_SEED, n_jobs=1,
    )
    importance = pd.DataFrame({
        "variavel": champion_spec.features,
        "aumento_rmse_medio": permutation.importances_mean,
        "aumento_rmse_desvio": permutation.importances_std,
    }).sort_values("aumento_rmse_medio", ascending=False)

    metadata = {
        "protocolo": "temporal: <=2013 / 2014-2016 / 2017-2019",
        "semente": RANDOM_SEED,
        "n_total": len(data), "n_distritos": int(data["area"].nunique()),
        "n_treino": len(train), "n_validacao": len(val), "n_teste": len(test),
        "melhor_modelo_por_validacao": champion,
        "modelo_configuracoes": {key: spec.description() for key, spec in chosen.items()},
        "versao_python": platform.python_version(), "versao_sklearn": sklearn.__version__,
        "versao_pandas": pd.__version__,
        "sha256_dados": {name: file_sha256(data_dir / name) for name in REQUIRED_FILES},
        "incerteza": uncertainty,
        "aviso": "Dados históricos agregados. Não usar como precificador de imóveis.",
    }
    outputs = {
        "validacao_temporal.csv": validation,
        "teste_temporal.csv": final,
        "geografico_cv_detalhado.csv": geographical,
        "erros_por_distrito.csv": by_area,
        "erros_por_ano.csv": by_year,
        "predicoes_teste.csv": test_predictions,
        "importancia_permutacao.csv": importance,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output / filename, index=False, float_format="%.8f")
    (output / "configuracao.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    create_figures(output, test, predictions[champion], by_area, by_year,
                   champion_spec.features, permutation.importances_mean)
    write_markdown_report(output / "relatorio_benchmark.md", validation, final, champion,
                          geographical, uncertainty, by_area, metadata)
    print("=== Validação temporal (seleção, £) ===")
    print(validation.round(3).to_string(index=False))
    print("=== Teste temporal final (R² sem unidade; demais métricas em £) ===")
    print(final.round(3).to_string(index=False))
    print("=== CV geográfico em treino+validação ===")
    print(geographical.groupby("modelo")[["MAE", "RMSE", "R2"]].agg(["mean", "std"]).round(3).to_string())
    print("CAMPEÃO_VALIDACAO:", champion)
    print("BENCHMARK_OK:", output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark temporal e diagnóstico geográfico")
    parser.add_argument("--dir", default=None, help="Caminho dos CSVs; padrão data/ ou LONDON_DATA_DIR")
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmark"))
    args = parser.parse_args()
    data_dir = ensure_data(args.dir)
    run(data_dir, args.output)


if __name__ == "__main__":
    main()
