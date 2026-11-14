"""Benchmark reproduzível para generalização temporal e geográfica.

Separação temporal: treino <= 2013; validação 2014–2016; teste 2017–2019.
Nenhuma decisão de seleção de modelo ou hiperparâmetro usa o teste.
As variáveis anuais são contemporâneas aos preços; isto NÃO é previsão ex ante.

INTEGRAÇÃO FUTURA: o protocolo vive em data_protocol.py e os modelos em
model_registry.py. Este módulo reexporta suas APIs para manter notebooks,
scripts e testes existentes compatíveis.
"""
from __future__ import annotations

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

if __package__:
    from .data_protocol import (TARGET, SIMPLE, MULTIPLE, RANDOM_SEED,
        TRAIN_LAST_YEAR, VAL_LAST_YEAR, TEST_LAST_YEAR, load_housing_data, temporal_split)
    from .model_registry import ModelSpec, candidates
else:
    from data_protocol import (TARGET, SIMPLE, MULTIPLE, RANDOM_SEED,
        TRAIN_LAST_YEAR, VAL_LAST_YEAR, TEST_LAST_YEAR, load_housing_data, temporal_split)
    from model_registry import ModelSpec, candidates

# Reexports explícitos: contratos anteriores usados em notebooks e testes.
__all__ = [
    "TARGET", "SIMPLE", "MULTIPLE", "RANDOM_SEED",
    "TRAIN_LAST_YEAR", "VAL_LAST_YEAR", "TEST_LAST_YEAR",
    "load_housing_data", "temporal_split", "ModelSpec", "candidates",
    "metric_values", "choose_from_validation", "evaluate_final", "grouped_cv",
    "bootstrap_by_district", "error_diagnostics", "file_sha256",
    "write_markdown_report",
]


def metric_values(y_true, predicted) -> dict[str, float]:
    actual = np.asarray(y_true, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    if actual.ndim != 1 or actual.shape != pred.shape or len(actual) == 0:
        raise ValueError("Dimensões de previsões incompatíveis")
    if not np.isfinite(pred).all():
        raise ValueError("Previsões NaN ou infinitas")
    return {
        "MAE": float(mean_absolute_error(actual, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, pred))),
        "R2": float(r2_score(actual, pred)),
    }


def _predict(spec: ModelSpec, train: pd.DataFrame, test: pd.DataFrame):
    estimator = spec.build()
    estimator.fit(train[list(spec.features)], train[TARGET])
    return estimator, estimator.predict(test[list(spec.features)])


def choose_from_validation(train: pd.DataFrame, val: pd.DataFrame):
    """Escolhe parâmetros com validação; o teste permanece intocado."""
    chosen, records = {}, []
    for family, variants in candidates().items():
        scored = []
        for spec in variants:
            model, predictions = _predict(spec, train, val)
            scored.append((metric_values(val[TARGET], predictions)["RMSE"], spec, model, predictions))
        _best_rmse, spec, _fitted, predictions = min(scored, key=lambda item: item[0])
        chosen[family] = spec
        records.append({"modelo": family, "configuracao": spec.description(),
                        **metric_values(val[TARGET], predictions)})
    table = pd.DataFrame.from_records(records).sort_values("RMSE").reset_index(drop=True)
    champion = str(table.iloc[0]["modelo"])
    return chosen, table, champion


def evaluate_final(chosen: dict[str, ModelSpec], train_val: pd.DataFrame, test: pd.DataFrame):
    """Reajuste pós-seleção em treino+validação; teste consultado só aqui."""
    scores, predictions, fitted = [], {}, {}
    for family, spec in chosen.items():
        model, pred = _predict(spec, train_val, test)
        fitted[family] = model
        predictions[family] = np.asarray(pred, dtype=float)
        scores.append({"modelo": family, "configuracao": spec.description(),
                       **metric_values(test[TARGET], pred)})
    return pd.DataFrame(scores).sort_values("RMSE").reset_index(drop=True), predictions, fitted


def grouped_cv(chosen: dict[str, ModelSpec], development: pd.DataFrame, splits: int = 5) -> pd.DataFrame:
    """Diagnóstico exploratório de extrapolação entre distritos, apenas no período de desenvolvimento."""
    if development["area"].nunique() < splits:
        raise ValueError("Distritos insuficientes para GroupKFold")
    cv = GroupKFold(n_splits=splits)
    details = []
    for fold, (idx_train, idx_test) in enumerate(cv.split(development, groups=development["area"]), 1):
        train = development.iloc[idx_train]
        test = development.iloc[idx_test]
        if set(train["area"]) & set(test["area"]):
            raise AssertionError("Vazamento de distrito em GroupKFold")
        for family, spec in chosen.items():
            _, pred = _predict(spec, train, test)
            details.append({"modelo": family, "fold": fold, "n_distritos": test["area"].nunique(),
                            **metric_values(test[TARGET], pred)})
    return pd.DataFrame(details).sort_values(["modelo", "fold"]).reset_index(drop=True)


def bootstrap_by_district(frame: pd.DataFrame, predictions: np.ndarray, *,
                          repetitions: int = 1000, seed: int = RANDOM_SEED) -> dict[str, float]:
    """IC descritivo percentil por reamostragem de distritos inteiros (não por linhas)."""
    if repetitions < 10:
        raise ValueError("São necessárias pelo menos 10 reamostragens")
    groups = [np.flatnonzero(frame["area"].to_numpy() == area)
              for area in sorted(frame["area"].unique())]
    if len(groups) < 2:
        raise ValueError("São necessários pelo menos dois distritos para bootstrap")
    actual = frame[TARGET].to_numpy(dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    if actual.shape != predictions.shape:
        raise ValueError("Dimensões incompatíveis com a amostra")
    rng = np.random.default_rng(seed)
    distribution = []
    for _ in range(repetitions):
        selected = rng.integers(0, len(groups), len(groups))
        positions = np.concatenate([groups[int(i)] for i in selected])
        m = metric_values(actual[positions], predictions[positions])
        distribution.append((m["MAE"], m["RMSE"]))
    confidence = np.percentile(np.asarray(distribution), [2.5, 97.5], axis=0)
    return {"MAE_ic95_inferior": float(confidence[0, 0]),
            "MAE_ic95_superior": float(confidence[1, 0]),
            "RMSE_ic95_inferior": float(confidence[0, 1]),
            "RMSE_ic95_superior": float(confidence[1, 1])}


def error_diagnostics(test: pd.DataFrame, predicted: np.ndarray):
    """Predições auditáveis e erros agregados por distrito e ano."""
    frame = test[["area", "date", "year", TARGET]].copy()
    frame["predito"] = predicted
    frame["residuo"] = frame[TARGET] - frame["predito"]
    frame["erro_absoluto"] = frame["residuo"].abs()
    by_area = frame.groupby("area", as_index=False).agg(
        n=("residuo", "size"), MAE=("erro_absoluto", "mean"),
        vies_real_menos_previsto=("residuo", "mean"),
    ).sort_values("MAE", ascending=False)
    by_year = frame.groupby("year", as_index=False).agg(
        n=("residuo", "size"), MAE=("erro_absoluto", "mean"),
        vies_real_menos_previsto=("residuo", "mean"),
    ).sort_values("year")
    return frame, by_area, by_year


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_markdown_report(path: Path, validation: pd.DataFrame, test: pd.DataFrame,
                          champion: str, cv_detail: pd.DataFrame, uncertainty: dict,
                          by_area: pd.DataFrame, metadata: dict) -> None:
    """Resumo gerado dos resultados reais, nunca de valores escritos manualmente."""
    cv = cv_detail.groupby("modelo", as_index=False).agg(
        RMSE_medio=("RMSE", "mean"), RMSE_desvio=("RMSE", "std"),
        MAE_medio=("MAE", "mean"))
    sections = [
        "# Benchmark reproduzido — Housing in London",
        "",
        "> Estudo acadêmico com médias de distritos; **não recomendado para precificação de imóveis**.",
        "",
        "## Protocolo",
        "",
        "- Fonte e hashes SHA-256 dos dois CSVs: `configuracao.json`.",
        f"- Treino: até {TRAIN_LAST_YEAR}; validação: {TRAIN_LAST_YEAR + 1}–{VAL_LAST_YEAR}; teste: {VAL_LAST_YEAR + 1}–{TEST_LAST_YEAR}.",
        "- A escolha de hiperparâmetros e do modelo foi feita **somente na validação**.",
        "- Após selecionar, os modelos foram reajustados em treino+validação para teste.",
        "- Features do mesmo ano podem não estar disponíveis antecipadamente; isto é uma avaliação retrospectiva, não previsão ex ante.",
        "",
        "## Validação temporal (MAE e RMSE em £)",
        "",
        validation.round(3).to_markdown(index=False),
        "",
        "## Teste temporal final — uma consulta por protocolo (MAE e RMSE em £)",
        "",
        test.round(3).to_markdown(index=False),
        "",
        f"**Modelo selecionado exclusivamente por RMSE de validação:** {champion}.",
        "",
        "## Diagnóstico geográfico (GroupKFold em dados de desenvolvimento)",
        "",
        cv.round(3).to_markdown(index=False),
        "",
        "**Nota:** a validação geográfica é diagnóstica e usa candidatos selecionados no mesmo período de desenvolvimento; não fornece estimativa independente de seleção de hiperparâmetros.",
        "",
        "## Incerteza e erros",
        "",
        f"- Bootstrap de distritos (1000 reamostragens, IC percentil descritivo de 95%): MAE £{uncertainty['MAE_ic95_inferior']:,.0f}–£{uncertainty['MAE_ic95_superior']:,.0f}, RMSE £{uncertainty['RMSE_ic95_inferior']:,.0f}–£{uncertainty['RMSE_ic95_superior']:,.0f}.",
        "- Reamostragem por distrito considera dependências entre anos do mesmo distrito, mas não corrige todas as formas de viés.",
        "- Distritos com maiores MAEs no teste: " + ", ".join(
            f"{item.area} (£{item.MAE:,.0f})" for item in by_area.head(5).itertuples(index=False)) + ".",
        "",
        "## Limitações",
        "",
        "Este conjunto contém **médias históricas agregadas por distrito**, não imóveis individuais; preços e atributos anuais podem ser conhecidos apenas retrospectivamente. Comparações não incorporam atributos físicos do imóvel, cenário inflacionário, alterações estruturais ou calibração de intervalos preditivos para imóveis. Os ICs são descritivos; não garantem cobertura futura. Nenhum modelo deve ser usado em avaliação comercial ou decisões financeiras.",
        "",
        "## Reprodutibilidade",
        "",
        "- Parâmetros de execução e hashes: `configuracao.json`.",
        f"- Linhas: {metadata['n_total']}; períodos: {metadata['n_treino']}/{metadata['n_validacao']}/{metadata['n_teste']}.",
        "- Resultados detalhados nos CSVs e figuras no mesmo diretório.",
    ]
    path.write_text("\n".join(sections) + "\n", encoding="utf-8")
