"""Verificações contratuais do benchmark: dados sintéticos, sem Kaggle ou rede."""
import numpy as np
import pandas as pd
import pytest

from scripts.london_benchmark import (
    TARGET, bootstrap_by_district, candidates, choose_from_validation,
    error_diagnostics, evaluate_final, grouped_cv, load_housing_data,
    metric_values, temporal_split,
)


@pytest.fixture
def synthetic_data():
    records = []
    for group in range(10):
        for year in range(2009, 2020):
            records.append({
                "area": f"borough_{group}", "date": pd.Timestamp(year, 12, 1),
                "year": year, TARGET: 150000 + group * 16000 + (year - 2009) * 9000,
                "median_salary": 22000 + group * 700 + (year - 2009) * 350,
                "number_of_jobs": 50000 + group * 1000,
                "recycling_pct": 25 + group,
            })
    return pd.DataFrame(records)


def test_temporal_order_and_exhaustive_partition(synthetic_data):
    train, val, test = temporal_split(synthetic_data)
    assert train["year"].max() == 2013
    assert set(val["year"]) == {2014, 2015, 2016}
    assert set(test["year"]) == {2017, 2018, 2019}
    assert len(train) + len(val) + len(test) == len(synthetic_data)
    assert train["date"].max() < val["date"].min() < test["date"].min()


def test_rejects_future_year_to_avoid_silent_protocol_drift(synthetic_data):
    additional = synthetic_data.iloc[[0]].copy()
    additional["year"] = 2020
    additional["date"] = pd.Timestamp(2020, 12, 1)
    with pytest.raises(ValueError, match="Novos anos"):
        temporal_split(pd.concat([synthetic_data, additional]))


def test_rejects_missing_test_period(synthetic_data):
    with pytest.raises(ValueError, match="observações suficientes"):
        temporal_split(synthetic_data[synthetic_data["year"] <= 2016])


def test_candidate_pipeline_imputes_missing_values(synthetic_data):
    data = synthetic_data.copy()
    data.loc[data.index[:5], "median_salary"] = np.nan
    estimator = candidates()["Ridge"][0].build()
    estimator.fit(data[["median_salary", "number_of_jobs", "recycling_pct"]], data[TARGET])
    prediction = estimator.predict(data[["median_salary", "number_of_jobs", "recycling_pct"]])
    assert np.isfinite(prediction).all()


def test_validation_and_refit_still_run_with_synthetic_data(synthetic_data):
    train, val, test = temporal_split(synthetic_data)
    chosen, scores, winner = choose_from_validation(train, val)
    assert len(scores) == 5 and winner in chosen
    results, predictions, _ = evaluate_final(chosen, pd.concat([train, val]), test)
    assert len(results) == 5 and set(predictions) == set(chosen)


def test_group_cv_never_reuses_borough(synthetic_data):
    dev = synthetic_data[synthetic_data["year"] <= 2016]
    candidate = candidates()["Linear múltipla"][0]
    folds = grouped_cv({"Linear múltipla": candidate}, dev, splits=5)
    assert len(folds) == 5 and (folds["n_distritos"] == 2).all()


def test_group_bootstrap_is_reproducible(synthetic_data):
    test = synthetic_data[synthetic_data["year"] >= 2017].copy()
    prediction = test[TARGET].to_numpy() + 1000
    one = bootstrap_by_district(test, prediction, repetitions=50)
    two = bootstrap_by_district(test, prediction, repetitions=50)
    assert one == two
    assert one["MAE_ic95_inferior"] <= one["MAE_ic95_superior"]


def test_rejects_invalid_bootstrap_sample(synthetic_data):
    test = synthetic_data[synthetic_data["year"] >= 2017]
    with pytest.raises(ValueError, match="Dimensões incompatíveis"):
        bootstrap_by_district(test, np.ones(5), repetitions=50)


def test_metrics_invalid_shapes():
    with pytest.raises(ValueError, match="Dimensões"):
        metric_values([1, 2], [1])
    with pytest.raises(ValueError, match="NaN"):
        metric_values([1, 2], [1, float("nan")])


def test_year_and_borough_diagnostics(synthetic_data):
    test = synthetic_data[synthetic_data["year"] >= 2017]
    predictions, by_area, by_year = error_diagnostics(
        test, test[TARGET].to_numpy() + 1000)
    assert len(by_area) == 10 and len(by_year) == 3
    assert np.allclose(by_area["MAE"], 1000)
    assert np.allclose(predictions["residuo"], -1000)


def test_dataset_merge_rejects_duplicate_keys(tmp_path):
    monthly = pd.DataFrame({
        "area": ["a", "a"], "date": ["2019-12-01", "2019-12-01"],
        "borough_flag": [1, 1], TARGET: [120000, 130000]})
    yearly = pd.DataFrame({
        "area": ["a"], "date": ["2019-12-01"], "borough_flag": [1],
        "median_salary": [35000], "number_of_jobs": [10000], "recycling_pct": [30]})
    monthly.to_csv(tmp_path / "housing_in_london_monthly_variables.csv", index=False)
    yearly.to_csv(tmp_path / "housing_in_london_yearly_variables.csv", index=False)
    with pytest.raises(ValueError, match="duplicadas"):
        load_housing_data(tmp_path)


def test_dataset_merge_does_not_inflate_rows(tmp_path):
    monthly = pd.DataFrame({
        "area": ["a"], "date": ["2019-12-01"], "borough_flag": [1], TARGET: [120000]})
    yearly = pd.DataFrame({
        "area": ["a"], "date": ["2019-12-01"], "borough_flag": [1],
        "median_salary": ["#"], "number_of_jobs": [10000], "recycling_pct": [30]})
    monthly.to_csv(tmp_path / "housing_in_london_monthly_variables.csv", index=False)
    yearly.to_csv(tmp_path / "housing_in_london_yearly_variables.csv", index=False)
    df = load_housing_data(tmp_path)
    assert len(df) == 1 and pd.isna(df.loc[0, "median_salary"])
