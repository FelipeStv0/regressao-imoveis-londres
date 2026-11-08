"""Testes offline: não acessam rede ou datasets de terceiros."""

from pathlib import Path
from types import SimpleNamespace
import sys

import pandas as pd
import pytest

from scripts.prepare_data import DATASET_HANDLE, REQUIRED_FILES, ensure_data
from scripts.check_results import validate_metrics


def make_csvs(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_FILES:
        (directory / name).write_text("area,date,average_price\narea,2019-12-01,100000\n", encoding="utf-8")


def test_uses_local_files_without_calling_network(tmp_path, monkeypatch):
    make_csvs(tmp_path)
    monkeypatch.setitem(sys.modules, "kagglehub", SimpleNamespace(dataset_download=lambda *a, **k: pytest.fail("rede indevida")))
    assert ensure_data(tmp_path) == tmp_path.resolve()


def test_missing_data_fails_with_reproduction_guidance(tmp_path):
    with pytest.raises(FileNotFoundError, match="--download"):
        ensure_data(tmp_path)


def test_download_only_after_explicit_authorization(tmp_path, monkeypatch):
    calls = []
    def fake_download(handle, *, output_dir):
        calls.append((handle, output_dir))
        make_csvs(Path(output_dir))
        return output_dir
    monkeypatch.setitem(sys.modules, "kagglehub", SimpleNamespace(dataset_download=fake_download))
    assert ensure_data(tmp_path, allow_download=True) == tmp_path.resolve()
    assert calls == [(DATASET_HANDLE, str(tmp_path.resolve()))]


def test_incomplete_download_fails(tmp_path, monkeypatch):
    def fake_download(handle, *, output_dir):
        (Path(output_dir) / REQUIRED_FILES[0]).write_text("only one", encoding="utf-8")
        return output_dir
    monkeypatch.setitem(sys.modules, "kagglehub", SimpleNamespace(dataset_download=fake_download))
    with pytest.raises(FileNotFoundError, match="incompleto"):
        ensure_data(tmp_path, allow_download=True)


def test_validates_exported_metrics(tmp_path):
    file = tmp_path / "metricas.csv"
    pd.DataFrame({"modelo": ["Baseline", "Linear simples", "Linear múltipla", "Polinomial grau 2"],
                  "MAE teste": [10, 8, 9, 7], "RMSE teste": [15, 12, 13, 11],
                  "R² teste": [-0.1, 0.1, 0.05, 0.2]}).to_csv(file, index=False)
    assert len(validate_metrics(file)) == 4


def test_rejects_invalid_metric(tmp_path):
    file = tmp_path / "metricas.csv"
    pd.DataFrame({"modelo": ["Baseline", "Linear simples", "Linear múltipla", "Polinomial grau 2"],
                  "MAE teste": [10, 8, 9, 7], "RMSE teste": [5, 12, 13, 11],
                  "R² teste": [-0.1, 0.1, 0.05, 0.2]}).to_csv(file, index=False)
    with pytest.raises(ValueError, match="MAE e RMSE"):
        validate_metrics(file)
