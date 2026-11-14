"""Registro de modelos e pipelines; não escolhe hiperparâmetros pelo teste.

INTEGRAÇÃO FUTURA: adicione novas famílias somente a candidates() e preserve
imputação/padronização dentro do Pipeline para evitar vazamento estatístico.
"""
from __future__ import annotations

from dataclasses import dataclass
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if __package__:
    from .data_protocol import SIMPLE, MULTIPLE, RANDOM_SEED
else:
    from data_protocol import SIMPLE, MULTIPLE, RANDOM_SEED


@dataclass(frozen=True)
class ModelSpec:
    name: str
    features: tuple[str, ...]
    algorithm: str
    parameter: float | int | None = None

    def build(self):
        if self.algorithm == "baseline":
            return DummyRegressor(strategy="mean")
        if self.algorithm == "forest":
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(
                    n_estimators=200, min_samples_leaf=int(self.parameter),
                    random_state=RANDOM_SEED, n_jobs=1,
                )),
            ])
        if self.algorithm in ("linear", "ridge"):
            regressor = LinearRegression() if self.algorithm == "linear" else Ridge(alpha=float(self.parameter))
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", regressor),
            ])
        raise ValueError(f"Algoritmo desconhecido: {self.algorithm}")

    def description(self) -> str:
        return f"{self.algorithm}; parâmetro={self.parameter}" if self.parameter is not None else self.algorithm


def candidates() -> dict[str, list[ModelSpec]]:
    return {
        "Baseline": [ModelSpec("Baseline", SIMPLE, "baseline")],
        "Linear simples": [ModelSpec("Linear simples", SIMPLE, "linear")],
        "Linear múltipla": [ModelSpec("Linear múltipla", MULTIPLE, "linear")],
        "Ridge": [ModelSpec("Ridge", MULTIPLE, "ridge", a) for a in (0.1, 1.0, 10.0, 100.0, 1000.0)],
        "Random Forest": [ModelSpec("Random Forest", MULTIPLE, "forest", leaf) for leaf in (2, 5, 10)],
    }
