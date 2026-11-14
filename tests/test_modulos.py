"""Testes de compatibilidade após separação de dados e registro de modelos."""
from scripts import data_protocol, london_benchmark, model_registry


def test_public_api_compatibility():
    assert london_benchmark.temporal_split is data_protocol.temporal_split
    assert london_benchmark.load_housing_data is data_protocol.load_housing_data
    assert london_benchmark.ModelSpec is model_registry.ModelSpec
    assert london_benchmark.candidates is model_registry.candidates


def test_models_use_expected_features_and_seed():
    models = model_registry.candidates()
    assert set(models) == {"Baseline", "Linear simples", "Linear múltipla", "Ridge", "Random Forest"}
    assert model_registry.RANDOM_SEED == data_protocol.RANDOM_SEED
    assert all(spec.features for group in models.values() for spec in group)
    assert data_protocol.TRAIN_LAST_YEAR < data_protocol.VAL_LAST_YEAR < data_protocol.TEST_LAST_YEAR
