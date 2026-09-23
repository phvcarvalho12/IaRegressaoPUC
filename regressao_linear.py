"""Modelos e avaliacao da regressao linear do laboratorio."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict

LINEAR_MODELS = {
    "M1 idade": ["age_mid"],
    "M2 idade + idade2": ["age_mid", "age_sq"],
    "M3 idade + controles": [
        "age_mid", "age_sq", "log_gdp_pc", "log_pop", "log_area"
    ],
}


def cv_scores(data: pd.DataFrame, features: list[str], target: str = "ladder_mean", k: int = 5):
    """Calcula R2, RMSE e previsoes usando grupos de pais no cross-validation."""
    groups = data["iso3"]
    cv = GroupKFold(n_splits=min(k, groups.nunique()))
    predictions = cross_val_predict(
        LinearRegression(), data[features], data[target], groups=groups, cv=cv
    )
    return (
        r2_score(data[target], predictions),
        mean_squared_error(data[target], predictions) ** 0.5,
        predictions,
    )


def compare_models(data: pd.DataFrame, target: str = "ladder_mean") -> tuple[pd.DataFrame, dict]:
    """Compara os tres modelos lineares e retorna a tabela e as previsoes."""
    rows = []
    predictions = {}
    for name, features in LINEAR_MODELS.items():
        r2, rmse, predicted = cv_scores(data, features, target=target)
        rows.append({"modelo": name, "R2": r2, "RMSE": rmse})
        predictions[name] = predicted
    return pd.DataFrame(rows), predictions


def fit_full_model(data: pd.DataFrame) -> LinearRegression:
    """Ajusta o modelo M3 em todas as observacoes."""
    features = LINEAR_MODELS["M3 idade + controles"]
    return LinearRegression().fit(data[features], data["ladder_mean"])


def minimum_age(model: LinearRegression) -> float:
    """Estima o ponto minimo da parabola de idade do modelo M3."""
    age_coefficient = model.coef_[0]
    age_squared_coefficient = model.coef_[1]
    return -age_coefficient / (2 * age_squared_coefficient)
