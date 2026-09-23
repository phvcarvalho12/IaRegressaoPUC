"""Classificador e funcoes de consulta da regressao logistica."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LOGISTIC_FEATURES = ["age_mid", "age_sq", "log_gdp_pc", "log_pop", "log_area"]


def make_classifier():
    """Cria o pipeline padronizado de regressao logistica."""
    return make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)
    )


def evaluate_grouped(data: pd.DataFrame, features: list[str] = LOGISTIC_FEATURES) -> dict:
    """Avalia o classificador em paises inteiros separados para o teste."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(
        splitter.split(data, data["happy"], groups=data["iso3"])
    )
    model = make_classifier().fit(data.iloc[train_idx][features], data.iloc[train_idx]["happy"])
    probabilities = model.predict_proba(data.iloc[test_idx][features])[:, 1]
    predicted = (probabilities >= 0.5).astype(int)
    actual = data.iloc[test_idx]["happy"]
    return {
        "model": model,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "accuracy": accuracy_score(actual, predicted),
        "auc": roc_auc_score(actual, probabilities),
        "confusion_matrix": confusion_matrix(actual, predicted),
    }


def fit_final_model(data: pd.DataFrame, features: list[str] = LOGISTIC_FEATURES):
    """Treina o classificador final com todas as linhas."""
    return make_classifier().fit(data[features], data["happy"])


def check_happiness(country: str, age: float, model, country_lookup: pd.DataFrame, to_iso3, features: list[str] = LOGISTIC_FEATURES) -> dict:
    """Retorna a probabilidade e a classe prevista para um pais e idade."""
    iso3 = to_iso3(pd.Series([country])).iloc[0]
    if pd.isna(iso3) or iso3 not in country_lookup.index:
        raise ValueError(f"Pais desconhecido ou sem dados do Banco Mundial: {country}")
    info = country_lookup.loc[iso3]
    row = pd.DataFrame([{
        "age_mid": float(age),
        "age_sq": float(age) ** 2,
        "log_gdp_pc": info["log_gdp_pc"],
        "log_pop": info["log_pop"],
        "log_area": info["log_area"],
    }])[features]
    probability = float(model.predict_proba(row)[0, 1])
    return {
        "country": info["country"],
        "age": age,
        "p_happy": round(probability, 3),
        "predicted": "happy" if probability > 0.5 else "not happy",
    }
