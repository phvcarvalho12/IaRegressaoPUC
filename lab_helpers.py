"""Funcoes reutilizaveis do laboratorio de felicidade por idade e pais."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

OWID_AGE_URL = (
    "https://ourworldindata.org/grapher/"
    "cantril-ladder-age-groups.csv?v=1&csvType=full&useColumnShortNames=true"
)
OWID_AGE_LOCAL = "cantril-ladder-age-groups.csv"
WHR_XLSX = "https://files.worldhappiness.report/WHR26_Data_Figure_2.1.xlsx"
AGE_MID = {
    "Up to 29 years": 22,
    "30-44 years": 37,
    "45-59 years": 52,
    "60+ years": 70,
}
OWID_AGE_COLUMNS = {
    "cantril_ladder_score__age_group_up_to_29_years": "Up to 29 years",
    "cantril_ladder_score__age_group_30_44_years": "30-44 years",
    "cantril_ladder_score__age_group_45_59_years": "45-59 years",
    "cantril_ladder_score__age_group_60plus_years": "60+ years",
}
WB_INDICATORS = {
    "gdp_pc": "NY.GDP.PCAP.PP.KD",
    "pop": "SP.POP.TOTL",
    "area_km2": "AG.LND.TOTL.K2",
}


def _get_csv(url: str, local_name: str | None = None) -> pd.DataFrame:
    """Baixa um CSV; usa uma copia local quando ela existe."""
    local = Path(local_name) if local_name else None
    if local and local.exists():
        return pd.read_csv(local)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return pd.read_csv(BytesIO(response.content))


def load_owid_age(url: str = OWID_AGE_URL, local_name: str = OWID_AGE_LOCAL) -> pd.DataFrame:
    """Retorna uma linha por pais, faixa etaria e ano."""
    raw = _get_csv(url, local_name=local_name)
    raw = raw.rename(columns={"entity": "Entity", "code": "Code", "year": "Year"})
    raw = raw.rename(columns=OWID_AGE_COLUMNS)
    expected = ["Entity", "Code", "Year", *AGE_MID.keys()]
    missing = [column for column in expected if column not in raw.columns]
    if missing:
        raise ValueError(f"CSV do OWID sem colunas esperadas: {missing}")
    age = raw.melt(
        id_vars=["Entity", "Code", "Year"],
        value_vars=list(AGE_MID),
        var_name="age_group",
        value_name="ladder_mean",
    ).rename(columns={"Entity": "country", "Code": "iso3", "Year": "year"})
    age = age.dropna(subset=["iso3", "ladder_mean"]).copy()
    age["iso3"] = age["iso3"].str.upper()
    age["window"] = "2021-2023"
    return age[["country", "iso3", "year", "age_group", "ladder_mean", "window"]]


def to_iso3(names: Iterable[str] | pd.Series) -> pd.Series:
    """Converte nomes de pais para ISO3 usando country_converter."""
    import country_converter as coco

    series = pd.Series(names)
    converted = coco.convert(series.tolist(), to="ISO3", not_found=None)
    return pd.Series(converted, index=series.index).replace({"not found": np.nan})


def fetch_wb(indicator: str, start_year: int = 2019, end_year: int = 2023) -> pd.Series:
    """Busca o ultimo valor disponivel por ISO3 na janela do Banco Mundial."""
    url = f"https://api.worldbank.org/v2/country/all/indicator/{indicator}"
    params = {"format": "json", "per_page": 20000, "date": f"{start_year}:{end_year}"}
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or len(payload) < 2:
        raise ValueError(f"Resposta inesperada do Banco Mundial para {indicator}")
    rows = [row for row in payload[1] if row.get("value") is not None]
    frame = pd.DataFrame(rows)
    values = frame.sort_values("date").drop_duplicates("countryiso3code", keep="last")
    return values.set_index("countryiso3code")["value"].rename(indicator)


def build_country_table() -> pd.DataFrame:
    """Monta indicadores nacionais e remove agregados sem ISO3."""
    columns = []
    for name, indicator in WB_INDICATORS.items():
        values = fetch_wb(indicator).rename(name)
        values.index.name = "iso3"
        columns.append(values)
    countries = pd.concat(columns, axis=1).reset_index()
    countries = countries[countries["iso3"].str.len() == 3].copy()
    return countries.dropna(subset=list(WB_INDICATORS))


def make_synthetic_age_data(country_table: pd.DataFrame) -> pd.DataFrame:
    """Gera dados artificiais apenas para testar o pipeline offline."""
    rng = np.random.default_rng(42)
    rows = []
    for _, country in country_table.iterrows():
        for age_group, age_mid in AGE_MID.items():
            baseline = 4.5 + 0.35 * np.log(country["gdp_pc"])
            value = baseline + 0.0008 * (age_mid - 45) ** 2 + rng.normal(0, 0.25)
            rows.append({
                "country": country["iso3"], "iso3": country["iso3"], "year": 2023,
                "age_group": age_group, "ladder_mean": np.clip(value, 0, 10),
                "window": "synthetic", "synthetic": True,
            })
    return pd.DataFrame(rows)
