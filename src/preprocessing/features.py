"""
Seleção de variáveis para a modelagem.

Tudo o que entra no modelo vem do ano-base (t-1). O alvo é o atingimento da
meta no ano seguinte (t). Os mesmos nomes de coluna servem para treinar
(2023 -> 2024) e para pontuar o futuro (2024 -> 2025), por isso as colunas
são renomeadas de forma genérica ("taxa_base", "participacao_base" etc.).
"""

from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
CAMINHO_ABT = RAIZ / "data" / "processed" / "abt_alfabetizacao_municipios.csv"

ALVO = "target_atingiu_meta_2024"
IDENTIFICACAO = ["id_municipio", "nome_geografia", "sigla_uf"]
# A meta de 2024 é função quase exata da taxa de 2023 (Spearman 0,997), então o
# esforço (meta - taxa) não entra no modelo: seria redundante com taxa_base.
# A coluna continua em monta_features porque é usada na análise de metas futuras.
NUMERICAS = [
    "taxa_base",
    "media_portugues_base",
    "participacao_base",
    "log_populacao",
    "log_pib_per_capita",
]
CATEGORICAS = ["macrorregiao"]


def carrega_abt(caminho: Path = CAMINHO_ABT) -> pd.DataFrame:
    return pd.read_csv(caminho, dtype={"id_municipio": str})


def monta_features(abt: pd.DataFrame, ano_base: int = 2023, ano_alvo: int = 2024) -> pd.DataFrame:
    """Devolve as variáveis do modelo usando só informação do ano-base e a meta do ano-alvo."""
    if ano_base >= ano_alvo:
        raise ValueError("o ano-base precisa ser anterior ao ano-alvo")

    X = abt[IDENTIFICACAO + CATEGORICAS + ["log_populacao", "log_pib_per_capita"]].copy()
    X["taxa_base"] = abt[f"taxa_alfabetizacao_{ano_base}_pct"]
    X["media_portugues_base"] = abt[f"media_portugues_{ano_base}"]
    X["participacao_base"] = abt[f"participacao_{ano_base}_pct"]
    X["esforco_meta"] = abt[f"valor_meta_{ano_alvo}_pct"] - X["taxa_base"]
    return X
