"""
Construção da Analytical Base Table (ABT) da Fase 3.

Cada linha é um município (rede municipal, 2º ano do EF). As features vêm de
2023 e o target (atingiu a meta de 2024?) vem de 2024, então nada do ano-alvo
entra como preditor. As colunas de 2024 ficam na base só para auditoria e
para a EDA; a lista COLUNAS_ANO_ALVO existe para a etapa de modelagem
descartá-las de forma explícita.

Decisões que vieram da exploração dos dados (ver notebooks/01_eda_hipoteses):
  * só a rede MUNICIPAL tem meta em ftc_meta_alfabetizacao; as demais redes
    herdariam a meta municipal por engano se o join fosse só por município;
  * a distribuição por níveis (nivel_0..8) não existe em 2023, então nenhum
    indicador de nível entra como feature;
  * na fct_cobertura_avaliacao, presentes, cadernos válidos e taxa de presença
    estão zerados, e os 621 municípios de SP não aparecem na tabela;
  * o PIB da SIDRA (tabela 5938, variável 37) é total em mil R$, e o PIB per
    capita é calculado aqui dividindo pela população.

Uso (a partir da raiz do projeto):
    python src/preprocessing/build_dataset.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import requests

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
EXTERNAL_DIR = DATA_DIR / "external"
PROCESSED_DIR = DATA_DIR / "processed"

ANO_FEATURES = 2023
ANO_TARGET = 2024
REDE_ALVO = "MUNICIPAL"

SIDRA_PIB = "https://apisidra.ibge.gov.br/values/t/5938/n6/all/v/37/p/last%201?formato=json"
SIDRA_POP = "https://apisidra.ibge.gov.br/values/t/6579/n6/all/v/9324/p/last%201?formato=json"

# Colunas que só existem depois do ano-alvo: nunca podem ser features.
COLUNAS_ANO_ALVO = [
    "taxa_alfabetizacao_2024_pct",
    "media_portugues_2024",
    "participacao_2024_pct",
    "target_atingiu_meta_2024",
]

MACRORREGIAO = {"1": "Norte", "2": "Nordeste", "3": "Sudeste", "4": "Sul", "5": "Centro-Oeste"}


def localiza_csv(nome: str) -> Path:
    achados = list(DATA_DIR.rglob(nome))
    if not achados:
        raise FileNotFoundError(
            f"'{nome}' não encontrado em {DATA_DIR}. Copie as 6 tabelas da camada Gold para data/gold/."
        )
    return achados[0]


def le_gold() -> dict:
    ids = {"id_municipio": str, "codigo_uf_ibge": str}
    return {
        "res": pd.read_csv(localiza_csv("fct_resultado_alfabetizacao.csv")),
        "cob": pd.read_csv(localiza_csv("fct_cobertura_avaliacao.csv")),
        "meta": pd.read_csv(localiza_csv("ftc_meta_alfabetizacao.csv")),
        "geo": pd.read_csv(localiza_csv("dim_geografia.csv"), dtype=ids),
        "rede": pd.read_csv(localiza_csv("dim_rede.csv")),
        "serie": pd.read_csv(localiza_csv("dim_serie.csv")),
    }


def confere_dimensoes(rede: pd.DataFrame, serie: pd.DataFrame) -> None:
    """A base só faz sentido se o rótulo da rede está confirmado e há uma única série."""
    linha = rede.set_index("rede_key").loc[REDE_ALVO]
    if not linha["rotulo_confirmado"]:
        raise ValueError(f"O rótulo da rede {REDE_ALVO} não está confirmado em dim_rede.")
    if len(serie) != 1:
        raise ValueError(f"Esperava uma série em dim_serie, encontrei {len(serie)}.")


def consulta_sidra(url: str, coluna_valor: str, coluna_ano: str) -> pd.DataFrame:
    resposta = requests.get(url, timeout=60)
    resposta.raise_for_status()
    dados = resposta.json()
    cabecalho, linhas = dados[0], pd.DataFrame(dados[1:])
    chave_ano = next((k for k, v in cabecalho.items() if v == "Ano (Código)"), None)

    saida = pd.DataFrame(
        {
            "id_municipio": linhas["D1C"].astype(str).str.strip(),
            coluna_valor: pd.to_numeric(linhas["V"], errors="coerce"),
        }
    )
    saida[coluna_ano] = pd.to_numeric(linhas[chave_ano], errors="coerce") if chave_ano else np.nan
    return saida


def carrega_ibge() -> pd.DataFrame:
    """PIB total (mil R$) e população estimada por município, com cache em data/external."""
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    cache = EXTERNAL_DIR / "ibge_socioeconomico_municipios.csv"

    if cache.exists():
        ibge = pd.read_csv(cache, dtype={"id_municipio": str})
        # a primeira versão do cache gravava o PIB total com o nome 'pib_per_capita_brl'
        return ibge.rename(columns={"pib_per_capita_brl": "pib_total_mil_brl"})

    pib = consulta_sidra(SIDRA_PIB, "pib_total_mil_brl", "ano_pib")
    pop = consulta_sidra(SIDRA_POP, "populacao_estimada", "ano_populacao")
    ibge = pib.merge(pop, on="id_municipio", how="outer")
    ibge.to_csv(cache, index=False)
    return ibge


def monta_abt() -> pd.DataFrame:
    gold = le_gold()
    confere_dimensoes(gold["rede"], gold["serie"])

    # --- resultados: features em 2023, target em 2024 --------------------------------
    res = gold["res"]
    res = res[
        (res["rede_key"] == REDE_ALVO)
        & res["geo_key"].str.startswith("MUN:")
        & res["ano"].isin([ANO_FEATURES, ANO_TARGET])
    ]
    assert not res.duplicated(["ano", "geo_key"]).any(), "chave (ano, geo_key) duplicada em resultados"

    colunas = {
        "taxa_alfabetizacao_pct": "taxa_alfabetizacao",
        "media_portugues": "media_portugues",
        "percentual_participacao_pct": "participacao",
    }
    base = res[res["ano"] == ANO_FEATURES][["geo_key", *colunas]].rename(
        columns={k: f"{v}_{ANO_FEATURES}" + ("_pct" if k.endswith("_pct") else "") for k, v in colunas.items()}
    )
    alvo = res[res["ano"] == ANO_TARGET][["geo_key", *colunas]].rename(
        columns={k: f"{v}_{ANO_TARGET}" + ("_pct" if k.endswith("_pct") else "") for k, v in colunas.items()}
    )
    abt = alvo.merge(base, on="geo_key", how="inner")

    # --- tamanho da rede avaliada (a tabela não cobre SP) ----------------------------
    cob = gold["cob"]
    cob = cob[(cob["rede_key"] == REDE_ALVO) & (cob["ano"] == ANO_FEATURES)]
    cob = cob[["geo_key", "soma_peso_aluno"]].rename(columns={"soma_peso_aluno": "soma_peso_aluno_2023"})
    abt = abt.merge(cob, on="geo_key", how="left")

    # --- metas: usamos o que já era conhecido em 2023 (ano_referencia = 2023) --------
    meta = gold["meta"]
    meta = meta[
        (meta["ano_referencia"] == ANO_FEATURES)
        & (meta["rede_key"] == REDE_ALVO)
        & meta["geo_key"].str.startswith("MUN:")
    ]
    meta = (
        meta.pivot_table(index="geo_key", columns="ano_meta", values="valor_meta_pct", aggfunc="first")
        .rename(columns=lambda ano: f"valor_meta_{ano}_pct")
        .reset_index()
    )
    abt = abt.merge(meta, on="geo_key", how="left")

    # --- geografia -------------------------------------------------------------------
    geo = gold["geo"]
    geo = geo[geo["nivel_geografico"] == "MUNICIPIO"].copy()
    geo["macrorregiao"] = geo["codigo_uf_ibge"].str[0].map(MACRORREGIAO).fillna("Desconhecida")
    abt = abt.merge(
        geo[["geo_key", "id_municipio", "nome_geografia", "sigla_uf", "macrorregiao"]],
        on="geo_key",
        how="left",
    )

    # --- IBGE ------------------------------------------------------------------------
    abt = abt.merge(carrega_ibge(), on="id_municipio", how="left")
    abt["pib_per_capita_brl"] = abt["pib_total_mil_brl"] * 1000 / abt["populacao_estimada"]
    abt["log_pib_per_capita"] = np.log1p(abt["pib_per_capita_brl"])
    abt["log_populacao"] = np.log1p(abt["populacao_estimada"])

    # --- atributos derivados e target ------------------------------------------------
    abt["esforco_meta_2024_pp"] = abt["valor_meta_2024_pct"] - abt["taxa_alfabetizacao_2023_pct"]
    abt["acima_da_meta_em_2023"] = (abt["taxa_alfabetizacao_2023_pct"] >= abt["valor_meta_2024_pct"]).astype(int)

    antes = len(abt)
    abt = abt.dropna(subset=["taxa_alfabetizacao_2024_pct", "taxa_alfabetizacao_2023_pct", "valor_meta_2024_pct"])
    abt["target_atingiu_meta_2024"] = (abt["taxa_alfabetizacao_2024_pct"] >= abt["valor_meta_2024_pct"]).astype(int)
    print(f"Municípios com resultado em 2023 e 2024: {antes:,}; com meta e taxas completas: {len(abt):,}")

    return abt.sort_values("id_municipio").reset_index(drop=True)


def main() -> None:
    abt = monta_abt()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    saida = PROCESSED_DIR / "abt_alfabetizacao_municipios.csv"
    abt.to_csv(saida, index=False)

    print(f"ABT salva em {saida} ({abt.shape[0]:,} linhas, {abt.shape[1]} colunas)")
    print("Proporção do target:", abt["target_atingiu_meta_2024"].value_counts(normalize=True).round(3).to_dict())
    faltantes = abt[["pib_per_capita_brl", "populacao_estimada", "soma_peso_aluno_2023"]].isna().sum()
    print("Nulos:", faltantes.to_dict())


if __name__ == "__main__":
    main()
