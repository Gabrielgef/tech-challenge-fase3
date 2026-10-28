"""
Módulo de construção da Analytical Base Table (ABT) para a Fase 3.
Consome as 6 tabelas da camada Gold (3 Fatos + 3 Dimensões da Fase 2),
previne Data Leakage via lag temporal (2023 -> 2024) e enriquece a base
com indicadores socioeconómicos e demográficos da API do IBGE (PIB e população estimada).
"""

from pathlib import Path
import warnings
import requests
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Caminhos base do projeto (padrão CCDS)
PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
EXTERNAL_DIR = DATA_DIR / "external"
PROCESSED_DIR = DATA_DIR / "processed"


def localiza_csv(nome_arquivo: str) -> Path:
    """Procura recursivamente o ficheiro CSV dentro da pasta data/."""
    matches = list(DATA_DIR.rglob(nome_arquivo))
    if not matches:
        raise FileNotFoundError(
            f"Ficheiro '{nome_arquivo}' não encontrado dentro de {DATA_DIR}. "
            "Verifique se copiou as 6 bases da camada Gold para data/gold/."
        )
    return matches[0]


def consulta_api_sidra(url: str, nomes_colunas: str) -> pd.DataFrame:
    """Consulta a API SIDRA do IBGE e devolve um DataFrame por id_municipio."""
    try:
        response = requests.get(url, timeout=45)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data[1:])
        df = df[["D1C", "V"]].rename(columns={"D1C": "id_municipio", "V": nomes_colunas})
        df["id_municipio"] = df["id_municipio"].astype(str).str.strip()
        df[nomes_colunas] = pd.to_numeric(df[nomes_colunas], errors="coerce")
        return df
    except Exception as exc:
        print(f"[AVISO] Não foi possível obter {nomes_colunas} via API IBGE: {exc}")
        return pd.DataFrame(columns=["id_municipio", nomes_colunas])


def dados_ibge() -> pd.DataFrame:
    """
    Obtém PIB per capita (Tabela 5938 / Var 37) e População (Tabela 6579 / Var 9324)
    dos municípios brasileiros via API do IBGE, guardando cache em data/external/.
    """
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = EXTERNAL_DIR / "ibge_socioeconomico_municipios.csv"

    if cache_path.exists():
        print(f"[INFO] A carregar dados socioeconómicos do cache: {cache_path}")
        return pd.read_csv(cache_path, dtype={"id_municipio": str})

    print("[INFO] A consultar a API oficial do IBGE (SIDRA) para enriquecimento socioeconómico...")
    url_pib = "https://apisidra.ibge.gov.br/values/t/5938/n6/all/v/37/p/last%201?formato=json"
    df_pib = consulta_api_sidra(url_pib, "pib_per_capita_brl")

    url_pop = "https://apisidra.ibge.gov.br/values/t/6579/n6/all/v/9324/p/last%201?formato=json"
    df_pop = consulta_api_sidra(url_pop, "populacao_estimada")

    if not df_pib.empty and not df_pop.empty:
        df_ibge = pd.merge(df_pib, df_pop, on="id_municipio", how="outer")
        df_ibge.to_csv(cache_path, index=False)
        print(f"[OK] Dados do IBGE guardados em {cache_path} ({len(df_ibge)} municípios).")
        return df_ibge

    return pd.DataFrame(columns=["id_municipio", "pib_per_capita_brl", "populacao_estimada"])


def macro_regiao(codigo_uf: str) -> str:
    """Mapeia o primeiro dígito do código IBGE para a Macrorregião do Brasil."""
    prefix = str(codigo_uf).strip()
    mapping = {
        "1": "Norte",
        "2": "Nordeste",
        "3": "Sudeste",
        "4": "Sul",
        "5": "Centro-Oeste",
    }
    return mapping.get(prefix[0], "Desconhecida") if prefix else "Desconhecida"


def build_analytical_dataset() -> pd.DataFrame:
    """Constrói a tabela final para o eda e modelagem."""
    print("[1/5] A localizar e carregar as 6 tabelas da camada Gold (3 Fatos + 3 Dimensões)...")
    # 3 Tabelas Fato
    df_res = pd.read_csv(localiza_csv("fct_resultado_alfabetizacao.csv"))
    df_cob = pd.read_csv(localiza_csv("fct_cobertura_avaliacao.csv"))
    df_meta = pd.read_csv(localiza_csv("ftc_meta_alfabetizacao.csv"))

    # 3 Tabelas Dimensão
    df_geo = pd.read_csv(
        localiza_csv("dim_geografia.csv"),
        dtype={"id_municipio": str, "codigo_uf_ibge": str},
    )
    df_rede = pd.read_csv(localiza_csv("dim_rede.csv"))
    df_serie = pd.read_csv(localiza_csv("dim_serie.csv"))

    # Preparar dim_geografia (nível MUNICIPAL)
    df_geo_mun = df_geo[df_geo["nivel_geografico"] == "MUNICIPIO"].copy()
    df_geo_mun["id_municipio"] = (
        df_geo_mun["id_municipio"].astype(str).str.replace(".0", "", regex=False)
    )
    df_geo_mun["macrorregiao"] = df_geo_mun["id_municipio"].apply(macro_regiao)

    print("[2/5] A unificar tabelas fato (Resultados + Cobertura) por ano, geografia, série e rede...")
    keys = ["ano", "geo_key", "serie_key", "rede_key"]
    cols_cob = keys + [
        "quantidade_registros",
        "quantidade_presentes",
        "quantidade_cadernos_validos",
        "soma_peso_aluno",
        "taxa_presenca_pct",
        "taxa_cadernos_validos_pct",
        "media_proficiencia_ponderada",
    ]
    df_full = pd.merge(
        df_res,
        df_cob[[c for c in cols_cob if c in df_cob.columns]],
        on=keys,
        how="left",
    )

    # Filtrar registos municipais (geo_key começando por 'MUN:')
    df_full = df_full[df_full["geo_key"].astype(str).str.startswith("MUN:")].copy()

    print("[3/5] A estruturar separação temporal (2023 -> 2024) para evitar Data Leakage...")
    df_2023 = df_full[df_full["ano"] == 2023].copy()
    df_2024 = df_full[df_full["ano"] == 2024].copy()

    # Features históricas de 2023 (t-1): permitidas no modelo preditivo
    rename_2023 = {
        "taxa_alfabetizacao_pct": "taxa_alfabetizacao_2023_pct",
        "media_portugues": "media_portugues_2023",
        "percentual_participacao_pct": "participacao_2023_pct",
        "quantidade_registros": "qtd_registros_2023",
        "soma_peso_aluno": "soma_peso_aluno_2023",
        "media_proficiencia_ponderada": "proficiencia_ponderada_2023",
        "nivel_0_pct": "nivel_0_2023_pct",
        "nivel_1_pct": "nivel_1_2023_pct",
        "nivel_2_pct": "nivel_2_2023_pct",
        "nivel_3_pct": "nivel_3_2023_pct",
    }
    cols_2023_keep = ["geo_key", "serie_key", "rede_key"] + [
        k for k in rename_2023.keys() if k in df_2023.columns
    ]
    df_2023_feat = df_2023[cols_2023_keep].rename(columns=rename_2023)

    # Desfecho de 2024 (t): usado para construir o Target (y) e auditoria na EDA
    cols_2024_keep = [
        "geo_key",
        "serie_key",
        "rede_key",
        "taxa_alfabetizacao_pct",
        "media_portugues",
        "percentual_participacao_pct",
    ]
    df_2024_target = df_2024[[c for c in cols_2024_keep if c in df_2024.columns]].rename(
        columns={
            "taxa_alfabetizacao_pct": "taxa_alfabetizacao_2024_pct",
            "media_portugues": "media_portugues_2024_auditoria",
            "percentual_participacao_pct": "participacao_2024_auditoria_pct",
        }
    )

    df_abt = pd.merge(
        df_2024_target,
        df_2023_feat,
        on=["geo_key", "serie_key", "rede_key"],
        how="inner",
    )

    print("[4/5] A integrar ftc_meta_alfabetizacao e as 3 Dimensões (Geografia, Rede e Série)...")
    # Pivotar ftc_meta_alfabetizacao (metas de 2024 a 2030 por município)
    df_meta_mun = df_meta[df_meta["geo_key"].astype(str).str.startswith("MUN:")].copy()
    df_meta_pivot = (
        df_meta_mun.pivot_table(
            index="geo_key",
            columns="ano_meta",
            values="valor_meta_pct",
            aggfunc="first",
        )
        .add_prefix("valor_meta_")
        .add_suffix("_pct")
        .reset_index()
    )

    # Join com Metas
    df_abt = pd.merge(df_abt, df_meta_pivot, on="geo_key", how="left")

    # Join com Dimensão 1: dim_geografia
    cols_geo = ["geo_key", "id_municipio", "nome_geografia", "sigla_uf", "macrorregiao"]
    df_abt = pd.merge(df_abt, df_geo_mun[cols_geo], on="geo_key", how="left")

    # Join com Dimensão 2: dim_rede
    cols_rede = ["rede_key", "nome_rede", "escopo_rede", "dependencia_administrativa"]
    df_abt = pd.merge(
        df_abt,
        df_rede[[c for c in cols_rede if c in df_rede.columns]],
        on="rede_key",
        how="left",
    )

    # Join com Dimensão 3: dim_serie
    cols_serie = ["serie_key", "nome_serie", "ano_escolar", "etapa_ensino"]
    df_abt = pd.merge(
        df_abt,
        df_serie[[c for c in cols_serie if c in df_serie.columns]],
        on="serie_key",
        how="left",
    )

    print("[5/5] A enriquecer com indicadores socioeconómicos do IBGE...")
    df_ibge = dados_ibge()
    if not df_ibge.empty:
        df_abt = pd.merge(df_abt, df_ibge, on="id_municipio", how="left")
    else:
        df_abt["pib_per_capita_brl"] = np.nan
        df_abt["populacao_estimada"] = np.nan

    # =========================================================================
    # ENGENHARIA DE ATRIBUTOS (FEATURE ENGINEERING) E DEFINIÇÃO DO TARGET (y)
    # =========================================================================
    if "valor_meta_2024_pct" in df_abt.columns:
        df_abt["esforco_necessario_meta_2024_pp"] = (
            df_abt["valor_meta_2024_pct"] - df_abt["taxa_alfabetizacao_2023_pct"]
        )
        df_abt["ja_superava_meta_em_2023"] = (
            df_abt["taxa_alfabetizacao_2023_pct"] >= df_abt["valor_meta_2024_pct"]
        ).astype(int)

    # Soma de alunos nos níveis mais baixos de proficiência em 2023 (Níveis 0 a 2 = vulnerabilidade)
    low_lvl_cols = [c for c in ["nivel_0_2023_pct", "nivel_1_2023_pct", "nivel_2_2023_pct"] if c in df_abt.columns]
    if low_lvl_cols:
        df_abt["proporcao_baixo_nivel_2023_pct"] = df_abt[low_lvl_cols].sum(axis=1)

    df_abt["log_pib_per_capita"] = np.log1p(df_abt["pib_per_capita_brl"].clip(lower=0))
    df_abt["log_populacao"] = np.log1p(df_abt["populacao_estimada"].clip(lower=0))

    # Target Principal: Atingiu a Meta Oficial de 2024 (1 = Sim, 0 = Não)
    df_abt["target_atingiu_meta_2024"] = np.where(
        df_abt["taxa_alfabetizacao_2024_pct"] >= df_abt["valor_meta_2024_pct"], 1, 0
    )

    # Target Secundário: Acima do Limiar Nacional de 2024 (59.9%)
    df_abt["target_acima_limiar_nacional"] = np.where(
        df_abt["taxa_alfabetizacao_2024_pct"] >= 59.9, 1, 0
    )

    df_abt = df_abt.dropna(subset=["taxa_alfabetizacao_2024_pct", "valor_meta_2024_pct"]).copy()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / "abt_alfabetizacao_municipios.csv"
    df_abt.to_csv(output_path, index=False)

    print("\n" + "=" * 70)
    print(f"[SUCESSO] Base Analítica (ABT) gerada integrando as 6 tabelas Gold: {output_path}")
    print(f"Total de registos válidos (Município/Rede): {len(df_abt):,}")
    print(f"Municípios únicos mapeados: {df_abt['id_municipio'].nunique():,}")
    print(f"Redes presentes: {df_abt['nome_rede'].unique().tolist()}")
    print(
        f"Distribuição do Target (target_atingiu_meta_2024):\n"
        f"{df_abt['target_atingiu_meta_2024'].value_counts(normalize=True).round(4) * 100}"
    )
    print("=" * 70)

    return df_abt


if __name__ == "__main__":
    build_analytical_dataset()