"""Métricas de validação e de teste. A classe de interesse para a política pública é a 0 (não cumpre a meta)."""

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_validate

SCORING = {
    "auc": "roc_auc",
    "ap": "average_precision",
    "f1": "f1",
    "precisao_nao_cumpre": make_scorer(precision_score, pos_label=0, zero_division=0),
    "recall_nao_cumpre": make_scorer(recall_score, pos_label=0, zero_division=0),
}


def valida_cruzado(pipeline, X, y, cv, grupos=None) -> dict:
    """Validação cruzada com todas as métricas; devolve média e desvio de cada uma."""
    r = cross_validate(
        pipeline, X, y, cv=cv, groups=grupos, scoring=SCORING, return_train_score=True, n_jobs=-1
    )
    saida = {}
    for nome in SCORING:
        saida[nome] = r[f"test_{nome}"].mean()
        saida[f"{nome}_dp"] = r[f"test_{nome}"].std()
    saida["auc_treino"] = r["train_auc"].mean()
    return saida


def metricas_teste(y, proba_cumpre, limiar: float = 0.5) -> dict:
    pred = (proba_cumpre >= limiar).astype(int)
    return {
        "auc": roc_auc_score(y, proba_cumpre),
        "ap": average_precision_score(y, proba_cumpre),
        "f1": f1_score(y, pred),
        "precisao_nao_cumpre": precision_score(y, pred, pos_label=0, zero_division=0),
        "recall_nao_cumpre": recall_score(y, pred, pos_label=0, zero_division=0),
        "acuracia": (pred == y).mean(),
    }


def tabela_comparacao(resultados: dict) -> pd.DataFrame:
    linhas = []
    for nome, r in resultados.items():
        linhas.append(
            {
                "modelo": nome,
                "AUC": r["auc"],
                "± AUC": r["auc_dp"],
                "AUC treino": r["auc_treino"],
                "AP": r["ap"],
                "F1": r["f1"],
                "precisão (não cumpre)": r["precisao_nao_cumpre"],
                "recall (não cumpre)": r["recall_nao_cumpre"],
            }
        )
    return pd.DataFrame(linhas).set_index("modelo").round(3)
