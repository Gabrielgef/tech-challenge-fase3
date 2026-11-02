"""Gráficos usados no notebook de modelagem."""

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

RAIZ = Path(__file__).resolve().parents[2]
IMAGENS = RAIZ / "images"
COR_NAO, COR_SIM, COR_NEUTRA = "#c0504d", "#4f8f7a", "#8a8a8a"


def salva(fig, nome: str) -> None:
    IMAGENS.mkdir(exist_ok=True)
    fig.savefig(IMAGENS / nome, dpi=150, bbox_inches="tight")


def curvas_roc_pr(y, probas: dict):
    """ROC e precisão x recall para um ou mais modelos (probabilidade de cumprir a meta)."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for nome, p in probas.items():
        RocCurveDisplay.from_predictions(y, p, name=nome, ax=ax[0])
        PrecisionRecallDisplay.from_predictions(y, p, name=nome, ax=ax[1])
    ax[0].plot([0, 1], [0, 1], ls="--", color=COR_NEUTRA, lw=1)
    ax[0].set_title("Curva ROC")
    ax[1].axhline(y.mean(), ls="--", color=COR_NEUTRA, lw=1)
    ax[1].set_title("Precisão x recall")
    plt.tight_layout()
    return fig


def matriz_confusao(y, pred):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y, pred, display_labels=["não cumpre", "cumpre"], cmap="Greens", colorbar=False, ax=ax
    )
    ax.set_title("Matriz de confusão (teste)")
    ax.grid(False)
    plt.tight_layout()
    return fig
