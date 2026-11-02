"""
Pipeline de modelagem: pré-processamento e modelo no mesmo objeto.

Imputação, escala e codificação ficam dentro do Pipeline, então são ajustadas
só com os dados de treino de cada fold (sem vazamento para a validação).
"""

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.preprocessing.features import CATEGORICAS, NUMERICAS

SEED = 42


def monta_preprocessador(numericas=NUMERICAS, categoricas=CATEGORICAS) -> ColumnTransformer:
    num = Pipeline([("imputa", SimpleImputer(strategy="median")), ("escala", StandardScaler())])
    cat = Pipeline(
        [
            ("imputa", SimpleImputer(strategy="most_frequent")),
            ("codifica", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    etapas = [("num", num, list(numericas))]
    if categoricas:
        etapas.append(("cat", cat, list(categoricas)))
    return ColumnTransformer(etapas)


def monta_pipeline(modelo, numericas=NUMERICAS, categoricas=CATEGORICAS) -> Pipeline:
    return Pipeline([("prep", monta_preprocessador(numericas, categoricas)), ("modelo", modelo)])


def candidatos(seed: int = SEED) -> dict:
    return {
        "Baseline (classe majoritária)": DummyClassifier(strategy="most_frequent"),
        "Regressão logística": LogisticRegression(max_iter=1000),
        "Random forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=10, n_jobs=-1, random_state=seed),
        "Gradient boosting": GradientBoostingClassifier(random_state=seed),
    }


GRADE_RANDOM_FOREST = {
    "modelo__n_estimators": [200, 300, 500],
    "modelo__max_depth": [4, 6, 8, None],
    "modelo__min_samples_leaf": [5, 10, 20, 40],
    "modelo__max_features": ["sqrt", 0.5, 1.0],
}


def busca_hiperparametros(pipeline, grade, X, y, cv, n_iter=12, seed=SEED) -> RandomizedSearchCV:
    busca = RandomizedSearchCV(
        pipeline,
        param_distributions=grade,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=seed,
        n_jobs=-1,
        refit=True,
    )
    return busca.fit(X, y)
