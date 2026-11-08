"""Técnicas de aprendizaje automático: regresión logística, Random Forest y gradient boosting."""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import SEMILLA

MODELOS_ML = ["Regresión logística", "Random Forest", "Gradient boosting"]

# Hiperparámetros probados en la búsqueda en grilla (criterio: AUC-PR en validación)
GRILLAS = {
    "Regresión logística": [{"C": c} for c in [0.01, 0.1, 1.0]],
    "Random Forest": [{"max_depth": md, "min_samples_leaf": ml} for md in [8, 16, None] for ml in [5, 20]],
    "Gradient boosting": [{"learning_rate": lr, "max_depth": md, "l2_regularization": 1.0}
                          for lr in [0.03, 0.1] for md in [3, 6]],
}


def crear_modelo(nombre, params, codificador, semilla=SEMILLA):
    """Construye un modelo sin entrenar."""
    if nombre == "Regresión logística":
        prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), codificador.idx_cat),
                                  ("num", StandardScaler(), codificador.idx_num)])
        return Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=2000, **params))])
    if nombre == "Random Forest":
        return RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=semilla, **params)
    if nombre == "Gradient boosting":
        return HistGradientBoostingClassifier(categorical_features=codificador.idx_cat, random_state=semilla,
                                              max_iter=400, early_stopping=False, **params)
    raise ValueError(f"Modelo desconocido: {nombre}")


def ajustar(modelo, M, y, w=None):
    """Entrena el modelo; si hay pesos (ponderación de clases) los pasa a fit()."""
    if w is None:
        return modelo.fit(M, y)
    if isinstance(modelo, Pipeline):
        return modelo.fit(M, y, clf__sample_weight=w)
    return modelo.fit(M, y, sample_weight=w)


def probabilidad(modelo, M):
    """Probabilidad de diagnóstico específico (clase 1)."""
    return modelo.predict_proba(M)[:, 1]
