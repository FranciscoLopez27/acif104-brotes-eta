"""Métricas, selección del umbral e intervalos de confianza por bootstrap."""
import numpy as np
from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score, recall_score,
                             precision_score, balanced_accuracy_score, brier_score_loss,
                             precision_recall_curve)

from .config import SEMILLA


# ---------------------------------------------------------------- umbral de decisión
def umbral_f1(y, p):
    """Umbral que maximiza el F1 (criterio usado en la fase 2)."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    return float(thr[np.nanargmax(f1[:-1])])


def umbral_para_recall(y, p, recall_min=0.70):
    """Umbral MÁS ALTO que logra una exhaustividad >= recall_min (criterio de RNF05).
    Entre todos los umbrales que cumplen el requisito, es el que da mayor precisión."""
    prec, rec, thr = precision_recall_curve(y, p)
    cumple = rec[:-1] >= recall_min                   # rec[:-1] se alinea con thr
    return float(thr[cumple].max()) if cumple.any() else float(thr.min())


# ---------------------------------------------------------------- métricas
def metricas(y, p, umbral):
    """Todas las métricas del informe a partir de probabilidades p y un umbral."""
    pred = (p >= umbral).astype(int)
    return {"AUC-PR": average_precision_score(y, p), "AUC-ROC": roc_auc_score(y, p),
            "Recall": recall_score(y, pred), "Precisión": precision_score(y, pred, zero_division=0),
            "F1": f1_score(y, pred), "Exactitud bal.": balanced_accuracy_score(y, pred),
            "Brier": brier_score_loss(y, p), "Umbral": umbral}


# ---------------------------------------------------------------- incertidumbre
def bootstrap_ic(y, p, metrica=average_precision_score, n=1000, nivel=0.95, semilla=SEMILLA):
    """Intervalo de confianza por bootstrap (percentiles): se remuestrean los brotes con reemplazo."""
    rng = np.random.default_rng(semilla)
    y, p = np.asarray(y), np.asarray(p)
    valores = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():                  # muestra sin positivos o sin negativos: se descarta
            continue
        valores.append(metrica(y[i], p[i]))
    a = (1 - nivel) / 2
    return metrica(y, p), float(np.quantile(valores, a)), float(np.quantile(valores, 1 - a))


def bootstrap_diferencia(y, p1, p2, metrica=average_precision_score, n=1000, nivel=0.95, semilla=SEMILLA):
    """Diferencia pareada metrica(p1) - metrica(p2) con su intervalo de confianza.
    Si el intervalo incluye 0, la diferencia entre los dos modelos no es concluyente."""
    rng = np.random.default_rng(semilla)
    y, p1, p2 = np.asarray(y), np.asarray(p1), np.asarray(p2)
    difs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        difs.append(metrica(y[i], p1[i]) - metrica(y[i], p2[i]))   # misma muestra para ambos modelos
    a = (1 - nivel) / 2
    return metrica(y, p1) - metrica(y, p2), float(np.quantile(difs, a)), float(np.quantile(difs, 1 - a))
