"""Calibración de probabilidades: método de Platt e isotónico, y error de calibración"""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
EPS = 1e-4   # evita log(0) cuando el modelo entrega probabilidades exactas de 0 o 1

def _logit(p):
    """Transforma probabilidades a la escala logit, log(p/(1-p))"""
    p = np.clip(np.asarray(p, dtype=float), EPS, 1-EPS)
    return np.log(p/(1-p))


class Calibrador:
    """Transforma las probabilidades de un modelo en probabilidades calibradas.
    metodo="platt"     → regresión logística sobre el logit de la probabilidad (curva en S, 2 parámetros).
    metodo="isotonica" → función escalonada creciente (flexible, necesita más datos).
    Se ajusta con predicciones FUERA DE MUESTRA (por ejemplo, las de la validación cruzada temporal),
    nunca con las del conjunto de entrenamiento ni con las del conjunto de prueba."""

    def __init__(self, metodo="platt"):
        if metodo not in ("platt", "isotonica"):
            raise ValueError(f"Método desconocido: {metodo}")
        self.metodo = metodo

    def fit(self, p, y):
        if self.metodo == "platt":
            self.modelo = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p).reshape(-1, 1), y)
        else:
            self.modelo = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(np.asarray(p, float), y)
        return self

    def transform(self, p):
        if self.metodo == "platt":
            return self.modelo.predict_proba(_logit(p).reshape(-1, 1))[:, 1]
        return self.modelo.predict(np.asarray(p, dtype=float))


def error_calibracion(y, p, n_tramos=10):
    """Error esperado de calibración (ECE): se ordenan los brotes por probabilidad, se dividen en
    n_tramos de igual tamaño y se promedia |probabilidad media − proporción real| ponderado por tramo.
    0 = calibración perfecta."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    orden = np.argsort(p)
    tramos = np.array_split(orden, n_tramos)
    return float(sum(len(t)*abs(p[t].mean()-y[t].mean()) for t in tramos)/len(y))
