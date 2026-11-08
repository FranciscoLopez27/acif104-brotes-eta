"""Técnicas de balanceo de clases. Se aplican SOLO al conjunto de entrenamiento."""
import numpy as np
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler

from .config import NUM_ENTERAS, SEMILLA


def balancear(M, y, metodo, codificador, semilla=SEMILLA):
    """Devuelve (M, y, pesos) según la técnica.

    - "Sin balanceo": datos originales (referencia).
    - "Ponderación de clases": mismos datos; cada positivo pesa (negativos / positivos).
    - "SMOTE-NC": positivos sintéticos interpolados entre 5 vecinos (apto para datos mixtos).
    - "Submuestreo": elimina negativos al azar hasta igualar las clases.
    """
    if metodo == "Sin balanceo":
        return M, y, None
    if metodo == "Ponderación de clases":
        w = np.where(y == 1, (y == 0).sum() / (y == 1).sum(), 1.0)
        return M, y, w
    if metodo == "SMOTE-NC":
        sm = SMOTENC(categorical_features=codificador.idx_cat, random_state=semilla, k_neighbors=5)
        Mb, yb = sm.fit_resample(M, y)
        for col in NUM_ENTERAS:                       # variables enteras: se redondean los valores sintéticos
            j = codificador.columna(col)
            Mb[:, j] = np.round(Mb[:, j])
        return Mb, yb, None
    if metodo == "Submuestreo":
        Mb, yb = RandomUnderSampler(random_state=semilla).fit_resample(M, y)
        return Mb, yb, None
    raise ValueError(f"Técnica de balanceo desconocida: {metodo}")
