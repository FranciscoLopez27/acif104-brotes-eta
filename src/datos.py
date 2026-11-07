"""Carga, limpieza, construcción de variables, partición temporal y codificación de los brotes de ETA."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from .config import RUTA_DATOS, CAT, NUM

TEXTO_CAT = CAT  # variables categóricas que se limpian


def cargar(ruta=RUTA_DATOS):
    """Lee la hoja de datos del Excel del DEIS y limpia espacios en los nombres de columnas."""
    df = pd.read_excel(ruta, sheet_name="ETA 2011-2025")
    df.columns = [c.strip() for c in df.columns]
    return df


def preparar(df, solo_confirmados=True):
    """Aplica las decisiones del análisis exploratorio.

    solo_confirmados=True  → población de la fase 1: brotes con diagnóstico y conclusión "se confirma".
    solo_confirmados=False → todos los brotes con diagnóstico (análisis de sensibilidad del sesgo de selección).
    """
    x = df[df["Tipo de Diagnostico"].notna()].copy()
    x["conclusion"] = x["Conclusión del brote"].astype("string").str.strip().fillna("Sin conclusión")
    if solo_confirmados:
        x = x[x["conclusion"] == "Realizada, se confirma"].copy()
    x["y"] = (x["Tipo de Diagnostico"] == "Diagnostico especifico").astype(int)   # 1 = específico

    # Correctitud: fechas de relleno → faltante; retrasos negativos o > 60 días → faltante
    x.loc[x["Fecha de Ingestión"] < "2010-01-01", "Fecha de Ingestión"] = pd.NaT
    ret = (x["Fecha de notificación"] - x["Fecha de Ingestión"]).dt.days
    x["retraso_notif"] = ret.where(ret.between(0, 60))

    # Atípicos: incubación acotada a 30 días
    x["incubacion"] = x["Periodo de Incubación Días"].clip(upper=30)

    # Calendario
    x["mes"] = x["Fecha de notificación"].dt.month
    x["temporada_estival"] = x["mes"].isin([12, 1, 2, 3]).astype(int)

    # Categóricas: escritura unificada; "Sin definir"/"Sin identificar"/vacíos → "Desconocido"
    for c in TEXTO_CAT:
        s = x[c].astype("string").str.strip().str.capitalize()
        x[c] = s.where(~s.isin(["Sin definir", "Sin identificar"])).fillna("Desconocido").astype(object)

    x["tasa_hist_region"] = tasa_regional_historica(x)
    return x[x["Año estadistico"] >= 2012].copy()          # 2011 solo sirve como historia


def tasa_regional_historica(x, m=50):
    """Tasa de diagnóstico específico de la región calculada SOLO con años anteriores,
    suavizada hacia la tasa nacional (m = peso del suavizado)."""
    out = pd.Series(np.nan, index=x.index)
    for anio in sorted(x["Año estadistico"].unique()):
        prev = x[x["Año estadistico"] < anio]
        actual = x["Año estadistico"] == anio
        if len(prev) == 0:
            continue
        glob = prev.y.mean()
        g = prev.groupby("Región de consumo").y.agg(["sum", "count"])
        suav = (g["sum"] + m * glob) / (g["count"] + m)
        out[actual] = x.loc[actual, "Región de consumo"].map(suav).fillna(glob).values
    return out


def particionar(datos, fin_entrenamiento=2020, fin_validacion=2022, min_comuna=15, min_alimento=10):
    """Partición temporal y agrupación de categorías poco frecuentes (contadas solo en entrenamiento).

    Devuelve un diccionario con los DataFrames 'tr', 'va', 'te' (prueba = años posteriores a fin_validacion)."""
    d = datos.copy()
    d["conjunto"] = np.where(d["Año estadistico"] <= fin_entrenamiento, "Entrenamiento",
                     np.where(d["Año estadistico"] <= fin_validacion, "Validación", "Prueba"))
    entrenamiento = d["conjunto"] == "Entrenamiento"
    frec_com = d.loc[entrenamiento, "comuna de consumo"].value_counts()
    d["comuna de consumo"] = d["comuna de consumo"].where(
        d["comuna de consumo"].map(frec_com).fillna(0) >= min_comuna, "Otra comuna")
    frec_ali = d.loc[entrenamiento, "Alimento Sospechoso"].value_counts()
    d["Alimento Sospechoso"] = d["Alimento Sospechoso"].where(
        d["Alimento Sospechoso"].map(frec_ali).fillna(0) >= min_alimento, "Otro alimento")
    return {"datos": d,
            "tr": d[d.conjunto == "Entrenamiento"],
            "va": d[d.conjunto == "Validación"],
            "te": d[d.conjunto == "Prueba"]}


def resumen_particion(datos_particionados):
    """Tabla con años, brotes, positivos y prevalencia de cada conjunto."""
    d = datos_particionados
    t = d.groupby("conjunto").agg(años=("Año estadistico", lambda s: f"{s.min()}–{s.max()}"),
                                  brotes=("y", "size"), específicos=("y", "sum"), prevalencia=("y", "mean"))
    t = t.loc[[c for c in ["Entrenamiento", "Validación", "Prueba"] if c in t.index]]
    t["% del total"] = (t.brotes / t.brotes.sum() * 100).round(1)
    t["prevalencia"] = (t.prevalencia * 100).round(1)
    return t


class Codificador:
    """Convierte los predictores en una matriz numérica.
    Se ajusta SOLO con entrenamiento: categorías → enteros (desconocida = -1) y faltantes → mediana."""

    def __init__(self):
        self.enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1)
        self.idx_cat = list(range(len(CAT)))
        self.idx_num = list(range(len(CAT), len(CAT) + len(NUM)))

    def fit(self, X):
        self.enc.fit(X[CAT])
        self.mediana = X[NUM].median()
        self.cardinalidad = [len(c) for c in self.enc.categories_]
        return self

    def transform(self, X):
        return np.hstack([self.enc.transform(X[CAT]), X[NUM].fillna(self.mediana).values])

    def columna(self, nombre):
        """Posición en la matriz de una variable numérica."""
        return len(CAT) + NUM.index(nombre)
