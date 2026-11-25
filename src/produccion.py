"""Modelo de producción: entrenamiento con todos los años, artefacto guardado y predicción de un brote nuevo.

El artefacto reúne todo lo que la aplicación necesita para predecir igual que en el entrenamiento:
modelo, codificador, calibrador, umbral, tasas históricas regionales y catálogos para los formularios."""
import joblib
import numpy as np
import pandas as pd

from . import datos as D, balanceo as B, modelos_ml as ML
from .config import (CAT, NUM, RAIZ, SEMILLA)

# elegidos en el notebook 02
PARAMS_RF = {"max_depth":None, "min_samples_leaf":5}
RUTA_MODELO = RAIZ/"modelos"/"modelo_v1.joblib"


# ------entrenamiento------
def entrenar_rf(part, semilla=SEMILLA):
  """Random Forest con ponderación de clases entrenado con part['tr']. Devuelve (modelo, codificador)."""
  cod = D.Codificador().fit(part["tr"][CAT+NUM])
  Mb, yb, w = B.balancear(cod.transform(part["tr"][CAT+NUM]), part["tr"].y.values, "Ponderación de clases", cod, semilla=semilla)
  rf = ML.ajustar(ML.crear_modelo("Random Forest", PARAMS_RF, cod, semilla=semilla), Mb, yb, w)
  return rf, cod

def predicciones_fuera_de_muestra(datos, anios):
  """Validación cruzada temporal de ventana creciente: para cada año k, entrena con los años
  anteriores y predice k. Devuelve (probabilidades, etiquetas, años)."""
  p, y, a = [], [], []
  for k in anios:
      part_k = D.particionar(datos, fin_entrenamiento=k-1, fin_validacion=k)
      rf_k, cod_k = entrenar_rf(part_k)
      p.append(ML.probabilidad(rf_k, cod_k.transform(part_k["va"][CAT+NUM])))
      y.append(part_k["va"].y.values)
      a.append(part_k["va"]["Año estadistico"].values)
  return np.concatenate(p), np.concatenate(y), np.concatenate(a)

def tasa_para_anio(datos_con_historia, anio, m=50):
  """Tasa histórica regional para brotes de `anio`, calculada con TODOS los años anteriores
  y suavizada hacia la tasa nacional (misma fórmula que datos.tasa_regional_historica)."""
  prev = datos_con_historia[datos_con_historia["Año estadistico"]<anio]
  glob = prev.y.mean()
  g = prev.groupby("Región de consumo").y.agg(["sum", "count"])
  tasas = (g["sum"]+m*glob)/(g["count"]+m)
  return {"por_region": tasas.to_dict(), "nacional": float(glob)}


def catalogos(datos):
  """Listas cerradas para los formularios: comunas por región y la jerarquía de alimentos."""
  comunas = (datos.groupby("Región de consumo")["comuna de consumo"]
             .apply(lambda s: sorted(s.dropna().unique())).to_dict())
  jerarquia = ["Grandes grupos alimentos sospechoso", "Grupo Alimento Sospechoso",
               "Sub agrupación alimento sospechoso", "Alimento Sospechoso"]
  alimentos = datos[jerarquia].drop_duplicates().sort_values(jerarquia).reset_index(drop=True)
  return {"comunas_por_region": comunas,
          "alimentos": alimentos,
          "local_elaboracion": sorted(datos["Local de elaboración"].unique()),
          "local_consumo": sorted(datos["Local consumo"].unique())}


# ------artefacto------
def guardar_artefacto(artefacto, ruta=RUTA_MODELO):
  ruta.parent.mkdir(parents=True, exist_ok=True)
  # comprimido: ~14 MB, cabe en GitHub
  joblib.dump(artefacto, ruta, compress=3)          
  return ruta

def cargar_artefacto(ruta=RUTA_MODELO):
  return joblib.load(ruta)

# ------predicción de un brote nuevo------
def preparar_brote(entrada, artefacto):
  """Convierte los datos del formulario en la fila de predictores que espera el modelo.
  entrada: dict con fecha_notificacion, fecha_ingestion (o None), incubacion (días o None),
           region, comuna, local_elaboracion, local_consumo, grandes_grupos, grupo, subagrupacion, alimento."""
  f_not = pd.Timestamp(entrada["fecha_notificacion"])
  f_ing = pd.Timestamp(entrada["fecha_ingestion"]) if entrada.get("fecha_ingestion") else pd.NaT
  retraso = (f_not - f_ing).days if pd.notna(f_ing) else np.nan
  tasas = artefacto["tasas"]
  fila = {
      "Región de consumo": entrada["region"],
      "comuna de consumo": entrada["comuna"],
      "Local de elaboración": entrada["local_elaboracion"],
      "Local consumo": entrada["local_consumo"],
      "Grupo Alimento Sospechoso": entrada["grupo"],
      "Alimento Sospechoso": entrada["alimento"],
      "Sub agrupación alimento sospechoso": entrada["subagrupacion"],
      "Grandes grupos alimentos sospechoso": entrada["grandes_grupos"],
      "Año estadistico": int(f_not.year),
      "incubacion": min(entrada["incubacion"], 30) if entrada.get("incubacion") is not None else np.nan,
      "retraso_notif": retraso if (pd.notna(retraso) and 0 <= retraso <= 60) else np.nan,
      "mes": int(f_not.month),
      "Semana estadistica": int(f_not.isocalendar().week),
      "temporada_estival": int(f_not.month in (12, 1, 2, 3)),
      "tasa_hist_region": tasas["por_region"].get(entrada["region"], tasas["nacional"]),
  }
  # Categorías poco frecuentes: igual que en el entrenamiento
  cod = artefacto["codificador"]
  if fila["comuna de consumo"] not in set(cod.enc.categories_[CAT.index("comuna de consumo")]):
      fila["comuna de consumo"] = "Otra comuna"
  if fila["Alimento Sospechoso"] not in set(cod.enc.categories_[CAT.index("Alimento Sospechoso")]):
      fila["Alimento Sospechoso"] = "Otro alimento"
  return pd.DataFrame([fila])[CAT + NUM]

def nivel(prob, artefacto):
  """Nivel de prioridad según los cortes guardados en el artefacto."""
  cortes = artefacto["niveles"]
  if prob >= cortes["alto"]:
      return "Alto"
  return "Medio" if prob >= cortes["medio"] else "Bajo"


def predecir(fila, artefacto):
  """Probabilidad calibrada con la capacidad real de la región y con el nivel de referencia."""
  cod, modelo, cal = artefacto["codificador"], artefacto["modelo"], artefacto["calibrador"]
  M = cod.transform(fila)
  M_ref = M.copy()
  M_ref[:, cod.columna("tasa_hist_region")] = artefacto["referencia_tasa"]
  p_real = float(cal.transform(ML.probabilidad(modelo, M))[0])
  p_ref = float(cal.transform(ML.probabilidad(modelo, M_ref))[0])
  return {"prob_real": p_real, "prob_referencia": p_ref,
          "nivel_real": nivel(p_real, artefacto), "nivel_referencia": nivel(p_ref, artefacto)}


# ---------------------------------------------------------------- explicación (SHAP)
NOMBRES_VARIABLES = {
    "Región de consumo": "Región", "comuna de consumo": "Comuna",
    "Local de elaboración": "Lugar de elaboración", "Local consumo": "Lugar de consumo",
    "Grupo Alimento Sospechoso": "Grupo de alimento", "Alimento Sospechoso": "Alimento",
    "Sub agrupación alimento sospechoso": "Subagrupación de alimento",
    "Grandes grupos alimentos sospechoso": "Grandes grupos de alimento",
    "Año estadistico": "Año", "incubacion": "Incubación (días)", "retraso_notif": "Retraso de notificación (días)",
    "mes": "Mes", "Semana estadistica": "Semana", "temporada_estival": "Temporada estival",
    "tasa_hist_region": "Tasa histórica de la región",
}


def explicar(fila, artefacto, explainer, n=5):
    """Factores que más influyeron en la probabilidad (SHAP), expresados en la escala CALIBRADA.

    SHAP explica la probabilidad sin calibrar del Random Forest. Como la calibración de Platt es creciente,
    cada contribución se reescala por el mismo factor para que todas sumen exactamente la diferencia entre la
    probabilidad calibrada del brote y la de un brote promedio (el valor base calibrado). Así se conserva el
    signo y el tamaño relativo de cada factor. Devuelve (DataFrame de factores, valor base calibrado)."""
    import shap  # noqa: F401  (el explainer ya viene creado; se importa para dejar explícita la dependencia)

    cod, modelo, cal = artefacto["codificador"], artefacto["modelo"], artefacto["calibrador"]
    M = cod.transform(fila)
    sv = explainer.shap_values(M, check_additivity=False)
    sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if np.ndim(sv) == 3 else sv)   # clase "específico"
    sv = np.asarray(sv)[0]
    base = explainer.expected_value
    base = float(base[1]) if np.ndim(base) else float(base)

    p_cruda = float(ML.probabilidad(modelo, M)[0])
    base_cal, p_cal = (float(v) for v in cal.transform(np.array([base, p_cruda])))
    factor = (p_cal - base_cal) / (p_cruda - base) if abs(p_cruda - base) > 1e-9 else 0.0

    valores = fila.iloc[0]
    tabla = pd.DataFrame({"variable": [NOMBRES_VARIABLES[c] for c in CAT + NUM],
                          "valor": [valores[c] for c in CAT + NUM],
                          "contribucion": sv * factor})
    tabla = tabla.reindex(tabla.contribucion.abs().sort_values(ascending=False).index).head(n)
    return tabla.reset_index(drop=True), base_cal
