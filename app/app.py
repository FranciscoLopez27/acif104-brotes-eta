"""Priorización ETA: aplicación en Streamlit (paso 0, prueba de despliegue).
Verifica que el modelo de producción se carga en Streamlit Community Cloud y que predice igual que en Colab."""
import sys
import time
from pathlib import Path
import streamlit as st
# carpeta raíz del repositorio
RAIZ = Path(__file__).resolve().parents[1]
# para importar los módulos de src/
sys.path.insert(0, str(RAIZ))
# noqa: E402
from src import produccion as PR  
st.set_page_config(page_title="Priorización ETA", page_icon="🧪", layout="wide")
# se carga una sola vez y se reutiliza
@st.cache_resource                                  
def cargar_modelo():
    return PR.cargar_artefacto()

art = cargar_modelo()
st.title("Priorización ETA · prueba de despliegue")
st.write(f"Modelo v{art['version']}, entrenado con brotes {art['anios_entrenamiento']} "
         f"el {art['fecha_entrenamiento']}. Umbral del nivel alto: {art['umbral']:.3f}.")

# Mismo brote de ejemplo de la celda 10 del notebook 04: debe dar el mismo resultado
alim = art["catalogos"]["alimentos"]
fila_alim = alim[alim["Grupo Alimento Sospechoso"].str.startswith("Huevos")].iloc[0]
ejemplo = {"fecha_notificacion": "2026-01-14", "fecha_ingestion": "2026-01-10", "incubacion": 1,
           "region": "Del bíobío", "comuna": "Concepción",
           "local_elaboracion": art["catalogos"]["local_elaboracion"][0],
           "local_consumo": art["catalogos"]["local_consumo"][0],
           "grandes_grupos": fila_alim.iloc[0], "grupo": fila_alim.iloc[1],
           "subagrupacion": fila_alim.iloc[2], "alimento": fila_alim.iloc[3]}
t0 = time.perf_counter()
resultado = PR.predecir(PR.preparar_brote(ejemplo, art), art)
st.subheader("Brote de ejemplo (Concepción, huevos)")
st.json({k: (round(v, 4) if isinstance(v, float) else v) for k, v in resultado.items()})
st.caption(f"Predicción en {1000*(time.perf_counter()-t0):.0f} ms")

import sklearn, numpy, pandas, shap                 # noqa: E402
st.caption(f"Python {sys.version.split()[0]} · scikit-learn {sklearn.__version__} · numpy {numpy.__version__} · "
           f"pandas {pandas.__version__} · shap {shap.__version__} · streamlit {st.__version__}")
