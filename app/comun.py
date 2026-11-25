"""Funciones compartidas por las páginas de la aplicación: carga del modelo y formato de textos."""
import sys
from pathlib import Path

import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]          # carpeta raíz del repositorio
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))                   # para importar los módulos de src/

from src import produccion as PR                    # noqa: E402

PALABRAS_MINUSCULA = {"de", "del", "la", "las", "los", "y", "el", "en", "a"}


@st.cache_resource                                  # se carga una sola vez por servidor
def cargar_modelo():
    return PR.cargar_artefacto()


@st.cache_resource
def cargar_explicador():
    """Explicador SHAP con 100 brotes reales de fondo (modo «interventional»).

    Con el modo por defecto, el punto de partida de SHAP sería 0,5, porque la ponderación de clases equilibra
    ambas clases al entrenar. Con brotes reales de fondo, el punto de partida es la probabilidad promedio
    de un brote real, que es más fácil de interpretar."""
    import shap
    from src import datos as D
    from src.config import CAT, NUM
    art = cargar_modelo()
    entrenamiento = D.particionar(D.preparar(D.cargar()), fin_entrenamiento=2025, fin_validacion=2025)["tr"]
    fondo = art["codificador"].transform(entrenamiento.sample(100, random_state=42)[CAT + NUM])
    return shap.TreeExplainer(art["modelo"], data=fondo, feature_perturbation="interventional")


def nombre_region(region):
    """'Del bíobío' → 'Biobío'; 'Metropolitana de santiago' → 'Metropolitana de Santiago'."""
    for prefijo in ("Del ", "De "):
        if region.startswith(prefijo):
            region = region[len(prefijo):]
    palabras = region.replace("bíobío", "biobío").split()
    return " ".join(p if (i > 0 and p in PALABRAS_MINUSCULA) else p[:1].upper() + p[1:]
                    for i, p in enumerate(palabras))


def porcentaje(p):
    return f"{100 * p:.0f}%"


COLORES_NIVEL = {"Alto": "#2a6f97", "Medio": "#e07a1f", "Bajo": "#8d99ae"}
