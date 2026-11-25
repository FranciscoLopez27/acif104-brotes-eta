"""Priorización ETA: aplicación en Streamlit.

Estima, al notificar un brote de enfermedad transmitida por alimentos, la probabilidad de que llegue a
diagnóstico específico, como apoyo a la priorización de la investigación en las SEREMI de Salud.
Ejecutar desde la raíz del repositorio:  streamlit run app/app.py"""
import streamlit as st

from comun import cargar_modelo

st.set_page_config(page_title="Priorización ETA", page_icon="🧪", layout="wide")

art = cargar_modelo()

paginas = [
    st.Page("paginas/registrar.py", title="Registrar brote", icon="📝", default=True),
]
navegacion = st.navigation(paginas)

with st.sidebar:
    st.markdown("**Priorización ETA**  \nApoyo a la investigación de brotes")
    st.caption(f"Modelo v{art['version']}: Random Forest con ponderación de clases y calibración de Platt, "
               f"entrenado con brotes {art['anios_entrenamiento']}.")
    st.caption("Proyecto ACIF104 · Universidad Andrés Bello. Herramienta de apoyo: no reemplaza el criterio "
               "del profesional a cargo.")

navegacion.run()
