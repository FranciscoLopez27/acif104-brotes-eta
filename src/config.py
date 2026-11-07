"""Configuración común del proyecto: rutas, semilla, variables y estilo de los gráficos."""
from pathlib import Path

import matplotlib.pyplot as plt

# ---------------------------------------------------------------- rutas
RAIZ = Path(__file__).resolve().parents[1]                      # carpeta raíz del repositorio
RUTA_DATOS = RAIZ / "data" / "BD_Brotes_ETA_2011_2025_comuna.xlsx"
DIR_FIG = RAIZ / "results" / "figuras"                          # figuras para el informe
DIR_TAB = RAIZ / "results" / "tablas"                           # tablas en CSV para el informe
DIR_FIG.mkdir(parents=True, exist_ok=True)
DIR_TAB.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- reproducibilidad
SEMILLA = 42

# ---------------------------------------------------------------- clasificación de las 41 variables
GRUPOS = {
    "Conocidas al notificar (predictores)": [
        "Año estadistico", "Semana estadistica", "Fecha de notificación", "Fecha de Ingestión",
        "Fecha Primeros Síntomas", "Periodo de Incubación Días", "Región de consumo", "comuna de consumo",
        "Local de elaboración", "Local consumo", "Grupo Alimento Sospechoso", "Alimento Sospechoso",
        "Sub agrupación alimento sospechoso", "Grandes grupos alimentos sospechoso"],
    "Cifras de afectados (excluidas)": [
        "Expuestos", "Enfermos", "Total de Hospitalizados", "Total sin atención", "Total ambulatorios",
        "Total enfermos <1", "Total enfermos 1-4", "Total enfermos 5-14", "Total enfermos 15-44",
        "Total enfermos 45-64", "Total enfermos >65"],
    "Resultados de la investigación (excluidas)": [
        "Duración Brote Días", "Fallecidos", "Tasa Ataque", "Tasa Letalidad", "Glosa_contaminación",
        "Glosa_supervivencia", "Glosa proliferación", "Proceso de pérdida de inocuidad",
        "Lugar Perdida Inocuidad", "Diagnostico especifico", "Código CIE-10", "Diagnostico agrupado",
        "Conclusión del brote"],
    "Variables objetivo": ["Tipo de Diagnostico", "Tipo de agente"],
    "Identificador": ["N° Brote ETA"],
}

# ---------------------------------------------------------------- predictores del modelo
CAT = ["Región de consumo", "comuna de consumo", "Local de elaboración", "Local consumo",
       "Grupo Alimento Sospechoso", "Alimento Sospechoso", "Sub agrupación alimento sospechoso",
       "Grandes grupos alimentos sospechoso"]                    # 8 categóricas
NUM = ["Año estadistico", "incubacion", "retraso_notif", "mes", "Semana estadistica",
       "temporada_estival", "tasa_hist_region"]                  # 7 numéricas
NUM_ENTERAS = ["Año estadistico", "mes", "Semana estadistica", "temporada_estival"]   # se redondean tras SMOTE-NC

BALANCEOS = ["Sin balanceo", "Ponderación de clases", "SMOTE-NC", "Submuestreo"]

# ---------------------------------------------------------------- estilo de gráficos
AZUL, NARANJO, GRIS, VERDE = "#2a6f97", "#e07a1f", "#8d99ae", "#6a994e"
plt.rcParams.update({"figure.dpi": 110, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})


def guardar(fig, nombre):
    """Ajusta márgenes, guarda la figura en results/figuras (200 dpi) y la muestra."""
    fig.tight_layout()
    fig.savefig(DIR_FIG / nombre, dpi=200, bbox_inches="tight")
    plt.show()


def guardar_tabla(tabla, nombre):
    """Guarda una tabla (DataFrame o Series) en results/tablas como CSV."""
    tabla.to_csv(DIR_TAB / f"{nombre}.csv")
