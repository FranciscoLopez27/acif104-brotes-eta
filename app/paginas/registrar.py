"""Pantallas «Registrar brote» y «Resultado» (RF01, RF02, RF03, RF05; RNF01, RNF02)."""
import time
from datetime import date, timedelta

import matplotlib.pyplot as plt
import streamlit as st

from comun import PR, cargar_explicador, cargar_modelo, nombre_region, porcentaje

art = cargar_modelo()
cat = art["catalogos"]
alimentos = cat["alimentos"]
DESCONOCIDO = "Desconocido"

st.title("Registrar un brote notificado")
st.caption("Ingrese solo la información disponible al momento de la notificación. "
           "No se solicitan datos personales ni cifras de afectados.")

# ---------------------------------------------------------------- formulario (RF01)
# Las listas dependientes (comuna según región, alimento según grupo) se eligen fuera de st.form
# para que se actualicen al instante.
st.subheader("Fechas del brote")
c1, c2, c3 = st.columns(3)
f_notif = c1.date_input("Fecha de notificación *", value=date.today(), max_value=date.today(), format="DD/MM/YYYY")
ingestion_conocida = c2.checkbox("Se conoce la fecha de ingestión", value=True)
f_ing = c2.date_input("Fecha de ingestión", value=date.today() - timedelta(days=2), max_value=date.today(),
                      format="DD/MM/YYYY", disabled=not ingestion_conocida)
incub_conocida = c3.checkbox("Se conoce el período de incubación", value=True)
incubacion = c3.number_input("Período de incubación (días)", min_value=0, max_value=65, value=1, step=1,
                             disabled=not incub_conocida)

st.subheader("Lugar")
c1, c2 = st.columns(2)
regiones = sorted(cat["comunas_por_region"], key=nombre_region)
region = c1.selectbox("Región de consumo *", regiones, format_func=nombre_region)
comuna = c2.selectbox("Comuna de consumo *", cat["comunas_por_region"][region])
l_elab = c1.selectbox("Lugar de elaboración *", cat["local_elaboracion"])
l_cons = c2.selectbox("Lugar de consumo *", cat["local_consumo"])

st.subheader("Alimento sospechoso")
c1, c2 = st.columns(2)
gg = c1.selectbox("Grandes grupos de alimento *", sorted(alimentos.iloc[:, 0].unique()))
sub = alimentos[alimentos.iloc[:, 0] == gg]
grupo = c2.selectbox("Grupo de alimento *", sorted(sub.iloc[:, 1].unique()))
sub = sub[sub.iloc[:, 1] == grupo]
subagr = c1.selectbox("Subagrupación de alimento *", sorted(sub.iloc[:, 2].unique()))
sub = sub[sub.iloc[:, 2] == subagr]
alimento = c2.selectbox("Alimento *", sorted(sub.iloc[:, 3].unique()))
st.caption(f"Listas cerradas con las categorías de la base del DEIS. Si el alimento aún no se conoce, elija «{DESCONOCIDO}».")

# ---------------------------------------------------------------- validación (RF02)
errores, avisos = [], []
retraso = (f_notif - f_ing).days if ingestion_conocida else None
if retraso is not None:
    if retraso < 0:
        errores.append("La fecha de ingestión no puede ser posterior a la fecha de notificación.")
    elif retraso > 60:
        avisos.append(f"El retraso de notificación es de {retraso} días. Valores sobre 60 se consideran "
                      "error de registro y el modelo los tratará como desconocidos.")
if incub_conocida and incubacion > 30:
    avisos.append("Incubaciones sobre 30 días son muy escasas; el modelo las considera como 30 días.")
for e in errores:
    st.error(e)
for a in avisos:
    st.warning(a)
if retraso is not None and not errores and retraso <= 60:
    st.success(f"Fechas coherentes: retraso de notificación de {retraso} días.")

calcular = st.button("Estimar probabilidad", type="primary", disabled=bool(errores))

# ---------------------------------------------------------------- resultado (RF03, RF05)
if calcular:
    entrada = {"fecha_notificacion": f_notif, "fecha_ingestion": f_ing if ingestion_conocida else None,
               "incubacion": int(incubacion) if incub_conocida else None,
               "region": region, "comuna": comuna, "local_elaboracion": l_elab, "local_consumo": l_cons,
               "grandes_grupos": gg, "grupo": grupo, "subagrupacion": subagr, "alimento": alimento}
    t0 = time.perf_counter()
    fila = PR.preparar_brote(entrada, art)
    res = PR.predecir(fila, art)
    factores, base = PR.explicar(fila, art, cargar_explicador())
    st.session_state["ultimo"] = {"entrada": entrada, "resultado": res, "factores": factores, "base": base,
                                  "ms": 1000 * (time.perf_counter() - t0)}

if "ultimo" in st.session_state:
    u = st.session_state["ultimo"]
    res, factores, base, e = u["resultado"], u["factores"], u["base"], u["entrada"]
    st.divider()
    st.header("Probabilidad de diagnóstico específico")
    st.caption(f"{e['comuna']}, {nombre_region(e['region'])} · {e['alimento']} · notificado el "
               f"{e['fecha_notificacion']:%d-%m-%Y}")

    c1, c2 = st.columns(2)
    with c1.container(border=True):
        st.markdown("**Con la capacidad actual de la región**")
        st.metric("Probabilidad", porcentaje(res["prob_real"]), label_visibility="collapsed")
        st.markdown(f":{'blue' if res['nivel_real'] == 'Alto' else 'orange' if res['nivel_real'] == 'Medio' else 'gray'}"
                    f"-badge[Nivel {res['nivel_real'].lower()}]")
        st.progress(min(res["prob_real"], 1.0))
        tasa = art["tasas"]["por_region"].get(e["region"], art["tasas"]["nacional"])
        st.caption(f"Usa la tasa histórica de diagnóstico de {nombre_region(e['region'])} ({porcentaje(tasa)}), "
                   "calculada con años anteriores.")
    with c2.container(border=True):
        st.markdown("**Con un nivel de investigación de referencia**")
        st.metric("Probabilidad", porcentaje(res["prob_referencia"]), label_visibility="collapsed")
        st.markdown(f":{'blue' if res['nivel_referencia'] == 'Alto' else 'orange' if res['nivel_referencia'] == 'Medio' else 'gray'}"
                    f"-badge[Nivel {res['nivel_referencia'].lower()}]")
        st.progress(min(res["prob_referencia"], 1.0))
        st.caption(f"Como si la región tuviera la tasa histórica del percentil 75 nacional "
                   f"({porcentaje(art['referencia_tasa'])}).")

    dif = 100 * (res["prob_referencia"] - res["prob_real"])
    if dif >= 10:
        st.info(f"**Lectura:** la diferencia de {dif:.0f} puntos indica que el resultado depende en buena parte de "
                "los recursos de investigación de la región. Una toma de muestras oportuna podría aumentar la "
                "probabilidad de identificar el agente.")
    elif dif <= -10:
        st.info(f"**Lectura:** la región investiga sobre el nivel de referencia; sin esa capacidad, la "
                f"probabilidad sería {abs(dif):.0f} puntos menor.")
    else:
        st.info("**Lectura:** la probabilidad casi no depende de la capacidad de la región: refleja "
                "principalmente las características del brote.")

    c1, c2 = st.columns([2, 1])
    with c1.container(border=True):
        st.subheader("Factores que más influyeron (SHAP)")
        etiquetas = []
        for _, f in factores.iterrows():
            v = f["valor"]
            if f["variable"] == "Tasa histórica de la región":
                v = porcentaje(v)
            elif f["variable"] == "Región":
                v = nombre_region(v)
            elif f["variable"] == "Temporada estival":
                v = "sí" if v == 1 else "no"
            elif isinstance(v, float):
                v = "sin dato" if v != v else f"{v:.0f}"
            texto = f"{f['variable']}: {v}"
            etiquetas.append(texto if len(texto) <= 48 else texto[:46] + "…")
        puntos = 100 * factores["contribucion"].values
        fig, ax = plt.subplots(figsize=(7, 0.55 * len(puntos) + 0.6))
        ax.barh(range(len(puntos))[::-1], puntos, color=["#2a6f97" if p > 0 else "#e07a1f" for p in puntos])
        ax.set_yticks(range(len(puntos))[::-1], etiquetas)
        ax.axvline(0, color="black", lw=0.8)
        for i, p in zip(range(len(puntos))[::-1], puntos):
            ax.text(p, i, f" {p:+.1f} ", va="center", ha="left" if p > 0 else "right", fontsize=9)
        ax.set_xlabel("Puntos porcentuales sobre la probabilidad")
        ax.spines[["top", "right"]].set_visible(False)
        lim = max(abs(puntos).max() * 1.35, 1)
        ax.set_xlim(-lim, lim)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption(f"En azul, los factores que aumentan la probabilidad; en naranjo, los que la disminuyen. "
                   f"Parten de la probabilidad promedio de un brote ({porcentaje(base)}). "
                   "Contribuciones calculadas con SHAP (TreeExplainer) y expresadas en la escala calibrada. "
                   "Los datos sin registrar se reemplazan por la mediana del entrenamiento.")
    with c2.container(border=True):
        st.subheader("Niveles de prioridad")
        st.markdown(f"- :blue-badge[Alto] ≥ {porcentaje(art['niveles']['alto'])}\n"
                    f"- :orange-badge[Medio] {porcentaje(art['niveles']['medio'])} a {porcentaje(art['niveles']['alto'])}\n"
                    f"- :gray-badge[Bajo] < {porcentaje(art['niveles']['medio'])}")
        st.caption("El nivel alto corresponde al umbral que detecta al menos el 70% de los brotes con "
                   "diagnóstico específico (RNF05), elegido con validación cruzada temporal.")

    st.caption(f"Probabilidad calibrada para un brote que se confirme como ETA. Es un apoyo a la priorización "
               f"y no reemplaza el criterio del profesional a cargo. Calculada en {u['ms']:.0f} ms con el modelo "
               f"v{art['version']}.")
