# Predicción del diagnóstico específico en brotes de ETA en Chile

Proyecto del curso **Aprendizaje de Máquinas (ACIF104)**, Universidad Andrés Bello.
Integrantes: Sandro Pizarro y Francisco López.

Modelo de clasificación que estima, **al momento de la notificación**, la probabilidad de que un brote de enfermedad transmitida por alimentos (ETA) llegue a **diagnóstico específico** (agente identificado). Busca apoyar a las SEREMI de Salud en la priorización de la investigación de laboratorio.

- **Datos:** base de brotes de ETA 2011–2025 por comuna del DEIS, Ministerio de Salud de Chile (16.856 brotes, 41 variables), incluida en `data/`.
- **Modelo final:** Random Forest con ponderación de clases, entrenado con brotes 2012–2022.
- **Maqueta de la aplicación:** https://franciscolopez27.github.io/acif104-brotes-eta/ (carpeta `docs/`).

## Resultados principales

Evaluación con brotes **no vistos** en el entrenamiento (prueba 2023–2025, 4.493 brotes, 17,7% con diagnóstico específico):

| Modelo | AUC-PR (IC 95%) | Exhaustividad | Precisión |
|---|---|---|---|
| **Random Forest (ponderación de clases)** | **0,499 (0,467–0,537)** | **0,715** | **0,429** |
| Gradient boosting (sin balanceo) | 0,490 (0,457–0,529) | 0,716 | 0,386 |
| MLP refinado (promedio de 5 semillas) | 0,445 (0,410–0,483) | 0,448 | 0,413 |
| Regresión logística (ponderación) | 0,422 (0,394–0,459) | 0,802 | 0,356 |

Línea base del AUC-PR (clasificador al azar): 0,177. El umbral de decisión (0,397) se eligió con validación cruzada temporal para lograr una exhaustividad ≥ 0,70 (requisito RNF05). La variable más influyente según SHAP es la tasa histórica de diagnóstico de la región.

## Estructura del repositorio

```
acif104-brotes-eta/
├── data/                     Base de datos del DEIS (Excel) y su descripción
├── src/                      Código reutilizable del proyecto
│   ├── config.py             Rutas, semilla, variables y estilo de gráficos
│   ├── datos.py              Carga, limpieza, variables, partición temporal y codificación
│   ├── balanceo.py           Ponderación de clases, SMOTE-NC y submuestreo
│   ├── modelos_ml.py         Regresión logística, Random Forest y gradient boosting
│   ├── redes.py              MLP, MLP con embeddings y Transformer tabular (PyTorch)
│   └── evaluacion.py         Métricas, umbral para recall ≥ 0,70 e intervalos por bootstrap
├── notebooks/
│   ├── 01_analisis_exploratorio.ipynb       Completitud, correctitud, atípicos, tendencias y partición
│   ├── 02_modelos_iniciales.ipynb           3 técnicas de ML y 3 redes × 4 escenarios de balanceo
│   └── 03_refinamiento_y_evaluacion.ipynb   Semillas, IC, refinamiento del MLP, umbral, sesgo de selección, SHAP
├── results/
│   ├── figuras/              Figuras generadas por los notebooks (PNG)
│   └── tablas/               Tablas generadas por los notebooks (CSV, prefijo = notebook)
├── docs/                     Maqueta navegable de la aplicación (GitHub Pages)
└── requirements.txt          Dependencias
```

## Instalación y ejecución

Los notebooks se ejecutan **en orden** (01 → 02 → 03). Cada uno guarda sus figuras y tablas en `results/`.

### Opción A: Google Colab (recomendada)

1. En Colab: **Archivo → Abrir cuaderno → GitHub**, escribir `FranciscoLopez27/acif104-brotes-eta` y elegir un notebook de `notebooks/`.
2. **Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU T4** (necesario para los notebooks 02 y 03).
3. **Entorno de ejecución → Ejecutar todas.** La primera celda clona el repositorio e instala lo que falta (`imbalanced-learn` y `shap`).

### Opción B: entorno local (VS Code o Jupyter)

Requiere Python 3.10 o superior.

```bash
git clone https://github.com/FranciscoLopez27/acif104-brotes-eta.git
cd acif104-brotes-eta
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

**GPU NVIDIA (opcional):** para entrenar las redes con CUDA, instalar PyTorch con el comando que indica https://pytorch.org/get-started/locally/ según la versión de CUDA (se revisa con `nvidia-smi`). Sin GPU, las redes se entrenan en CPU, más lento.

Luego abrir los notebooks en VS Code (extensión Jupyter, kernel `.venv`) o con `jupyter notebook` y ejecutar todas las celdas. La primera celda detecta el entorno local y ajusta la carpeta de trabajo.

### Tiempos aproximados (GPU T4 de Colab)

| Notebook | Tiempo |
|---|---|
| 01 Análisis exploratorio | < 1 minuto |
| 02 Modelos iniciales | 5–10 minutos |
| 03 Refinamiento y evaluación | 20–30 minutos |

## Reproducibilidad

- Semilla fija (`SEMILLA = 42` en `src/config.py`); los experimentos de robustez usan 5 semillas.
- Los modelos de ML son reproducibles exactamente. Las redes en GPU pueden variar en el tercer decimal porque algunas operaciones de CUDA no son deterministas.
- Partición temporal: entrenamiento 2012–2020, validación 2021–2022 y prueba 2023–2025. Toda transformación (codificación, imputación, agrupación de categorías, tasa histórica regional) se ajusta solo con años anteriores para evitar la fuga de información.

## Uso de los módulos

```python
from src import datos as D, balanceo as B, modelos_ml as ML, evaluacion as E
from src.config import CAT, NUM

datos = D.preparar(D.cargar())                       # brotes confirmados con diagnóstico
p = D.particionar(datos, fin_entrenamiento=2022, fin_validacion=2022)
cod = D.Codificador().fit(p["tr"][CAT + NUM])
M_tr, M_te = cod.transform(p["tr"][CAT + NUM]), cod.transform(p["te"][CAT + NUM])
Mb, yb, w = B.balancear(M_tr, p["tr"].y.values, "Ponderación de clases", cod)
rf = ML.ajustar(ML.crear_modelo("Random Forest", {"max_depth": None, "min_samples_leaf": 5}, cod), Mb, yb, w)
prob = ML.probabilidad(rf, M_te)
print(E.metricas(p["te"].y.values, prob, umbral=0.397))
```

## Fuente de los datos

Departamento de Estadísticas e Información de Salud. (2026). *Base de datos de brotes de enfermedades transmitidas por alimentos (ETA) 2011–2025, por comuna* [Conjunto de datos]. Ministerio de Salud de Chile. http://epi.minsal.cl/brotes-eta-basedatos/

Los datos son públicos y no contienen información personal.
