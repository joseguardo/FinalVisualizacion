# Guía de Librerías de Python para Ciencia de Datos

## 📊 Análisis y Manipulación de Datos

### NumPy
**Qué es:** Librería fundamental para computación científica en Python.

**Cuándo usarla:**
- Necesitas trabajar con arrays multidimensionales
- Requieres operaciones matemáticas rápidas
- Trabajas con álgebra lineal, transformadas de Fourier o números aleatorios

**Para qué:**
- Crear y manipular arrays numéricos eficientemente
- Realizar operaciones matemáticas vectorizadas
- Base para otras librerías como Pandas y Scikit-learn

```python
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
promedio = np.mean(arr)
```

---

### Pandas
**Qué es:** Librería para análisis y manipulación de datos estructurados.

**Cuándo usarla:**
- Trabajas con datos tabulares (CSV, Excel, bases de datos)
- Necesitas limpiar, transformar o analizar datasets
- Requieres operaciones de groupby, merge o pivot

**Para qué:**
- Leer/escribir datos en diversos formatos
- Filtrar, agrupar y transformar datos
- Análisis exploratorio de datos (EDA)
- Manejar datos faltantes

```python
import pandas as pd
df = pd.read_csv('datos.csv')
resumen = df.groupby('categoria').mean()
```

---

## 📈 Visualización de Datos

### Seaborn
**Qué es:** Librería de visualización estadística basada en Matplotlib.

**Cuándo usarla:**
- Necesitas gráficos estadísticos rápidos y elegantes
- Quieres visualizar distribuciones, correlaciones o relaciones estadísticas
- Trabajas con DataFrames de Pandas y quieres gráficos automáticos
- Prefieres gráficos estáticos de alta calidad para publicaciones

**Para qué:**
- Crear gráficos estadísticos con estilo profesional
- Visualizar distribuciones (histogramas, boxplots, violin plots)
- Matrices de correlación y heatmaps
- Gráficos de regresión con intervalos de confianza
- Pairplots para análisis multivariable

```python
import seaborn as sns
sns.set_style("whitegrid")
sns.scatterplot(data=df, x='variable_x', y='variable_y', hue='categoria')
sns.heatmap(df.corr(), annot=True)
```

**Comparación con Plotly:**
- **Seaborn:** Gráficos estáticos, más rápido para análisis exploratorio, mejor integración con análisis estadístico
- **Plotly:** Gráficos interactivos, ideal para dashboards y presentaciones web

---

### Plotly Graph Objects
**Qué es:** Módulo de bajo nivel de Plotly para crear gráficos interactivos personalizados.

**Cuándo usarla:**
- Necesitas control total sobre cada elemento del gráfico
- Quieres gráficos complejos o altamente personalizados
- Requieres interactividad avanzada

**Para qué:**
- Crear gráficos interactivos detallados
- Personalizar cada aspecto visual
- Combinar múltiples tipos de gráficos

```python
import plotly.graph_objects as go
fig = go.Figure(data=go.Scatter(x=[1,2,3], y=[4,5,6]))
fig.show()
```

---

### Plotly Express
**Qué es:** API de alto nivel de Plotly para crear gráficos rápidamente.

**Cuándo usarla:**
- Quieres crear gráficos interactivos con pocas líneas de código
- Necesitas visualizaciones estándar (scatter, line, bar, etc.)
- Trabajas directamente con DataFrames de Pandas

**Para qué:**
- Crear visualizaciones rápidas y elegantes
- Exploración visual de datos
- Gráficos interactivos con mínimo código

```python
import plotly.express as px
fig = px.scatter(df, x='variable_x', y='variable_y', color='categoria')
```

---

## 🖥️ Dashboards Web

### Dash
**Qué es:** Framework para crear aplicaciones web analíticas interactivas.

**Cuándo usarla:**
- Necesitas crear dashboards interactivos
- Quieres compartir análisis a través de una web
- Requieres actualizaciones dinámicas basadas en inputs del usuario

**Para qué:**
- Construir aplicaciones web de ciencia de datos
- Crear interfaces para modelos de ML
- Dashboards empresariales interactivos

```python
import dash
from dash import dcc, html

app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Graph(figure=fig),
    html.H1('Mi Dashboard')
])
```

---

## 📉 Modelado Estadístico

### Statsmodels
**Qué es:** Librería para estimación de modelos estadísticos y pruebas.

**Cuándo usarla:**
- Necesitas modelos estadísticos clásicos (regresión, ANOVA, etc.)
- Requieres pruebas de hipótesis y diagnósticos estadísticos
- Trabajas con series temporales (ARIMA, SARIMAX)

**Para qué:**
- Regresión lineal con estadísticas detalladas
- Análisis de series temporales
- Pruebas estadísticas inferenciales
- Obtener p-values, intervalos de confianza, etc.

```python
import statsmodels.api as sm
X = sm.add_constant(X)
modelo = sm.OLS(y, X).fit()
print(modelo.summary())
```

---

## 🤖 Machine Learning (Scikit-learn)

### Preparación de Datos

#### train_test_split
**Para qué:** Dividir datos en conjuntos de entrenamiento y prueba.
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

#### StandardScaler
**Para qué:** Normalizar/estandarizar características (media=0, std=1).
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
```

---

### Modelos de Regresión

#### LinearRegression
**Cuándo:** Predicción de valores continuos con relación lineal.
**Para qué:** Regresión lineal simple o múltiple.

#### ElasticNet
**Cuándo:** Regresión con regularización (combina L1 y L2).
**Para qué:** Prevenir overfitting y selección de características.

#### RandomForestRegressor
**Cuándo:** Relaciones no lineales complejas en regresión.
**Para qué:** Predicción robusta con ensemble de árboles de decisión.

```python
from sklearn.ensemble import RandomForestRegressor
modelo = RandomForestRegressor(n_estimators=100)
modelo.fit(X_train, y_train)
predicciones = modelo.predict(X_test)
```

---

### Modelos de Clasificación

#### LinearSVC
**Cuándo:** Clasificación binaria o multiclase con datos linealmente separables.
**Para qué:** Support Vector Classification rápida y eficiente.

#### RandomForestClassifier
**Cuándo:** Clasificación con relaciones complejas no lineales.
**Para qué:** Clasificación robusta con ensemble de árboles.

```python
from sklearn.ensemble import RandomForestClassifier
clasificador = RandomForestClassifier()
clasificador.fit(X_train, y_train)
y_pred = clasificador.predict(X_test)
```

---

### Modelos de Clustering

#### KMeans
**Cuándo:** Agrupar datos en K clusters predefinidos.
**Para qué:** Segmentación de clientes, compresión de imágenes, etc.

#### DBSCAN
**Cuándo:** Clusters de forma arbitraria y detección de outliers.
**Para qué:** Clustering basado en densidad sin especificar número de clusters.

```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(X)
```

---

### Otros Algoritmos

#### NearestNeighbors
**Cuándo:** Encontrar los K vecinos más cercanos.
**Para qué:** Sistemas de recomendación, detección de anomalías, preprocesamiento para otros algoritmos.

---

### Métricas de Evaluación

#### Regresión
- **r2_score:** Coeficiente de determinación (0-1, mejor=1)
- **mean_absolute_error:** Error absoluto promedio
- **mean_squared_error:** Error cuadrático medio

```python
from sklearn.metrics import r2_score, mean_absolute_error
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
```

#### Clasificación
- **classification_report:** Precision, recall, F1-score por clase
- **confusion_matrix:** Matriz de confusión para ver errores de clasificación

```python
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, y_pred))
matriz = confusion_matrix(y_test, y_pred)
```

#### Clustering
- **silhouette_score:** Calidad de los clusters (-1 a 1, mejor=1)

```python
from sklearn.metrics import silhouette_score
score = silhouette_score(X, labels)
```

---

## 🎯 Flujo de Trabajo Típico

1. **Importar y explorar datos:** Pandas, NumPy
2. **Visualizar:** Seaborn (análisis exploratorio), Plotly Express/Graph Objects (interactividad)
3. **Preprocesar:** StandardScaler, train_test_split
4. **Modelar:** Elegir algoritmo según el problema
5. **Evaluar:** Métricas apropiadas
6. **Presentar:** Dash para dashboards interactivos

---

## 💡 Consejos

- **NumPy + Pandas:** Siempre juntas para manipulación de datos
- **Seaborn:** Ideal para análisis exploratorio rápido con gráficos estadísticos
- **Plotly Express:** Inicio rápido, Graph Objects para control fino
- **Seaborn vs Plotly:** Usa Seaborn para análisis estático, Plotly para interactividad
- **Scikit-learn:** Ecosistema completo para ML
- **Statsmodels:** Cuando necesitas estadísticas detalladas
- **Dash:** Ideal para compartir resultados con no-programadores