# Predicción de actividad semanal en X

Este proyecto estudia y modela la actividad de publicación semanal de una cuenta influyente en la red social X (Twitter), con el objetivo de predecir el número de tweets que se publicarán en una semana futura a partir del historial de actividad pasada.

El trabajo se plantea como un proyecto largo, dividido en distintas fases.

---

## 1. Planteamiento del problema

Se plantea un problema de **aprendizaje supervisado de regresión**.

- **Entrada (X):** número de tweets publicados por semana en el pasado.
- **Salida (y):** número de tweets publicados en la semana siguiente.

El objetivo es aprender patrones temporales en la actividad de publicación que permitan realizar predicciones a corto plazo.

---

## 2. Datos

Se utiliza un dataset público que contiene el histórico de publicaciones de la cuenta analizada desde 2010 hasta 2025.

Para esta primera entrega:
- Se utilizan únicamente tweets y retweets.
- Los quote tweets se dejan como posible extensión en fases posteriores del proyecto.

Los datos originales se encuentran en la carpeta:
data/raw/


---

## 3. Análisis exploratorio de datos (EDA)

El notebook `01_eda.ipynb` contiene un análisis exploratorio que incluye:
- inspección del tamaño y estructura del dataset,
- conversión de la variable temporal,
- agregación del número de tweets por semana,
- estadísticas descriptivas,
- visualización de la serie temporal resultante.

Este análisis permite validar la viabilidad del problema y detectar patrones temporales relevantes.

---

## 4. Métricas de evaluación

Al tratarse de un problema de regresión, se utilizarán las siguientes métricas:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

Estas métricas permiten evaluar el error de predicción en términos absolutos y cuadráticos.

---

## 5. Estado del arte

La predicción de series temporales ha sido ampliamente estudiada en la literatura. A continuación se resumen algunos modelos comúnmente utilizados para problemas similares, junto con las métricas empleadas y resultados típicos reportados.

| Modelo | Tipo | Métrica | Resultado reportado |
|------|------|---------|---------------------|
| ARIMA | Estadístico | RMSE | Dependiente de la serie |
| Prophet | Estadístico | MAE | Buen desempeño con estacionalidad |
| LSTM | Deep Learning | MAE | Mejora frente a modelos base |
| GRU | Deep Learning | RMSE | Resultados competitivos |

Los resultados mostrados corresponden a valores reportados en trabajos previos de la literatura y no a experimentos propios realizados en esta fase del proyecto.

---

## 6. Estructura del proyecto

AP/
├── data/
│ ├── raw/ # Datos originales
│ └── processed/ # Datos procesados
├── notebooks/ # Notebooks de análisis
├── src/ # Código fuente
├── results/ # Resultados y figuras
├── README.md
└── requirements.txt
