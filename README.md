# Predicción de actividad semanal en X

Este proyecto estudia y modela la actividad de publicación semanal de Elon Musk
en X (Twitter), con el objetivo de predecir el número de tweets de la semana siguiente
a partir del historial de actividad pasada.

---

## 1. Planteamiento del problema

Problema de **aprendizaje supervisado de regresión sobre series temporales**.

- **Entrada (X):** ventana de semanas pasadas de actividad (tweet_count)
- **Salida (y):** número de tweets de la semana siguiente
- **Estrategia:** one-step-ahead forecasting con ventana deslizante

---

## 2. Datos

Dataset público de Kaggle con el histórico de publicaciones de Elon Musk (2010–2025).

- **Fuente:** [Elon Musk Tweets 2010–2025](https://www.kaggle.com/datasets/dadalyndell/elon-musk-tweets-2010-to-2025-march)
- **Ruta local:** `data/raw/all_musk_posts.csv`
- **Serie resultante:** 680 semanas (2010–2025), filtrada desde 2018 para modelado
- **Período de modelado:** 2018–2025 (380 semanas)

### Características clave de la serie
- No estacionaria (ADF p-value = 0.9954)
- Cambio estructural en 2022 (adquisición de Twitter)
- Media: 81 tweets/semana | Máximo: 671 tweets/semana
- Distribución fuertemente sesgada a la derecha

---

## 3. Métricas de evaluación

- **MAE** (Mean Absolute Error) — error medio en tweets/semana
- **RMSE** (Root Mean Squared Error) — penaliza errores grandes

---

## 4. Estructura del proyecto
```
AP/
├── data/
│   └── raw/                          # Dataset original
├── notebooks/
│   ├── 01_eda.ipynb                  # Análisis exploratorio
│   ├── 02_models.ipynb               # Modelos simples (baseline)
│   ├── 02_lstm_weekly_tweets.ipynb   # Modelo LSTM
│   ├── 03_gru_weekly_tweets.ipynb    # Modelo GRU
│   └── 04_bilstm_gru_weekly_tweets.ipynb  # Modelo BiLSTM+GRU
├── models/
│   ├── simple_nn.py                  # Red neuronal simple
│   ├── lstm_model.py                 # Arquitectura LSTM
│   ├── gru_model.py                  # Arquitectura GRU
│   └── lstm_gru_model.py             # Arquitectura BiLSTM+GRU
├── figures/                          # Visualizaciones generadas
├── README.md
└── requirements.txt
```

---

## 5. Resultados

Todos los modelos se evalúan sobre el mismo período de test (2024–2025).
División cronológica: 70% train / 15% val / 15% test.

### Entrega 02 — Modelos simples (datos desde 2018, ventana 4 semanas)

| Modelo | RMSE Test | MAE Test | Parámetros |
|---|---|---|---|
| Regresión Lineal | 122.43 | 97.40 | — |
| Random Forest | 205.65 | 158.41 | — |
| Red Neuronal Simple | 291.03 | 255.97 | 13 |

### Entrega 03 — Deep Learning (datos desde 2018, ventana 12 semanas)

| Modelo | RMSE Test | MAE Test | Parámetros |
|---|---|---|---|
| LSTM | 198.58 | 159.90 | 17,953 |
| BiLSTM+GRU | 234.57 | 199.36 | 12,929 |
| GRU | 275.44 | 242.71 | 3,905 |

### Conclusión global

La **Regresión Lineal** obtiene el mejor resultado en test (RMSE=122.43),
superando a todos los modelos de deep learning. El factor limitante no es
la capacidad del modelo sino el **cambio estructural de la serie en 2022**
(adquisición de Twitter), que introduce un régimen de actividad muy diferente
al período de entrenamiento. En este contexto, la capacidad de extrapolación
lineal resulta más ventajosa que la complejidad de las arquitecturas recurrentes.

---

## 6. Análisis exploratorio (EDA)

El notebook `01_eda.ipynb` incluye:
- Inspección del dataset (55,099 tweets, 24 columnas)
- Agregación semanal y visualización de la serie temporal
- Test de estacionariedad (ADF) y análisis ACF/PACF
- Detección de gaps (96 semanas, concentradas en 2010–2011)
- Detección de outliers (90 semanas, 13.2%, todas post-2022)

---

## 7. Requisitos
```bash
pip install -r requirements.txt
```

Dependencias principales: `tensorflow==2.15.0`, `scikit-learn`, `pandas`,
`numpy`, `matplotlib`, `statsmodels`.
