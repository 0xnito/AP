# Predicción de Actividad en X — Elon Musk

Predicción del número de tweets semanales y diarios de Elon Musk usando modelos de Machine Learning y Deep Learning.

---

## Descripción del problema

Problema de **regresión supervisada sobre series temporales**.

- **Entrada (X):** ventana de semanas/días pasados de actividad
- **Salida (y):** número de tweets de la siguiente semana/día
- **Estrategia:** one-step-ahead forecasting con ventana deslizante
- **Periodo:** 2018–2025 (ambas granularidades)

---

## Dataset

Dataset público de Kaggle con el histórico de publicaciones de Elon Musk (2010–2025).

- **Fuente:** [Elon Musk Tweets 2010–2025](https://www.kaggle.com/datasets/dadalyndell/elon-musk-tweets-2010-to-2025-march)
- **Ruta local:** `data/raw/all_musk_posts.csv`
- **Serie semanal:** 380 semanas (2018–2025)
- **Serie diaria:** 2,660 días (2018–2025)

### Características clave

| Característica | Semanal | Diario |
|---|---|---|
| Media actividad | 134.4 tweets/sem | 19.2 tweets/día |
| Máximo | 671 tweets/sem | 158 tweets/día |
| Media pre-adquisición (2018–2022) | 40.9 tweets/sem | ~6 tweets/día |
| Media post-adquisición (2022–2025) | 280.7 tweets/sem | ~44 tweets/día |
| Estacionariedad | No (ADF p=0.9954) | No estacionaria |

> **Cambio estructural:** La adquisición de Twitter en octubre 2022 multiplica la actividad por 3–4x, siendo el principal reto del proyecto.

---

## Estructura del proyecto

```
AP/
├── data/
│   └── raw/                              # Dataset original
├── notebooks/
│   ├── 01_eda.ipynb                      # Análisis exploratorio
│   ├── 02_models.ipynb                   # Modelos simples baseline
│   ├── 02_lstm_weekly_tweets.ipynb       # LSTM semanal base
│   ├── 03_gru_weekly_tweets.ipynb        # GRU semanal base
│   ├── 04_bilstm_gru_weekly_tweets.ipynb # BiLSTM+GRU semanal base
│   ├── 05_lstm_regime_variable.ipynb     # Experimento variable régimen
│   ├── SEMANAL_Mejorado.ipynb            # Todos los modelos semanales (versión final)
│   └── DIARIO_Mejorado.ipynb             # Todos los modelos diarios (versión final)
├── models/
│   ├── linear_regression.py
│   ├── random_forest.py
│   ├── simple_nn_mejorado.py
│   ├── lstm_model_mejorado.py
│   ├── gru_model_mejorado.py
│   ├── lstm_gru_model_mejorado.py
│   └── simple_nn.py
├── figures/                              # Visualizaciones generadas
├── README.md
└── requirements.txt
```

---

## Métricas de evaluación

- **MAE** (Mean Absolute Error) — error medio en tweets/semana o tweets/día
- **RMSE** (Root Mean Squared Error) — penaliza errores grandes
- **R²** — proporción de varianza explicada (1=perfecto, 0=naive, <0=peor que naive)

División cronológica: **70% train / 15% validación / 15% test**

---

## Resultados — Granularidad Semanal

Ventana de entrada: **16 semanas** | 1 feature (tweet_count normalizado)

| Modelo | RMSE Train | RMSE Val | RMSE Test | MAE Test | R² Test | Params |
|---|---|---|---|---|---|---|
| **Regresión Lineal** ✅ | 29.46 | 76.55 | **120.23** | **95.72** | **+0.246** | 7 |
| LSTM Apilado | 29.82 | 72.40 | 133.47 | 113.79 | +0.004 | 118,337 |
| BiLSTM+GRU | 29.47 | 71.62 | 134.36 | 113.85 | -0.009 | 73,409 |
| GRU Apilado | 33.62 | 76.02 | 134.58 | 113.65 | -0.013 | 22,945 |
| Red Neuronal Simple | 16.38 | 85.30 | 160.02 | 125.52 | -0.335 | 3,457 |
| Random Forest | 14.09 | 76.02 | 206.74 | 159.53 | -1.229 | 56,456 |
| LSTM + Régimen | 28.84 | 78.72 | 245.17 | 208.24 | -2.361 | 118,849 |

> ✅ **Mejor modelo semanal:** Regresión Lineal (RMSE=120.23, R²=+0.246)

---

## Resultados — Granularidad Diaria

Ventana de entrada: **30 días** | 11 features (tweet_log + features temporales)

| Modelo | RMSE Train | RMSE Val | RMSE Test | MAE Test | R² Test | Params |
|---|---|---|---|---|---|---|
| **GRU Apilado** ✅ | 8.13 | 19.36 | **26.77** | **20.68** | **+0.149** | 24,865 |
| Red Neuronal Simple | 7.86 | 20.22 | 28.41 | 20.98 | +0.035 | 3,969 |
| BiLSTM+GRU | 7.97 | 18.57 | 28.72 | 22.23 | +0.021 | 78,529 |
| LSTM Apilado | 7.91 | 18.99 | 29.01 | 21.32 | +0.001 | 123,457 |
| LSTM + Régimen | 8.04 | 19.89 | 33.00 | 24.00 | -0.293 | 123,969 |
| Regresión Lineal | 8.34 | 21.54 | 34.00 | 25.19 | -0.383 | 15 |
| Random Forest | 5.67 | 21.26 | 37.88 | 28.10 | -0.716 | 198,822 |

> ✅ **Mejor modelo diario:** GRU Apilado (RMSE=26.77, R²=+0.149)

---

## Comparativa Semanal vs Diario

| Modelo | Ranking Semanal | R² Semanal | Ranking Diario | R² Diario |
|---|---|---|---|---|
| Regresión Lineal | #1 | +0.246 | #5 | -0.383 |
| LSTM Apilado | #2 | +0.004 | #4 | +0.001 |
| BiLSTM+GRU | #3 | -0.009 | #3 | +0.021 |
| GRU Apilado | #4 | -0.013 | #1 | +0.149 |
| Red Neuronal Simple | #5 | -0.335 | #2 | +0.035 |
| Random Forest | #6 | -1.229 | #7 | -0.716 |
| LSTM + Régimen | #7 | -2.361 | #6 | -0.293 |

---

## Arquitecturas

### Modelos Simples

| Modelo | Arquitectura | Lags Semanal | Lags Diario |
|---|---|---|---|
| Regresión Lineal | OLS estándar | 6 lags | 14 lags (log) |
| Random Forest | 300 est., max_depth=10, L2 | 6 lags | 14 lags (log) |
| Red Neuronal Simple | Dense(64→32→16) + BN + Dropout(0.2) | 6 lags | 14 lags (log) |

### Modelos Deep Learning

| Modelo | Arquitectura |
|---|---|
| LSTM Apilado | LSTM(128, ret) → Dropout(0.3) → LSTM(64) → BN → Dense(32) → Dense(1) |
| GRU Apilado | GRU(64, ret) → Dropout(0.3) → GRU(32) → BN → Dense(16) → Dense(1) |
| BiLSTM+GRU | BiLSTM(64, ret) → GRU(64) → BN → Dropout(0.3) → Dense(32) → Dense(1) |
| LSTM+Régimen | LSTM apilado + feature binaria `post_acquisition` |

**Hiperparámetros compartidos:** lr=5e-4, EarlyStopping patience=60, ReduceLROnPlateau factor=0.5/patience=15, L2=0.001, scaler ajustado solo sobre train.

---

## Conclusiones

1. **El cambio estructural de 2022 es el principal limitante.** La actividad pasa de ~41 a ~281 tweets/semana. Ningún modelo entrenado con datos pre-2022 puede anticipar completamente este salto.

2. **Semanal: gana la Regresión Lineal** (RMSE=120.23, R²=+0.246). Los tres modelos RNN empatan ~RMSE=134 — la complejidad adicional no aporta con una sola feature.

3. **Diario: gana el GRU Apilado** (RMSE=26.77, R²=+0.149). Las 11 features temporales hacen que todos los modelos RNN obtengan R² positivo en test.

4. **GRU: mejor eficiencia paramétrica.** Con ~23k parámetros queda #4 en semanal y #1 en diario, superando al LSTM que tiene 5x más parámetros.

5. **La variable de régimen empeora en ambas granularidades.** Semanal: RMSE 133→245 (+84%). Diario: RMSE 29→33 (+14%). La varianza intrínseca del periodo post-2022 no se captura con un flag binario.

---

## Instalación

```bash
pip install -r requirements.txt
```

**Dependencias principales:** `tensorflow==2.15.0`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `statsmodels`
