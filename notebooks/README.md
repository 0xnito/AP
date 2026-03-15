AP/
├── data/
│   └── raw/                       # CSV original (no versionado)
├── models/                        # Definiciones de modelos (.py)
│   ├── lstm_model.py
│   ├── gru_model.py
│   ├── lstm_gru_model.py          # BiLSTM + GRU
│   
├── notebooks/
│   ├── 01_eda.ipynb               # Análisis exploratorio
│   ├── 02_lstm_weekly_tweets.ipynb
│   ├── 03_gru_weekly_tweets.ipynb
│   ├── 04_bilstm_gru_weekly_tweets.ipynb
│   
├── saved_models/                  # Modelos entrenados (.keras)
└── README.md

## Modelos deep learning
Métricas en escala original (tweets/semana). Split cronológico: 70% train / 15% val / 15% test. Ventana de entrada: 8 semanas.
| Modelo | Parámetros | Train RMSE | Val RMSE | Test RMSE | Test MAE | Test R² | Test MAPE (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| LSTM | ~21 K | 20.91 | 50.68 | 102.75 | 79.11 | 0.4113 | 26.67 |
| GRU | ~16 K | 22.27 | 45.72 | 113.78 | 84.53 | 0.2781 | 26.39 |
| BiLSTM+GRU | ~43 K | 19.66 | 47.34 | 109.89 | 83.37 | 0.3266 | 26.28 |

--------------------------------------------------------------------------------------------------------------------------------------

## Modelos simples

| Modelo | RMSE Train | RMSE Val | RMSE Test | Nº Parámetros |
|--------|:----------:|:--------:|:---------:|:-------------:|
| Regresión Lineal | 19.28 | 44.10 | 104.73 | — |
| Random Forest (100 árboles, max_depth=5) | 14.31 | 69.55 | 245.41 | — |
| Red Neuronal  (1 oculta, 4 neuronas) | 18.85 | 53.49 | 146.9 | 25 |


## División de datos

| Conjunto | Proporción |
|----------|-----------|
| Train | 70 % |
| Validación | 15 % |
| Test | 15 % |

## Red Neuronal Mínima

Arquitectura definida en `models/simple_nn.py`:

```
Input(1) → Linear(1→4) → ReLU → Linear(4→1) → Output(1)
```

Parámetros: `(1×4 + 4) + (4×1 + 1) = 8 + 5 = **13 parámetros**`

## Métrica

Se utiliza **RMSE**  en escala original (número de tweets).
