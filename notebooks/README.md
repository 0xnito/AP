
## Modelos

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
