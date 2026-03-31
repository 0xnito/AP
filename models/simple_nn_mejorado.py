"""Red Neuronal Mejorada para prediccion semanal/diaria de tweets.

Arquitectura: Dense(64,relu) -> BN -> Drop(0.2) -> Dense(32,relu) -> BN -> Drop(0.2) -> Dense(16,relu) -> Dense(1).

Mejoras respecto a la version base:
  - 3 capas ocultas en vez de 1 (64 -> 32 -> 16)
  - BatchNormalization para estabilizar el entrenamiento
  - Dropout(0.2) para regularizacion
  - Learning rate reducido a 0.001 con Adam
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization


def build_model(n_lags: int = 6) -> keras.Model:
    """
    Parameters
    ----------
    n_lags : int
        Numero de lags/features de entrada.
    """
    model = keras.Sequential([
        Dense(64, activation="relu", input_shape=(n_lags,), name="hidden_1"),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation="relu", name="hidden_2"),
        BatchNormalization(),
        Dropout(0.2),
        Dense(16, activation="relu", name="hidden_3"),
        Dense(1, activation="linear", name="output"),
    ], name="SimpleNN_Mejorado")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )
    return model


def count_parameters(model: keras.Model) -> int:
    return model.count_params()
