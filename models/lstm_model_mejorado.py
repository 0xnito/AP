"""LSTM Apilado Mejorado para prediccion semanal/diaria de tweets.
Entrada: (seq_len, n_features).
Soporta 1 feature (univariante) o multiples features.

Arquitectura: LSTM(128, ret=True) -> Dropout -> LSTM(64) -> BN -> Dropout -> Dense(32, relu) -> Dense(1).

Mejoras respecto a la version base:
  - 2 capas LSTM apiladas (stacked)
  - Mayor numero de unidades: 128 en la primera capa, 64 en la segunda
  - BatchNormalization tras la segunda LSTM para estabilidad
  - Capa Dense(32) adicional antes de la salida
  - L2 regularizacion (0.001) en ambas capas LSTM
  - Learning rate 5e-4 (mas conservador que 1e-3)
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2


def build_lstm_model_mejorado(seq_len: int, n_features: int = 1,
                               lstm_units_1: int = 128, lstm_units_2: int = 64,
                               dropout: float = 0.3,
                               learning_rate: float = 5e-4) -> tf.keras.Model:
    """
    Parameters
    ----------
    seq_len : int
        Longitud de la secuencia (numero de pasos temporales).
    n_features : int
        Numero de features por timestep.
    lstm_units_1 : int
        Unidades de la primera capa LSTM.
    lstm_units_2 : int
        Unidades de la segunda capa LSTM.
    dropout : float
        Tasa de Dropout.
    learning_rate : float
        Learning rate de Adam.
    """
    model = Sequential([
        LSTM(lstm_units_1, return_sequences=True,
             kernel_regularizer=l2(0.001),
             input_shape=(seq_len, n_features)),
        Dropout(dropout),
        LSTM(lstm_units_2, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(dropout),
        Dense(32, activation="relu"),
        Dense(1),
    ], name="LSTM_Apilado_Mejorado")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def count_parameters(model: tf.keras.Model) -> int:
    return model.count_params()
