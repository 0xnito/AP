"""GRU Apilado Mejorado para prediccion semanal/diaria de tweets.
Entrada: (seq_len, n_features).

Arquitectura: GRU(64, ret=True) -> Dropout -> GRU(32) -> BN -> Dropout -> Dense(16, relu) -> Dense(1).

Mejoras respecto a la version base:
  - 2 capas GRU apiladas (stacked) en vez de 1
  - Mayor numero de unidades: 64 en la primera, 32 en la segunda
  - BatchNormalization tras la segunda GRU para estabilidad
  - Capa Dense(16) adicional antes de la salida
  - L2 regularizacion (0.001) en ambas capas GRU
  - Learning rate 5e-4
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2


def build_gru_model_mejorado(seq_len: int, n_features: int = 1,
                               gru_units_1: int = 64, gru_units_2: int = 32,
                               dropout: float = 0.3,
                               learning_rate: float = 5e-4) -> tf.keras.Model:
    """
    Parameters
    ----------
    seq_len : int
        Longitud de la secuencia.
    n_features : int
        Numero de features por timestep.
    gru_units_1 : int
        Unidades de la primera capa GRU.
    gru_units_2 : int
        Unidades de la segunda capa GRU.
    dropout : float
        Tasa de Dropout.
    learning_rate : float
        Learning rate de Adam.
    """
    model = Sequential([
        GRU(gru_units_1, return_sequences=True,
            kernel_regularizer=l2(0.001),
            input_shape=(seq_len, n_features)),
        Dropout(dropout),
        GRU(gru_units_2, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(dropout),
        Dense(16, activation="relu"),
        Dense(1),
    ], name="GRU_Apilado_Mejorado")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def count_parameters(model: tf.keras.Model) -> int:
    return model.count_params()
