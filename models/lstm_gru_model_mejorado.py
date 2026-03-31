"""BiLSTM + GRU Mejorado para prediccion semanal/diaria de tweets.
Entrada: (seq_len, n_features).

Arquitectura: BiLSTM(64, ret=True) -> GRU(64) -> BN -> Dropout -> Dense(32, relu) -> Dense(1).

Mejoras respecto a la version base:
  - Mayor numero de unidades: 64 en vez de 32 (tanto BiLSTM como GRU)
  - BatchNormalization tras GRU para estabilidad de gradientes
  - Capa Dense(32) adicional antes de la salida
  - L2 regularizacion (0.001) en ambas capas recurrentes
  - Learning rate 5e-4
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2


def build_lstm_gru_model_mejorado(seq_len: int, n_features: int = 1,
                                    units: int = 64, dropout: float = 0.3,
                                    learning_rate: float = 5e-4) -> tf.keras.Model:
    """
    Parameters
    ----------
    seq_len : int
        Longitud de la secuencia.
    n_features : int
        Numero de features por timestep.
    units : int
        Unidades en BiLSTM y GRU.
    dropout : float
        Tasa de Dropout.
    learning_rate : float
        Learning rate de Adam.
    """
    model = Sequential([
        Bidirectional(
            LSTM(units, return_sequences=True, kernel_regularizer=l2(0.001)),
            input_shape=(seq_len, n_features),
        ),
        GRU(units, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(dropout),
        Dense(32, activation="relu"),
        Dense(1),
    ], name="BiLSTM_GRU_Mejorado")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def count_parameters(model: tf.keras.Model) -> int:
    return model.count_params()
