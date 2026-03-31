"""GRU para prediccion diaria de tweets.
Entrada: (window_size=30, n_features=11).
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2


def build_gru_model(window_size: int, n_features: int) -> tf.keras.Model:
    """
    Parameters
    ----------
    window_size : int
        Numero de timesteps en la ventana.
    n_features : int
        Numero de features por timestep.
    """
    model = Sequential([
        GRU(48, return_sequences=False, kernel_regularizer=l2(0.001),
            input_shape=(window_size, n_features)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(24, activation="relu"),
        Dropout(0.2),
        Dense(1),
    ], name="GRU_Diario")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )
    return model


def count_parameters(model: tf.keras.Model) -> int:
    return model.count_params()
