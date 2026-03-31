"""CNN-LSTM para prediccion diaria de tweets.
Entrada: (window_size=30, n_features=11).
Conv1D extrae patrones locales; LSTM captura dependencias temporales.
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2


def build_cnn_lstm_model(window_size: int, n_features: int) -> tf.keras.Model:
    """
    Parameters
    ----------
    window_size : int
        Numero de timesteps en la ventana.
    n_features : int
        Numero de features por timestep.
    """
    model = Sequential([
        Conv1D(32, kernel_size=3, activation="relu", padding="same",
               input_shape=(window_size, n_features)),
        Conv1D(32, kernel_size=3, activation="relu", padding="same"),
        MaxPooling1D(pool_size=2),
        LSTM(32, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(16, activation="relu"),
        Dense(1),
    ], name="CNN_LSTM_Diario")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )
    return model


def count_parameters(model: tf.keras.Model) -> int:
    return model.count_params()
