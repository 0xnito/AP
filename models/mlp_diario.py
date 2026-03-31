"""MLP mejorado para prediccion diaria de tweets.
Entrada: ventana 30 dias x 11 features, aplanada (330 entradas).
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2


def build_mlp_model(input_dim: int) -> tf.keras.Model:
    """
    Parameters
    ----------
    input_dim : int
        Dimension de la entrada aplanada (window_size x n_features).
    """
    model = Sequential([
        Dense(64, activation="relu", kernel_regularizer=l2(0.001),
              input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation="relu", kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(16, activation="relu"),
        Dense(1),
    ], name="MLP_Diario")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )
    return model


def count_parameters(model: tf.keras.Model) -> int:
    return model.count_params()
