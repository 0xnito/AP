"""
Bidirectional LSTM + GRU Hybrid Model for Weekly Tweet Count Prediction
========================================================================
Combines bidirectional LSTM layers with GRU layers to capture both
forward and backward temporal dependencies.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


def build_lstm_gru_model(
    seq_len: int = 8,
    n_features: int = 1,
    units: int = 64,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """
    Bidirectional LSTM + GRU hybrid model.

    Architecture:
        Input → BiLSTM(64) → Dropout → GRU(32) → Dropout → Dense(16) → Dense(1)

    Parameters
    ----------
    seq_len : int
        Number of past weeks used as input.
    n_features : int
        Number of input features per time step.
    units : int
        Units in the LSTM layer; GRU uses units//2.
    dropout : float
        Dropout rate applied after each recurrent layer.
    learning_rate : float
        Adam optimizer learning rate.

    Returns
    -------
    tf.keras.Model
        Compiled Keras model.
    """
    inputs = layers.Input(shape=(seq_len, n_features), name="input")

    x = layers.Bidirectional(
        layers.LSTM(units, return_sequences=True), name="bilstm_1"
    )(inputs)
    x = layers.Dropout(dropout, name="dropout_1")(x)

    x = layers.GRU(units // 2, return_sequences=False, name="gru_1")(x)
    x = layers.Dropout(dropout, name="dropout_2")(x)

    x = layers.Dense(16, activation="relu", name="dense_1")(x)
    outputs = layers.Dense(1, name="output")(x)

    model = Model(inputs, outputs, name="BiLSTM_GRU_TweetPredictor")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


if __name__ == "__main__":
    model = build_lstm_gru_model()
    model.summary()
