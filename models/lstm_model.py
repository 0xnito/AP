"""
LSTM Model for Weekly Tweet Count Prediction
=============================================
Predicts the number of tweets in the next week based on
a sliding window of historical weekly tweet counts.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


def build_lstm_model(
    seq_len: int = 8,
    n_features: int = 1,
    lstm_units: int = 64,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """
    Stacked LSTM model for univariate time-series regression.

    Architecture:
        Input → LSTM(64) → Dropout → LSTM(32) → Dropout → Dense(16) → Dense(1)

    Parameters
    ----------
    seq_len : int
        Number of past weeks used as input.
    n_features : int
        Number of input features per time step.
    lstm_units : int
        Units in the first LSTM layer (second layer uses units//2).
    dropout : float
        Dropout rate applied after each LSTM layer.
    learning_rate : float
        Adam optimizer learning rate.

    Returns
    -------
    tf.keras.Model
        Compiled Keras model.
    """
    inputs = layers.Input(shape=(seq_len, n_features), name="input")

    x = layers.LSTM(lstm_units, return_sequences=True, name="lstm_1")(inputs)
    x = layers.Dropout(dropout, name="dropout_1")(x)

    x = layers.LSTM(lstm_units // 2, return_sequences=False, name="lstm_2")(x)
    x = layers.Dropout(dropout, name="dropout_2")(x)

    x = layers.Dense(16, activation="relu", name="dense_1")(x)
    outputs = layers.Dense(1, name="output")(x)

    model = Model(inputs, outputs, name="LSTM_TweetPredictor")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


if __name__ == "__main__":
    model = build_lstm_model()
    model.summary()
