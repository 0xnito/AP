"""Regresion Lineal para prediccion semanal de tweets."""
from sklearn.linear_model import LinearRegression


def build_model() -> LinearRegression:
    return LinearRegression()


def count_parameters(model: LinearRegression):
    try:
        return int(len(model.coef_) + 1)
    except AttributeError:
        return "N/A (no entrenado)"
