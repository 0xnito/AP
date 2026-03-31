"""Random Forest para prediccion semanal de tweets."""
from sklearn.ensemble import RandomForestRegressor


def build_model(n_estimators: int = 100, max_depth: int = 5,
                random_state: int = 42) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )


def count_parameters(model: RandomForestRegressor):
    try:
        return sum(t.tree_.node_count for t in model.estimators_)
    except AttributeError:
        return "N/A (no entrenado)"
