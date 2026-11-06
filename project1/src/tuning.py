from itertools import product


def get_hyperparameters(model_name):
    if model_name == "logreg":
        return [{"C": C} for C in [0.01, 0.1, 1, 10]]

    return [
        {"hidden_units": h, "n_epochs": e}
        for h, e in product([32, 64, 128], [10, 20])
    ]
    