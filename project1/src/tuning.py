from itertools import product


def get_hyperparameters(model_name):
    if model_name == "logreg":
        return [{"C": C} for C in [0.1, 1]]

    return [
        {"hidden_units": h, "n_epochs": e}
        for h, e in product([64, 128], [5])
    ]
    