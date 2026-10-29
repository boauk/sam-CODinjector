def compute_best_score(metrics):
    return (
        0.35 * metrics["Sm"]
        + 0.30 * metrics["Fomega"]
        + 0.20 * metrics["alphaE"]
        + 0.10 * metrics["Fx"]
        + 0.05 * (1.0 - metrics["MAE"])
    )
