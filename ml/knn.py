"""K-Nearest Neighbors regression for car prices, written from scratch in NumPy.

Two versions are provided, one per dataset:

* ``run_car_data()``  - Car_data.csv             (predicts ``selling_price``, INR)
* ``run_carprice()``  - CarPrice_Assignment.csv  (predicts ``price``, USD)

How it works: features are standardized (z-scores, fitted on the training split
only). To price a car, the model finds the ``k`` training cars closest to it
(Euclidean distance) and averages their prices, optionally weighting closer
cars more. ``k`` and the weighting are chosen by 5-fold cross-validation on the
training split. The test split is only used once, for the final score.

Usage::

    python3 -m ml.knn                    # both datasets
    python3 -m ml.knn --dataset carprice
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from ml import data


class KNNRegressor:
    """K-nearest-neighbors regressor.

    Parameters
    ----------
    k : int
        Number of neighbors to average.
    weights : {"uniform", "distance"}
        ``uniform`` takes a plain mean of the k neighbors' prices. ``distance``
        weights each neighbor by 1 / distance, so closer cars count more.
    """

    def __init__(self, k: int = 5, weights: str = "uniform"):
        if k < 1:
            raise ValueError("k must be >= 1")
        if weights not in ("uniform", "distance"):
            raise ValueError("weights must be 'uniform' or 'distance'")
        self.k = k
        self.weights = weights

    def fit(self, X, y):
        # KNN is a "lazy" learner: training just stores the data.
        self.X_train_ = np.asarray(X, dtype=float)
        self.y_train_ = np.asarray(y, dtype=float)
        if self.k > len(self.X_train_):
            raise ValueError(f"k={self.k} is larger than the training set ({len(self.X_train_)})")
        return self

    def _distances(self, X):
        # Pairwise Euclidean distances via |a-b|^2 = |a|^2 + |b|^2 - 2ab (fast, no Python loops).
        X = np.asarray(X, dtype=float)
        sq = (np.sum(X ** 2, axis=1)[:, None]
              + np.sum(self.X_train_ ** 2, axis=1)[None, :]
              - 2.0 * X @ self.X_train_.T)
        return np.sqrt(np.maximum(sq, 0.0))  # clip tiny negative rounding errors

    def predict(self, X):
        dist = self._distances(X)
        # Indices of the k smallest distances in each row ("stable" keeps ties deterministic).
        nn = np.argsort(dist, axis=1, kind="stable")[:, : self.k]
        nn_dist = np.take_along_axis(dist, nn, axis=1)
        nn_y = self.y_train_[nn]

        if self.weights == "uniform":
            return nn_y.mean(axis=1)

        # Distance weighting. If a test point exactly matches training points,
        # use only those exact matches (their weight would otherwise be infinite).
        exact = nn_dist == 0
        w = np.where(exact.any(axis=1, keepdims=True),
                     exact.astype(float),
                     1.0 / np.where(exact, 1.0, nn_dist))
        return np.sum(w * nn_y, axis=1) / np.sum(w, axis=1)


def cross_validate_k(X, y, k_values, weight_options=("uniform", "distance"), folds=5, seed=data.DEFAULT_SEED):
    """Pick k and weighting by k-fold cross-validated RMSE.

    Scaling is refitted inside each fold so the validation fold never leaks into it.
    Returns (best_params, table) where table lists the mean CV RMSE of every combination.
    """
    table = []
    for weights in weight_options:
        for k in k_values:
            rmses = []
            for tr, va in data.kfold_indices(len(X), folds, seed):
                scaler = data.StandardScaler().fit(X[tr])
                model = KNNRegressor(k, weights).fit(scaler.transform(X[tr]), y[tr])
                pred = model.predict(scaler.transform(X[va]))
                rmses.append(np.sqrt(np.mean((y[va] - pred) ** 2)))
            table.append({"k": k, "weights": weights, "cv_rmse": float(np.mean(rmses))})
    best = min(table, key=lambda row: row["cv_rmse"])
    return {"k": best["k"], "weights": best["weights"]}, table


def run(dataset_name: str, k_values, verbose: bool = True) -> dict:
    """Full pipeline for one dataset: load, split, tune k by CV, fit, score on the test split."""
    ds = data.load(dataset_name)
    X_train, X_test, y_train, y_test = data.train_test_split(ds.X, ds.y)

    best, cv_table = cross_validate_k(X_train, y_train, k_values)

    scaler = data.StandardScaler().fit(X_train)
    model = KNNRegressor(**best).fit(scaler.transform(X_train), y_train)
    test_pred = model.predict(scaler.transform(X_test))
    train_pred = model.predict(scaler.transform(X_train))

    # Baseline: always predict the mean training price. Any useful model must beat this.
    baseline = data.regression_metrics(y_test, np.full_like(y_test, y_train.mean()))

    result = {
        "model": "knn",
        "dataset": ds.name,
        "currency": ds.currency,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(X_train.shape[1]),
        "best_params": best,
        "cv_table": cv_table,
        "train_metrics": data.regression_metrics(y_train, train_pred),
        "test_metrics": data.regression_metrics(y_test, test_pred),
        "baseline_test_metrics": baseline,
        "y_test": y_test.tolist(),
        "y_pred": test_pred.tolist(),
    }
    if verbose:
        _print_result(result)
    return result


# ---- The two dataset-specific versions ------------------------------------ #

def run_car_data(verbose: bool = True) -> dict:
    """KNN on Car_data.csv (2,095 cars). Searches k = 1..30."""
    return run("car_data", k_values=range(1, 31), verbose=verbose)


def run_carprice(verbose: bool = True) -> dict:
    """KNN on CarPrice_Assignment.csv (205 cars). Small data, so k is searched over 1..15."""
    return run("carprice", k_values=range(1, 16), verbose=verbose)


VERSIONS = {"car_data": run_car_data, "carprice": run_carprice}


def _print_result(r: dict) -> None:
    cur = r["currency"]
    print(f"\n=== KNN on {r['dataset']} ===")
    print(f"train/test = {r['n_train']}/{r['n_test']} rows, {r['n_features']} features")
    print(f"best by 5-fold CV: k = {r['best_params']['k']}, weights = {r['best_params']['weights']}")
    print(f"train    : {data.format_metrics(r['train_metrics'], cur)}")
    print(f"test     : {data.format_metrics(r['test_metrics'], cur)}")
    print(f"baseline : {data.format_metrics(r['baseline_test_metrics'], cur)}  (predict the mean)")


def main(argv=None):
    parser = argparse.ArgumentParser(description="KNN car price regression")
    parser.add_argument("--dataset", choices=[*VERSIONS, "both"], default="both")
    parser.add_argument("--json", metavar="PATH", help="also write the results to this JSON file")
    args = parser.parse_args(argv)

    names = list(VERSIONS) if args.dataset == "both" else [args.dataset]
    results = [VERSIONS[n]() for n in names]
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
