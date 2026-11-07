"""Linear regression trained with batch gradient descent, written from scratch in NumPy.

Two versions are provided, one per dataset:

* ``run_car_data()``  - Car_data.csv             (predicts ``selling_price``, INR)
* ``run_carprice()``  - CarPrice_Assignment.csv  (predicts ``price``, USD)

The model is ``price ~ X @ w + b``. Gradient descent minimizes the ridge loss

    J(w, b) = 1/(2n) * sum((X @ w + b - y)^2) + (l2 / 2) * ||w||^2

by repeatedly stepping against the gradient

    dJ/dw = X.T @ (X @ w + b - y) / n + l2 * w
    dJ/db = mean(X @ w + b - y)

until the loss stops improving. Features are standardized first; without that,
gradient descent on columns like ``km_driven`` (~10^5) next to ``seats`` (~5)
converges extremely slowly or diverges.

Options tuned by 5-fold cross-validation on the training split:
* ``l2``         - ridge penalty strength (0 = plain least squares);
* ``log_target`` - fit log(price) instead of price, then exponentiate predictions.
  Car prices are right-skewed and depreciation is roughly multiplicative, so the
  log model is often a better fit.

Usage::

    python3 -m ml.gradient_descent                    # both datasets
    python3 -m ml.gradient_descent --dataset car_data
"""

from __future__ import annotations

import argparse
import itertools
import json

import numpy as np

from ml import data


class LinearRegressionGD:
    """Linear (ridge) regression fitted by full-batch gradient descent.

    Parameters
    ----------
    learning_rate : float or None
        Step size. ``None`` (default) picks ``1 / L``, where ``L`` is the largest
        eigenvalue of the loss's Hessian. That is the largest step size that is
        guaranteed not to diverge.
    max_iter : int
        Maximum number of gradient steps.
    tol : float
        Stop once the relative improvement of the loss in one step is below this.
    l2 : float
        Ridge penalty strength (``0`` = ordinary least squares).
    log_target : bool
        Fit ``log(y)`` and return ``exp(prediction)``.
    """

    def __init__(self, learning_rate=None, max_iter=50_000, tol=1e-9, l2=0.0, log_target=False):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.l2 = l2
        self.log_target = log_target

    def _loss(self, X, y, w, b):
        r = X @ w + b - y
        return 0.5 * np.mean(r ** 2) + 0.5 * self.l2 * np.dot(w, w)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if self.log_target:
            if np.any(y <= 0):
                raise ValueError("log_target requires all targets > 0")
            y = np.log(y)
        n, d = X.shape

        if self.learning_rate is None:
            # Hessian of J w.r.t. (w, b) is [[X.T X / n + l2 I, mean(X)], [mean(X).T, 1]].
            Xb = np.hstack([X, np.ones((n, 1))])
            hessian = Xb.T @ Xb / n
            hessian[:d, :d] += self.l2 * np.eye(d)
            lr = 1.0 / np.linalg.eigvalsh(hessian)[-1]
        else:
            lr = self.learning_rate
        self.learning_rate_ = float(lr)

        w = np.zeros(d)
        b = float(y.mean())          # start the bias at the mean; it saves many steps
        loss = self._loss(X, y, w, b)
        self.loss_history_ = [loss]
        self.converged_ = False

        for i in range(1, self.max_iter + 1):
            r = X @ w + b - y                          # residuals
            grad_w = X.T @ r / n + self.l2 * w
            grad_b = r.mean()
            w -= lr * grad_w
            b -= lr * grad_b

            new_loss = self._loss(X, y, w, b)
            if not np.isfinite(new_loss):
                raise FloatingPointError(f"gradient descent diverged at step {i}; lower the learning rate")
            self.loss_history_.append(new_loss)
            if abs(loss - new_loss) <= self.tol * max(abs(loss), 1e-12):
                self.converged_ = True
                break
            loss = new_loss

        self.coef_, self.intercept_, self.n_iter_ = w, b, i
        return self

    def predict(self, X):
        pred = np.asarray(X, dtype=float) @ self.coef_ + self.intercept_
        return np.exp(pred) if self.log_target else pred


def closed_form_ridge(X, y, l2=0.0):
    """Exact minimizer of the same loss, solved directly. Used to check gradient descent.

    Returns (w, b). The intercept is not penalized, so X and y are centered first.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = X.shape
    x_mean, y_mean = X.mean(axis=0), y.mean()
    Xc, yc = X - x_mean, y - y_mean
    A = Xc.T @ Xc / n + l2 * np.eye(d)
    # lstsq instead of solve: with l2 = 0 and collinear one-hot columns, A is singular.
    w = np.linalg.lstsq(A, Xc.T @ yc / n, rcond=None)[0]
    return w, y_mean - x_mean @ w


def cross_validate(X, y, l2_values, log_options=(False, True), folds=5, seed=data.DEFAULT_SEED):
    """Pick (l2, log_target) by k-fold cross-validated RMSE on the original price scale."""
    table = []
    for l2, log_target in itertools.product(l2_values, log_options):
        rmses = []
        for tr, va in data.kfold_indices(len(X), folds, seed):
            scaler = data.StandardScaler().fit(X[tr])
            model = LinearRegressionGD(l2=l2, log_target=log_target).fit(scaler.transform(X[tr]), y[tr])
            pred = model.predict(scaler.transform(X[va]))
            rmses.append(np.sqrt(np.mean((y[va] - pred) ** 2)))
        table.append({"l2": l2, "log_target": log_target, "cv_rmse": float(np.mean(rmses))})
    best = min(table, key=lambda row: row["cv_rmse"])
    return {"l2": best["l2"], "log_target": best["log_target"]}, table


def run(dataset_name: str, l2_values, verbose: bool = True) -> dict:
    """Full pipeline for one dataset: load, split, tune by CV, fit, score on the test split."""
    ds = data.load(dataset_name)
    X_train, X_test, y_train, y_test = data.train_test_split(ds.X, ds.y)

    best, cv_table = cross_validate(X_train, y_train, l2_values)

    scaler = data.StandardScaler().fit(X_train)
    Xs_train, Xs_test = scaler.transform(X_train), scaler.transform(X_test)
    model = LinearRegressionGD(**best).fit(Xs_train, y_train)
    test_pred = model.predict(Xs_test)
    train_pred = model.predict(Xs_train)

    # Sanity check: the closed-form solution of the same problem should give near-identical predictions.
    y_fit = np.log(y_train) if best["log_target"] else y_train
    w_cf, b_cf = closed_form_ridge(Xs_train, y_fit, best["l2"])
    cf_pred = Xs_test @ w_cf + b_cf
    if best["log_target"]:
        cf_pred = np.exp(cf_pred)

    baseline = data.regression_metrics(y_test, np.full_like(y_test, y_train.mean()))
    feature_names = list(ds.X.columns)
    # Coefficients are per one standard deviation of each feature (features are standardized).
    top = sorted(zip(feature_names, model.coef_), key=lambda t: -abs(t[1]))[:8]

    result = {
        "model": "gradient_descent",
        "dataset": ds.name,
        "currency": ds.currency,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(X_train.shape[1]),
        "best_params": best,
        "cv_table": cv_table,
        "learning_rate": model.learning_rate_,
        "n_iter": int(model.n_iter_),
        "converged": bool(model.converged_),
        "loss_history": [float(v) for v in model.loss_history_],
        "train_metrics": data.regression_metrics(y_train, train_pred),
        "test_metrics": data.regression_metrics(y_test, test_pred),
        "closed_form_test_metrics": data.regression_metrics(y_test, cf_pred),
        "baseline_test_metrics": baseline,
        "top_coefficients": [{"feature": f, "coef": float(c)} for f, c in top],
        "y_test": y_test.tolist(),
        "y_pred": test_pred.tolist(),
    }
    if verbose:
        _print_result(result)
    return result


# ---- The two dataset-specific versions ------------------------------------ #

L2_GRID = [0.0, 0.001, 0.01, 0.1, 1.0]


def run_car_data(verbose: bool = True) -> dict:
    """Gradient descent linear regression on Car_data.csv (2,095 cars, 23 features)."""
    return run("car_data", l2_values=L2_GRID, verbose=verbose)


def run_carprice(verbose: bool = True) -> dict:
    """Gradient descent linear regression on CarPrice_Assignment.csv.

    Only 164 training rows but 59 features, many of them rare one-hot brands, so
    regularization matters more here.
    """
    return run("carprice", l2_values=L2_GRID, verbose=verbose)


VERSIONS = {"car_data": run_car_data, "carprice": run_carprice}


def _print_result(r: dict) -> None:
    cur = r["currency"]
    p = r["best_params"]
    print(f"\n=== Gradient descent linear regression on {r['dataset']} ===")
    print(f"train/test = {r['n_train']}/{r['n_test']} rows, {r['n_features']} features")
    print(f"best by 5-fold CV: l2 = {p['l2']}, log_target = {p['log_target']}")
    print(f"learning rate = {r['learning_rate']:.4f}, steps = {r['n_iter']}, converged = {r['converged']}")
    print(f"train       : {data.format_metrics(r['train_metrics'], cur)}")
    print(f"test        : {data.format_metrics(r['test_metrics'], cur)}")
    print(f"closed form : {data.format_metrics(r['closed_form_test_metrics'], cur)}  (exact solution, for comparison)")
    print(f"baseline    : {data.format_metrics(r['baseline_test_metrics'], cur)}  (predict the mean)")
    print("largest coefficients (per 1 std of the feature"
          + (", in log-price units):" if p["log_target"] else f", in {cur}):"))
    for t in r["top_coefficients"]:
        print(f"  {t['feature']:<28} {t['coef']:+,.3f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Gradient descent linear regression for car prices")
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
