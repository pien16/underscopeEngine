"""Shared data loading, splitting, scaling and metrics for the car price models.

Both models (``ml/knn.py`` and ``ml/gradient_descent.py``) use the helpers here
so that they see exactly the same features and the same train/test split.
That keeps the KNN vs. gradient descent comparison fair.

Two datasets are supported:

* ``car_data``  -> ``Data/Car_data.csv``             (2,095 used cars, target ``selling_price``)
* ``carprice``  -> ``Data/CarPrice_Assignment.csv``  (205 cars, target ``price``)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "Data"

CAR_DATA_CSV = DATA_DIR / "Car_data.csv"
CARPRICE_CSV = DATA_DIR / "CarPrice_Assignment.csv"

# Year the "age" feature is measured from. The newest car in Car_data is 2020.
REFERENCE_YEAR = 2020

DEFAULT_SEED = 42
DEFAULT_TEST_SIZE = 0.2


# --------------------------------------------------------------------------- #
# Dataset loaders
# --------------------------------------------------------------------------- #

@dataclass
class Dataset:
    """A ready-to-model dataset: numeric feature matrix plus target."""

    name: str
    X: pd.DataFrame          # all-numeric features, one-hot encoded
    y: pd.Series             # target price
    target_name: str
    currency: str            # used only for labelling output


_OWNER_ORDER = {
    "Test Drive Car": 0,
    "First Owner": 1,
    "Second Owner": 2,
    "Third Owner": 3,
    "Fourth & Above Owner": 4,
}


def load_car_data(path: Path = CAR_DATA_CSV) -> Dataset:
    """Load ``Car_data.csv`` (Indian used-car listings, prices in INR).

    Feature engineering:
    * the unnamed index column is dropped;
    * ``year`` becomes ``age`` (years before 2020);
    * ``owner`` is ordinal (first owner = 1 ... fourth & above = 4);
    * ``transmission`` becomes a 0/1 ``is_automatic`` flag;
    * brand (``name``), ``fuel``, ``seller_type`` and ``Mileage Unit`` are one-hot encoded.
    """
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])

    df["age"] = REFERENCE_YEAR - df.pop("year")
    df["owner"] = df["owner"].map(_OWNER_ORDER)
    if df["owner"].isna().any():
        raise ValueError("Unknown owner category in Car_data.csv")
    df["is_automatic"] = (df.pop("transmission") == "Automatic").astype(int)
    df = df.rename(columns={
        "name": "brand",
        "max_power (in bph)": "max_power_bhp",
        "Mileage Unit": "mileage_unit",
        "Mileage": "mileage",
        "Engine (CC)": "engine_cc",
    })

    y = df.pop("selling_price").astype(float)
    X = pd.get_dummies(
        df,
        columns=["brand", "fuel", "seller_type", "mileage_unit"],
        drop_first=True,
        dtype=float,
    )
    return Dataset("car_data", X.astype(float), y, "selling_price", "INR")


# Misspellings in CarPrice_Assignment.csv's CarName column.
_BRAND_FIXES = {
    "maxda": "mazda",
    "porcshce": "porsche",
    "toyouta": "toyota",
    "vokswagen": "volkswagen",
    "vw": "volkswagen",
}

_WORD_TO_NUMBER = {
    "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "eight": 8, "twelve": 12,
}


def load_carprice(path: Path = CARPRICE_CSV) -> Dataset:
    """Load ``CarPrice_Assignment.csv`` (1985 US auto specs, prices in USD).

    Feature engineering:
    * ``car_ID`` is dropped (it is a row id, not a feature);
    * ``CarName`` is reduced to a brand, with spelling fixes (maxda -> mazda, vw -> volkswagen, ...);
    * ``doornumber`` and ``cylindernumber`` are converted from words to numbers;
    * the remaining categorical columns are one-hot encoded.
    """
    df = pd.read_csv(path)
    df = df.drop(columns=["car_ID"])

    brand = df.pop("CarName").str.split().str[0].str.lower()
    df["brand"] = brand.replace(_BRAND_FIXES)
    for col in ("doornumber", "cylindernumber"):
        df[col] = df[col].map(_WORD_TO_NUMBER)
        if df[col].isna().any():
            raise ValueError(f"Unknown word in {col} column")

    y = df.pop("price").astype(float)
    categorical = [
        "brand", "fueltype", "aspiration", "carbody", "drivewheel",
        "enginelocation", "enginetype", "fuelsystem",
    ]
    X = pd.get_dummies(df, columns=categorical, drop_first=True, dtype=float)
    return Dataset("carprice", X.astype(float), y, "price", "USD")


LOADERS = {
    "car_data": load_car_data,
    "carprice": load_carprice,
}


def load(name: str) -> Dataset:
    try:
        return LOADERS[name]()
    except KeyError:
        raise ValueError(f"Unknown dataset {name!r}; choose from {sorted(LOADERS)}") from None


# --------------------------------------------------------------------------- #
# Splitting and scaling
# --------------------------------------------------------------------------- #

def train_test_split(X, y, test_size: float = DEFAULT_TEST_SIZE, seed: int = DEFAULT_SEED):
    """Shuffle and split arrays into train and test parts (deterministic for a given seed)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(round(len(X) * test_size))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def kfold_indices(n: int, k: int = 5, seed: int = DEFAULT_SEED):
    """Yield (train_idx, val_idx) pairs for k-fold cross-validation."""
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), k)
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train_idx, val_idx


class StandardScaler:
    """Z-score scaling. Fit on training data only, then apply to train and test."""

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        # Columns that are constant in the training split are left unscaled (not divided by 0).
        self.scale_ = np.where(std > 0, std, 1.0)
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Return R^2, RMSE, MAE and MAPE (%) for a set of predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = y_true - y_pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "mae": float(np.mean(np.abs(resid))),
        "mape": float(np.mean(np.abs(resid) / np.abs(y_true)) * 100),
    }


def format_metrics(m: dict[str, float], currency: str) -> str:
    return (f"R^2 = {m['r2']:.3f} | RMSE = {m['rmse']:,.0f} {currency} | "
            f"MAE = {m['mae']:,.0f} {currency} | MAPE = {m['mape']:.1f}%")
