"""Tests for the KNN and gradient descent car price models.

Run with:  python3 -m unittest discover -s ml/tests -t .
"""

import unittest

import numpy as np

from ml import data
from ml.gradient_descent import LinearRegressionGD, closed_form_ridge
from ml.knn import KNNRegressor


class TestKNN(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[0.0], [1.0], [2.0], [10.0]])
        self.y = np.array([0.0, 10.0, 20.0, 100.0])

    def test_k1_returns_nearest_target(self):
        model = KNNRegressor(k=1).fit(self.X, self.y)
        np.testing.assert_allclose(model.predict([[0.9], [9.0]]), [10.0, 100.0])

    def test_uniform_average(self):
        model = KNNRegressor(k=3).fit(self.X, self.y)
        np.testing.assert_allclose(model.predict([[1.0]]), [10.0])  # mean of 0, 10, 20

    def test_distance_weighting_favours_closer_points(self):
        model = KNNRegressor(k=2, weights="distance").fit(self.X, self.y)
        # Neighbours 1.0 (dist 0.25, y=10) and 2.0 (dist 0.75, y=20): weights 4 and 4/3.
        np.testing.assert_allclose(model.predict([[1.25]]), [(4 * 10 + 4 / 3 * 20) / (4 + 4 / 3)])

    def test_distance_weighting_exact_match(self):
        model = KNNRegressor(k=3, weights="distance").fit(self.X, self.y)
        np.testing.assert_allclose(model.predict([[2.0]]), [20.0])

    def test_matches_brute_force(self):
        rng = np.random.default_rng(0)
        X, y, Q = rng.normal(size=(50, 4)), rng.normal(size=50), rng.normal(size=(10, 4))
        pred = KNNRegressor(k=5).fit(X, y).predict(Q)
        for q, p in zip(Q, pred):
            nearest = np.argsort(np.linalg.norm(X - q, axis=1))[:5]
            self.assertAlmostEqual(p, y[nearest].mean())

    def test_invalid_k(self):
        with self.assertRaises(ValueError):
            KNNRegressor(k=0)
        with self.assertRaises(ValueError):
            KNNRegressor(k=10).fit(self.X, self.y)


class TestGradientDescent(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.X = rng.normal(size=(200, 5))
        self.w = np.array([3.0, -2.0, 0.5, 0.0, 1.0])
        self.y = self.X @ self.w + 7.0 + rng.normal(scale=0.1, size=200)

    def test_recovers_true_weights(self):
        model = LinearRegressionGD().fit(self.X, self.y)
        self.assertTrue(model.converged_)
        np.testing.assert_allclose(model.coef_, self.w, atol=0.05)
        self.assertAlmostEqual(model.intercept_, 7.0, delta=0.05)

    def test_matches_closed_form_with_ridge(self):
        for l2 in (0.0, 0.1, 1.0):
            model = LinearRegressionGD(l2=l2).fit(self.X, self.y)
            w, b = closed_form_ridge(self.X, self.y, l2)
            np.testing.assert_allclose(model.coef_, w, atol=1e-4)
            self.assertAlmostEqual(model.intercept_, b, places=4)

    def test_loss_never_increases(self):
        model = LinearRegressionGD().fit(self.X, self.y)
        self.assertTrue(np.all(np.diff(model.loss_history_) <= 1e-12))

    def test_ridge_shrinks_weights(self):
        small = LinearRegressionGD(l2=0.0).fit(self.X, self.y)
        big = LinearRegressionGD(l2=10.0).fit(self.X, self.y)
        self.assertLess(np.linalg.norm(big.coef_), np.linalg.norm(small.coef_))

    def test_log_target_predicts_positive_prices(self):
        y = np.exp(self.X @ (self.w / 10) + 10)
        model = LinearRegressionGD(log_target=True).fit(self.X, y)
        pred = model.predict(self.X)
        self.assertTrue(np.all(pred > 0))
        np.testing.assert_allclose(pred, y, rtol=0.05)

    def test_too_large_learning_rate_raises(self):
        with self.assertRaises(FloatingPointError), np.errstate(over="ignore", invalid="ignore"):
            LinearRegressionGD(learning_rate=10.0).fit(self.X * 100, self.y)


class TestData(unittest.TestCase):
    def test_loaders_are_numeric_and_complete(self):
        for name, rows in (("car_data", 2095), ("carprice", 205)):
            ds = data.load(name)
            self.assertEqual(len(ds.X), rows)
            self.assertEqual(len(ds.y), rows)
            self.assertFalse(ds.X.isna().any().any(), name)
            self.assertTrue(all(np.issubdtype(t, np.number) for t in ds.X.dtypes), name)
            self.assertNotIn(ds.target_name, ds.X.columns)

    def test_carprice_brand_typos_fixed(self):
        cols = data.load("carprice").X.columns
        for bad in ("brand_maxda", "brand_vw", "brand_vokswagen", "brand_porcshce", "brand_toyouta", "brand_Nissan"):
            self.assertNotIn(bad, cols)

    def test_split_is_deterministic_and_disjoint(self):
        X = np.arange(100).reshape(-1, 1)
        a = data.train_test_split(X, X.ravel())
        b = data.train_test_split(X, X.ravel())
        np.testing.assert_array_equal(a[0], b[0])
        self.assertEqual(len(a[1]), 20)
        self.assertEqual(set(a[0].ravel()) & set(a[1].ravel()), set())

    def test_scaler_handles_constant_column(self):
        X = np.array([[1.0, 5.0], [3.0, 5.0]])
        out = data.StandardScaler().fit_transform(X)
        np.testing.assert_allclose(out, [[-1.0, 0.0], [1.0, 0.0]])

    def test_metrics(self):
        m = data.regression_metrics([1, 2, 3], [1, 2, 3])
        self.assertEqual(m["r2"], 1.0)
        self.assertEqual(m["rmse"], 0.0)


if __name__ == "__main__":
    unittest.main()
