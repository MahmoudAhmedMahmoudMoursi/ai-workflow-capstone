import os
import tempfile
import unittest
import pandas as pd
import numpy as np

from app import app
from model import evaluate_baseline, train_and_compare_models, predict_revenue
from monitoring import detect_drift, log_prediction_event


class TestAAVAILWorkflow(unittest.TestCase):
    """Isolated unit tests for API, model, and logging without touching production data."""

    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_log_file = os.path.join(self.temp_dir.name, "test_isolated.log")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_api_predict_single_country(self):
        response = self.client.post(
            "/predict",
            json={"country": "united_kingdom", "target_date": "2020-01-15"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["country"], "united_kingdom")
        self.assertIn("predicted_revenue", data)

    def test_api_predict_all_countries(self):
        response = self.client.post(
            "/predict_all",
            json={"target_date": "2020-01-15"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["country"], "all_countries")
        self.assertGreater(data["predicted_revenue"], 0)

    def test_api_invalid_input_returns_400(self):
        response = self.client.post(
            "/predict",
            json={"country": "invalid_country", "target_date": "not-a-date"},
        )
        self.assertEqual(response.status_code, 400)

    def test_model_training_and_baseline_comparison(self):
        X_train = pd.DataFrame({"feat1": np.arange(1, 21, dtype=float)})
        y_train = pd.Series(np.arange(100, 2000, 95, dtype=float)[:20])
        X_test = pd.DataFrame({"feat1": np.arange(21, 26, dtype=float)})
        y_test = pd.Series([2000.0, 2100.0, 2200.0, 2300.0, 2400.0])

        baseline = evaluate_baseline(y_test)
        self.assertIn("rmse", baseline)

        summary = train_and_compare_models(X_train, y_train, X_test, y_test)
        self.assertIsNotNone(summary["best_model"])
        preds = predict_revenue(summary["best_model"], X_test)
        self.assertEqual(len(preds), len(X_test))

    def test_logging_and_drift_detection_isolated(self):
        ref = pd.Series([100, 105, 98, 102, 101])
        cur = pd.Series([500, 520, 490, 510, 505])
        drift = detect_drift(ref, cur)
        self.assertTrue(drift["drift_detected"])

        log_prediction_event(
            country="united_kingdom",
            target_date="2020-01-15",
            prediction=12500.0,
            drift_info=drift,
            log_file=self.test_log_file,
        )
        self.assertTrue(os.path.exists(self.test_log_file))
        with open(self.test_log_file, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("united_kingdom", content)


if __name__ == "__main__":
    unittest.main()
