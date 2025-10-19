import unittest
import pandas as pd
from joblib import load
import os


class TestPretrainedModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_path = "model.joblib"
        cls.data_path = "samples/data.csv"

    def test_model_prediction(self):
        # Check files exist
        self.assertTrue(
            os.path.exists(self.model_path), f"Model file not found: {self.model_path}"
        )
        self.assertTrue(
            os.path.exists(self.data_path), f"Data file not found: {self.data_path}"
        )

        # Load model
        model = load(self.model_path)

        # Load data
        df = pd.read_csv(self.data_path)
        X = df.drop("species", axis=1)
        y = df["species"]

        # Run predictions
        preds = model.predict(X)

        # Assert predictions length matches data length
        # self.assertEqual(len(preds), len(X), "Prediction length mismatch")
        self.assertEqual(len(preds), len(X), "Prediction length mismatch")

        # Compare first 5 predictions to actual labels
        for i in range(5):
            self.assertEqual(
                preds[i],
                y.iloc[i],
                f"Prediction mismatch at index {i}: predicted {preds[i]}, expected {y.iloc[i]}",
            )


if __name__ == "__main__":
    unittest.main()
