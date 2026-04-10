"""
Unit tests for data pipeline.
"""
import unittest
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from model.src.data_pipeline import fetch_data, preprocess, prepare_features, normalize

class TestDataPipeline(unittest.TestCase):
    
    def setUp(self):
        self.ticker = "RELIANCE.NS"
        self.start = "2023-01-01"
        self.end = "2023-01-31"
        
    def test_fetch_data_returns_non_empty(self):
        """Test that fetch_data returns a non-empty dataframe."""
        df = fetch_data(self.ticker, self.start, self.end)
        self.assertFalse(df.empty)
        if isinstance(df.columns, pd.MultiIndex):
            self.assertIn("Close", df.columns.get_level_values(0))
        else:
            self.assertIn("Close", df.columns)
        
    def test_preprocess_no_nan(self):
        """Test that preprocess doesn't produce NaN values."""
        df = fetch_data(self.ticker, "2022-01-01", "2023-01-01") 
        processed_df = preprocess(df)
        self.assertFalse(processed_df.isnull().values.any())
        self.assertIn("SMA_20", processed_df.columns)
        
    def test_features_shape(self):
        """Test that features shape is correct."""
        df = fetch_data(self.ticker, "2022-01-01", "2023-01-01")
        processed_df = preprocess(df)
        norm_df = normalize(processed_df)
        X, y = prepare_features(norm_df)
        
        self.assertEqual(len(X), len(y))
        self.assertNotIn("Target", X.columns)
        self.assertNotIn("Target", norm_df.columns)

if __name__ == "__main__":
    unittest.main()
