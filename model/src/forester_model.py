"""
Forester Model module.
Uses ExtraTreesRegressor from scikit-learn as the Forester model.
"""
import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from model.config.settings import MODEL_PARAMS, MODELS_DIR

def train(X_train: pd.DataFrame, y_train: pd.Series) -> ExtraTreesRegressor:
    """
    Fits the Forester (ExtraTrees) model.
    """
    model = ExtraTreesRegressor(**MODEL_PARAMS)
    model.fit(X_train, y_train)
    return model

def evaluate(model: ExtraTreesRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluates the model and returns RMSE, MAE, R2.
    """
    predictions = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }

def predict(model: ExtraTreesRegressor, X: pd.DataFrame) -> pd.Series:
    """
    Returns predictions from the model.
    """
    preds = model.predict(X)
    return pd.Series(preds, index=X.index, name="Predicted_Close")

def save_model(model: ExtraTreesRegressor, ticker: str) -> None:
    """
    Pickles the model to models directory.
    """
    path = MODELS_DIR / f"{ticker}_forester.pkl"
    joblib.dump(model, path)

def load_model(ticker: str) -> ExtraTreesRegressor:
    """
    Loads pickled model.
    """
    path = MODELS_DIR / f"{ticker}_forester.pkl"
    return joblib.load(path)

def run_forester(ticker: str, X: pd.DataFrame, y: pd.Series) -> tuple[ExtraTreesRegressor, dict, pd.DataFrame]:
    """
    Full pipeline — train, evaluate, return predictions + actuals
    (Performs a simple 80-20 train-test split for this pipeline)
    """
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    model = train(X_train, y_train)
    save_model(model, ticker)
    
    metrics = evaluate(model, X_test, y_test)
    
    # Predict on the whole dataset for visualization
    all_predictions = predict(model, X)
    
    results_df = pd.DataFrame({
        'Actual': y,
        'Predicted': all_predictions
    }, index=y.index)
    
    return model, metrics, results_df
