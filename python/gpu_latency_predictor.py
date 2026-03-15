"""
GPU Kernel Latency Predictor

Predicts GPU kernel latency (ms) from hardware specs and matrix dimensions
using an sklearn MLPRegressor neural network.
"""

import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_data(csv_path: str) -> pd.DataFrame:
    """Load and validate the CSV dataset."""
    df = pd.read_csv(csv_path)

    expected_columns = [
        "gpu model name", "number of sms", "number of cores",
        "core clock speed", "memory bandwidth", "n", "m", "k",
        "latency in ms",
    ]
    missing = set(expected_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # Drop rows where the target is missing
    df = df.dropna(subset=["latency in ms"])

    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"GPU models: {df['gpu model name'].nunique()} unique")
    print(f"Target range: {df['latency in ms'].min():.4f} – {df['latency in ms'].max():.4f} ms")
    return df


def build_pipeline() -> Pipeline:
    """Build a preprocessing + MLPRegressor pipeline."""
    categorical_features = ["gpu model name"]
    numerical_features = [
        "number of sms", "number of cores", "core clock speed",
        "memory bandwidth", "n", "m", "k",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", MLPRegressor(
            random_state=42,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
        )),
    ])
    return pipeline


def tune_and_train(pipeline: Pipeline, X_train, y_train) -> Pipeline:
    """Run grid search over key hyperparameters and return the best model."""
    param_grid = {
        "regressor__hidden_layer_sizes": [
            (128, 64),
            (256, 128, 64),
            (128, 128, 64, 32),
        ],
        "regressor__activation": ["relu", "tanh"],
        "regressor__alpha": [1e-4, 1e-3, 1e-2],
        "regressor__learning_rate_init": [1e-3, 1e-4],
    }

    search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)

    print(f"\nBest params: {search.best_params_}")
    print(f"Best CV MAE:  {-search.best_score_:.4f} ms")
    return search.best_estimator_


def evaluate(model: Pipeline, X_test, y_test) -> dict:
    """Evaluate the trained model on the held-out test set."""
    y_pred = model.predict(X_test)

    metrics = {
        "MAE":  mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R²":   r2_score(y_test, y_pred),
    }

    print("\n--- Test Set Evaluation ---")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    return metrics


def main():
    if len(sys.argv) < 2:
        print("Usage: python gpu_latency_predictor.py <path_to_csv>")
        sys.exit(1)

    csv_path = sys.argv[1]

    # Load data
    df = load_data(csv_path)

    # Split features / target
    target = "latency in ms"
    feature_cols = [c for c in df.columns if c != target]
    X = df[feature_cols]
    y = df[target]

    # Train / test split (80/20), stratify not applicable for regression
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")

    # Build pipeline, tune, and train
    pipeline = build_pipeline()
    best_model = tune_and_train(pipeline, X_train, y_train)

    # Evaluate on test set
    evaluate(best_model, X_test, y_test)

    # Cross-validation on full dataset for additional confidence
    cv_scores = cross_val_score(
        best_model, X, y, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1
    )
    print(f"\n5-Fold CV MAE (full data): {-cv_scores.mean():.4f} ± {cv_scores.std():.4f} ms")


if __name__ == "__main__":
    main()
