import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# CONFIGURATION

RANDOM_STATE = 42
TARGET = "outbreak_7d"

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "raw" / "epiwatch_epidemiological_surveillance_dataset.csv"

REPORTS_DIR = BASE_DIR / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
MODELS_DIR = BASE_DIR / "models"

METRICS_PATH = TABLES_DIR / "model_performance_metrics.csv"
REPORTS_PATH = TABLES_DIR / "classification_reports.json"
BEST_MODEL_PATH = MODELS_DIR / "best_outbreak_prediction_model.pkl"

DROP_COLUMNS = [
    "outbreak_7d",
    "future_reported_cases_7d",
    "future_incidence_7d",
    "future_effective_R_7d",
    "true_cases",
    "date",
]


# UTILS

def create_directories() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Run src/simulate_data.py first."
        )

    df = pd.read_csv(path)

    if "date" not in df.columns:
        raise ValueError("Column 'date' is missing from the dataset.")

    df["date"] = pd.to_datetime(df["date"])

    return df


def validate_dataset(df: pd.DataFrame) -> None:
    required_columns = set(DROP_COLUMNS + [TARGET])

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"The dataset is missing required columns: {missing_columns}"
        )

    if df[TARGET].nunique() < 2:
        raise ValueError(
            "Target variable has only one class. "
            "Model training requires both outbreak and non-outbreak examples."
        )


def temporal_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    split_date = df["date"].quantile(1 - test_size)

    train_mask = df["date"] <= split_date
    test_mask = df["date"] > split_date

    X = df.drop(columns=DROP_COLUMNS)
    y = df[TARGET]

    X_train = X.loc[train_mask]
    X_test = X.loc[test_mask]
    y_train = y.loc[train_mask]
    y_test = y.loc[test_mask]

    print(f"Temporal split date: {split_date}")
    print(f"Training rows: {X_train.shape[0]}")
    print(f"Testing rows: {X_test.shape[0]}")
    print()

    return X_train, X_test, y_train, y_test


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_features = ["region"]

    numeric_features = [
        col for col in X.columns
        if col not in categorical_features
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def get_models() -> Dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }


def evaluate_model(
    model_name: str,
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    }

    return metrics


def train_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, dict], Pipeline]:
    preprocessor = build_preprocessor(X_train)
    models = get_models()

    results: List[Dict[str, float]] = []
    reports: Dict[str, dict] = {}

    best_model = None
    best_score = -1

    for model_name, model in models.items():

        print("=" * 70)
        print(f"Training: {model_name}")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(
            model_name=model_name,
            pipeline=pipeline,
            X_test=X_test,
            y_test=y_test,
        )

        results.append(metrics)

        y_pred = pipeline.predict(X_test)

        reports[model_name] = classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0,
        )

        print(f"ROC AUC:   {metrics['roc_auc']:.3f}")
        print(f"PR AUC:    {metrics['pr_auc']:.3f}")
        print(f"Recall:    {metrics['recall']:.3f}")
        print(f"F1-score:  {metrics['f1_score']:.3f}")
        print()

        if metrics["pr_auc"] > best_score:
            best_score = metrics["pr_auc"]
            best_model = pipeline

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by="pr_auc",
        ascending=False,
    )

    return results_df, reports, best_model


def save_outputs(
    results_df: pd.DataFrame,
    reports: Dict[str, dict],
    best_model: Pipeline,
) -> None:
    results_df.to_csv(METRICS_PATH, index=False)

    with open(REPORTS_PATH, "w") as f:
        json.dump(reports, f, indent=4)

    with open(BEST_MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)

    print("=" * 70)
    print("Training completed successfully")
    print()
    print("Model ranking:")
    print(results_df)
    print()
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Reports saved to: {REPORTS_PATH}")
    print(f"Best model saved to: {BEST_MODEL_PATH}")


# MAIN

def main() -> None:
    create_directories()

    df = load_data(DATA_PATH)

    print("Dataset loaded successfully")
    print(f"Shape: {df.shape}")
    print()

    validate_dataset(df)

    print("Target distribution:")
    print(df[TARGET].value_counts(normalize=True))
    print()

    X_train, X_test, y_train, y_test = temporal_train_test_split(df)

    results_df, reports, best_model = train_models(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    save_outputs(
        results_df=results_df,
        reports=reports,
        best_model=best_model,
    )


if __name__ == "__main__":
    main()
