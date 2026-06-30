import json
import pickle
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# CONFIGURATION
TARGET = "outbreak_7d"
DECISION_THRESHOLD = 0.50

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "raw" / "epiwatch_epidemiological_surveillance_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "best_outbreak_prediction_model.pkl"

REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

DROP_COLUMNS = [
    "outbreak_7d",
    "future_reported_cases_7d",
    "future_incidence_7d",
    "future_effective_R_7d",
    "true_cases",
    "date",
]


# LOADERS AND VALIDATION
def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Run src/simulate_data.py first."
        )

    df = pd.read_csv(DATA_PATH)

    if "date" not in df.columns:
        raise ValueError("The dataset must contain a 'date' column.")

    df["date"] = pd.to_datetime(df["date"])

    return df


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run src/train_model.py first."
        )

    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    return model


def validate_inputs(df: pd.DataFrame) -> None:
    required_columns = set(DROP_COLUMNS + [TARGET])
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if df[TARGET].nunique() < 2:
        raise ValueError(
            "The target has only one class. Evaluation requires both classes."
        )


# DATA SPLIT

def temporal_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.Series, pd.Timestamp]:
    split_date = df["date"].quantile(1 - test_size)

    test_mask = df["date"] > split_date

    X = df.drop(columns=DROP_COLUMNS)
    y = df[TARGET]

    X_test = X.loc[test_mask]
    y_test = y.loc[test_mask]

    return X_test, y_test, split_date


# EVALUATION

def predict_with_threshold(model, X_test: pd.DataFrame):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= DECISION_THRESHOLD).astype(int)

    return y_pred, y_proba


def calculate_metrics(
    y_test: pd.Series,
    y_pred,
    y_proba,
) -> pd.DataFrame:
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    metrics = {
        "decision_threshold": DECISION_THRESHOLD,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    return pd.DataFrame([metrics])


def save_predictions(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred,
    y_proba,
) -> None:
    predictions = X_test.copy()
    predictions["actual_outbreak_7d"] = y_test.values
    predictions["predicted_outbreak_7d"] = y_pred
    predictions["predicted_probability"] = y_proba

    predictions.to_csv(
        TABLES_DIR / "test_set_predictions.csv",
        index=False,
    )


def save_classification_report(y_test, y_pred) -> None:
    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(TABLES_DIR / "final_classification_report.csv")


def save_confusion_matrix(y_test, y_pred) -> None:
    cm = confusion_matrix(y_test, y_pred)

    cm_df = pd.DataFrame(
        cm,
        index=["Actual no outbreak", "Actual outbreak"],
        columns=["Predicted no outbreak", "Predicted outbreak"],
    )

    cm_df.to_csv(TABLES_DIR / "final_confusion_matrix.csv")

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No outbreak", "Outbreak"],
    )

    display.plot(values_format="d")
    plt.title("Confusion Matrix - Final Model")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def save_roc_curve(y_test, y_proba) -> None:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")

    plt.title("ROC Curve - Final Outbreak Prediction Model")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "roc_curve.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def save_precision_recall_curve(y_test, y_proba) -> None:
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.3f}")

    plt.title("Precision-Recall Curve - Final Outbreak Prediction Model")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "precision_recall_curve.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def save_threshold_analysis(y_test, y_proba) -> None:
    thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]
    rows = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        rows.append({
            "threshold": threshold,
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
        })

    threshold_df = pd.DataFrame(rows)
    threshold_df.to_csv(
        TABLES_DIR / "threshold_analysis.csv",
        index=False,
    )

    plt.figure(figsize=(8, 5))
    plt.plot(threshold_df["threshold"], threshold_df["precision"], marker="o", label="Precision")
    plt.plot(threshold_df["threshold"], threshold_df["recall"], marker="o", label="Recall")
    plt.plot(threshold_df["threshold"], threshold_df["f1_score"], marker="o", label="F1-score")

    plt.title("Threshold Sensitivity Analysis")
    plt.xlabel("Decision threshold")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "threshold_analysis.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def save_metadata(split_date, X_test, y_test) -> None:
    metadata = {
        "target": TARGET,
        "decision_threshold": DECISION_THRESHOLD,
        "test_size": 0.2,
        "temporal_split_date": str(split_date),
        "test_rows": int(X_test.shape[0]),
        "test_positive_rate": float(y_test.mean()),
        "evaluation_type": "temporal holdout",
    }

    with open(TABLES_DIR / "evaluation_metadata.json", "w") as file:
        json.dump(metadata, file, indent=4)


# MAIN

def main() -> None:
    df = load_dataset()
    validate_inputs(df)

    model = load_model()

    X_test, y_test, split_date = temporal_test_split(df)

    y_pred, y_proba = predict_with_threshold(model, X_test)

    metrics_df = calculate_metrics(y_test, y_pred, y_proba)
    metrics_df.to_csv(TABLES_DIR / "final_model_metrics.csv", index=False)

    save_predictions(X_test, y_test, y_pred, y_proba)
    save_classification_report(y_test, y_pred)
    save_confusion_matrix(y_test, y_pred)
    save_roc_curve(y_test, y_proba)
    save_precision_recall_curve(y_test, y_proba)
    save_threshold_analysis(y_test, y_proba)
    save_metadata(split_date, X_test, y_test)

    print("Final model evaluation completed successfully")
    print()
    print(metrics_df)
    print()
    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"Tables saved to: {TABLES_DIR}")
  
if __name__ == "__main__":
    main()
