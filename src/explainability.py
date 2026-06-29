import pickle
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.inspection import permutation_importance


# CONFIGURATION

RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "raw" / "epiwatch_epidemiological_surveillance_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "best_outbreak_prediction_model.pkl"

REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "outbreak_7d"

DROP_COLUMNS = [
    "outbreak_7d",
    "future_reported_cases_7d",
    "future_incidence_7d",
    "future_effective_R_7d",
    "true_cases",
    "date",
]


# LOADERS

def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Run src/simulate_data.py first."
        )

    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])

    return df


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run src/train_model.py first."
        )

    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    return model


# DATA PREPARATION

def temporal_test_split(df: pd.DataFrame, test_size: float = 0.2):
    split_date = df["date"].quantile(1 - test_size)

    test_mask = df["date"] > split_date

    X = df.drop(columns=DROP_COLUMNS)
    y = df[TARGET]

    X_test = X.loc[test_mask]
    y_test = y.loc[test_mask]

    return X_test, y_test


def get_feature_names(model) -> List[str]:
    preprocessor = model.named_steps["preprocessor"]

    numeric_features = preprocessor.transformers_[0][2]

    categorical_encoder = (
        preprocessor
        .named_transformers_["cat"]
        .named_steps["encoder"]
    )

    categorical_features = categorical_encoder.get_feature_names_out(["region"])

    feature_names = list(numeric_features) + list(categorical_features)

    return feature_names


# EXPLAINABILITY

def plot_permutation_importance(model, X_test, y_test) -> pd.DataFrame:
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring="average_precision",
        n_jobs=-1,
    )

    importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    })

    importance_df = importance_df.sort_values(
        "importance_mean",
        ascending=False,
    )

    importance_df.to_csv(
        TABLES_DIR / "permutation_importance.csv",
        index=False,
    )

    top_features = importance_df.head(15).sort_values("importance_mean")

    plt.figure(figsize=(9, 6))
    plt.barh(
        top_features["feature"],
        top_features["importance_mean"],
        xerr=top_features["importance_std"],
    )

    plt.title("Permutation Importance of Outbreak Prediction Features")
    plt.xlabel("Decrease in Precision-Recall AUC")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "permutation_importance.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    return importance_df


def plot_model_feature_importance(model) -> None:
    estimator = model.named_steps["model"]

    if not hasattr(estimator, "feature_importances_"):
        print("The selected model does not expose feature_importances_. Skipping.")
        return

    feature_names = get_feature_names(model)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": estimator.feature_importances_,
    })

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False,
    )

    importance_df.to_csv(
        TABLES_DIR / "model_feature_importance.csv",
        index=False,
    )

    top_features = importance_df.head(20).sort_values("importance")

    plt.figure(figsize=(9, 7))
    plt.barh(top_features["feature"], top_features["importance"])

    plt.title("Model-Based Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "model_feature_importance.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_shap_summary(model, X_test) -> None:
    try:
        import shap
    except ImportError:
        print("SHAP is not installed. Skipping SHAP analysis.")
        print("Install it with: pip install shap")
        return

    estimator = model.named_steps["model"]
    preprocessor = model.named_steps["preprocessor"]

    X_sample = X_test.sample(
        n=min(500, len(X_test)),
        random_state=RANDOM_STATE,
    )

    X_transformed = preprocessor.transform(X_sample)
    feature_names = get_feature_names(model)

    try:
        explainer = shap.Explainer(estimator, X_transformed)
        shap_values = explainer(X_transformed)

        plt.figure()
        shap.summary_plot(
            shap_values,
            X_transformed,
            feature_names=feature_names,
            show=False,
            max_display=20,
        )

        plt.tight_layout()
        plt.savefig(
            FIGURES_DIR / "shap_summary.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    except Exception as error:
        print("SHAP analysis could not be completed.")
        print(f"Reason: {error}")


# MAIN
def main() -> None:
    df = load_dataset()
    model = load_model()

    X_test, y_test = temporal_test_split(df)

    print("Running explainability analysis...")
    print(f"Test set shape: {X_test.shape}")
    print()

    plot_permutation_importance(model, X_test, y_test)
    plot_model_feature_importance(model)
    plot_shap_summary(model, X_test)

    print("Explainability analysis completed successfully.")
    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"Tables saved to: {TABLES_DIR}")


if __name__ == "__main__":
    main()
