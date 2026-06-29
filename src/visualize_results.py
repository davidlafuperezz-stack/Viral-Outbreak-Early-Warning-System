from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# CONFIGURATION

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "raw" / "epiwatch_epidemiological_surveillance_dataset.csv"
METRICS_PATH = BASE_DIR / "reports" / "tables" / "model_performance_metrics.csv"

FIGURES_DIR = BASE_DIR / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300


# LOADERS

def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Run src/simulate_data.py first."
        )

    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])

    return df


def load_metrics() -> pd.DataFrame:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Metrics file not found at {METRICS_PATH}. Run src/train_model.py first."
        )

    return pd.read_csv(METRICS_PATH)


def save_figure(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=DPI, bbox_inches="tight")
    plt.close()


# FIGURES

def plot_outbreak_distribution(df: pd.DataFrame) -> None:
    counts = df["outbreak_7d"].value_counts().sort_index()

    plt.figure(figsize=(7, 5))
    counts.plot(kind="bar")

    plt.title("Distribution of 7-Day Outbreak Risk Classes")
    plt.xlabel("Outbreak within 7 days")
    plt.ylabel("Number of observations")
    plt.xticks(
        ticks=[0, 1],
        labels=["No outbreak", "Outbreak"],
        rotation=0,
    )

    save_figure("outbreak_distribution.png")


def plot_model_performance(metrics: pd.DataFrame) -> None:
    performance_metrics = ["roc_auc", "pr_auc", "recall", "f1_score"]

    metrics_plot = metrics.set_index("model")[performance_metrics]

    plt.figure(figsize=(10, 6))
    metrics_plot.plot(kind="bar")

    plt.title("Model Performance Comparison")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Metric")

    save_figure("model_performance.png")


def plot_total_cases_over_time(df: pd.DataFrame) -> None:
    daily_cases = (
        df.groupby("date")["reported_cases"]
        .sum()
        .reset_index()
    )

    plt.figure(figsize=(11, 5))
    plt.plot(daily_cases["date"], daily_cases["reported_cases"])

    plt.title("Total Reported Cases Over Time")
    plt.xlabel("Date")
    plt.ylabel("Reported cases")

    save_figure("reported_cases_over_time.png")


def plot_regional_case_trajectories(df: pd.DataFrame) -> None:
    top_regions = (
        df.groupby("region")["reported_cases"]
        .sum()
        .sort_values(ascending=False)
        .head(6)
        .index
    )

    subset = df[df["region"].isin(top_regions)]

    plt.figure(figsize=(12, 6))

    for region in top_regions:
        region_data = subset[subset["region"] == region]
        plt.plot(
            region_data["date"],
            region_data["reported_cases"],
            label=region,
            linewidth=1.5,
        )

    plt.title("Reported Cases Over Time in Highest-Burden Regions")
    plt.xlabel("Date")
    plt.ylabel("Reported cases")
    plt.legend(title="Region")

    save_figure("regional_case_trajectories.png")


def plot_incidence_by_region(df: pd.DataFrame) -> None:
    regional_incidence = (
        df.groupby("region")["incidence_per_100k"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(11, 6))
    regional_incidence.plot(kind="bar")

    plt.title("Average Incidence per 100,000 by Region")
    plt.xlabel("Region")
    plt.ylabel("Average incidence per 100,000")
    plt.xticks(rotation=45, ha="right")

    save_figure("incidence_by_region.png")


def plot_feature_correlation(df: pd.DataFrame) -> None:
    selected_features = [
        "reported_cases",
        "Rt",
        "effective_R",
        "spatial_pressure",
        "mobility_index",
        "vaccination_rate",
        "temperature",
        "humidity",
        "rainfall",
        "healthcare_pressure",
        "incidence_per_100k",
        "outbreak_7d",
    ]

    corr = df[selected_features].corr()

    plt.figure(figsize=(10, 8))
    plt.imshow(corr, aspect="auto")
    plt.colorbar(label="Correlation coefficient")

    plt.xticks(
        range(len(selected_features)),
        selected_features,
        rotation=45,
        ha="right",
    )
    plt.yticks(range(len(selected_features)), selected_features)

    plt.title("Correlation Matrix of Epidemiological and Environmental Features")

    save_figure("feature_correlation_matrix.png")


def plot_best_model_ranking(metrics: pd.DataFrame) -> None:
    ranked = metrics.sort_values("pr_auc", ascending=True)

    plt.figure(figsize=(9, 5))
    plt.barh(ranked["model"], ranked["pr_auc"])

    plt.title("Model Ranking by Precision-Recall AUC")
    plt.xlabel("PR AUC")
    plt.ylabel("Model")
    plt.xlim(0, 1)

    save_figure("model_ranking_pr_auc.png")


# MAIN

def main() -> None:
    df = load_dataset()
    metrics = load_metrics()

    plot_outbreak_distribution(df)
    plot_model_performance(metrics)
    plot_total_cases_over_time(df)
    plot_regional_case_trajectories(df)
    plot_incidence_by_region(df)
    plot_feature_correlation(df)
    plot_best_model_ranking(metrics)

    print("Figures generated successfully")
    print(f"Saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
