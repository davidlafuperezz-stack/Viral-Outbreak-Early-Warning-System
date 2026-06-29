import numpy as np
import pandas as pd
from pathlib import Path

# EpiWatch: Synthetic Spatio-Temporal Epidemiological Simulator
SEED = 42
N_REGIONS = 20
N_DAYS = 365
START_DATE = "2023-01-01"

np.random.seed(SEED)

# PATHS
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = DATA_DIR / "epiwatch_epidemiological_surveillance_dataset.csv"

# REGIONS
regions = [f"Region_{i}" for i in range(N_REGIONS)]
dates = pd.date_range(START_DATE, periods=N_DAYS)

# Synthetic geographic coordinates
latitudes = np.random.uniform(35, 45, size=N_REGIONS)
longitudes = np.random.uniform(-10, 5, size=N_REGIONS)

# DISTANCE-BASED ADJACENCY MATRIX
coords = np.column_stack([latitudes, longitudes])

dist_matrix = np.zeros((N_REGIONS, N_REGIONS))

for i in range(N_REGIONS):
    for j in range(N_REGIONS):
        dist_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])

adj_matrix = 1 / (dist_matrix + 1e-6)
np.fill_diagonal(adj_matrix, 0)
adj_matrix = adj_matrix / adj_matrix.sum(axis=1, keepdims=True)

# INITIAL CONDITIONS
population = np.random.randint(100_000, 5_000_000, size=N_REGIONS)
population_density = np.random.uniform(50, 2000, size=N_REGIONS)

cases = np.random.randint(5, 30, size=N_REGIONS).astype(float)
Rt = np.random.uniform(1.0, 1.5, size=N_REGIONS)

vaccination_rate = np.random.uniform(0.4, 0.7, size=N_REGIONS)
mobility_index = np.random.uniform(0.3, 0.8, size=N_REGIONS)

healthcare_capacity = np.random.uniform(0.4, 1.0, size=N_REGIONS)
testing_rate = np.random.uniform(0.2, 0.9, size=N_REGIONS)

temperature_baseline = np.random.normal(18, 5, size=N_REGIONS)
humidity_baseline = np.random.uniform(40, 80, size=N_REGIONS)

intervention_active = np.zeros(N_REGIONS)
intervention_days = np.zeros(N_REGIONS)

records = []

# SIMULATION LOOP
for t, date in enumerate(dates):

    new_cases_global = np.zeros(N_REGIONS)
    current_cases = cases.copy()

    seasonal_component = np.sin(2 * np.pi * t / 365)

    for i, region in enumerate(regions):

        # ClimatE
        temperature = (
            temperature_baseline[i]
            + 8 * seasonal_component
            + np.random.normal(0, 2)
        )

        humidity = (
            humidity_baseline[i]
            - 10 * seasonal_component
            + np.random.normal(0, 5)
        )

        humidity = np.clip(humidity, 20, 100)
        rainfall = max(0, np.random.gamma(shape=2, scale=2))

        # Spatial pressure
        spatial_pressure = np.sum(adj_matrix[i] * current_cases)

        # Imported cases
        imported_cases = np.random.poisson(
            lam=mobility_index[i] * spatial_pressure * 0.02
        )

        # Rt stochastic evolution
        Rt[i] = Rt[i] * np.random.normal(1.0, 0.025)
        Rt[i] = np.clip(Rt[i], 0.5, 3.0)

        # Climate modifier
        climate_factor = 1.0

        if temperature < 10:
            climate_factor += 0.20

        if humidity > 80:
            climate_factor += 0.10

        if rainfall > 8:
            climate_factor += 0.05

        # Intervention mechanism
        incidence_per_100k_previous = (current_cases[i] / population[i]) * 100_000

        if incidence_per_100k_previous > 50 or current_cases[i] > 200:
            intervention_active[i] = 1
            intervention_days[i] = 21

        if intervention_days[i] > 0:
            intervention_effect = 0.70
            intervention_days[i] -= 1
        else:
            intervention_active[i] = 0
            intervention_effect = 1.0

        # Mobility adapts to interventions
        mobility_index[i] = np.clip(
            mobility_index[i] * intervention_effect + np.random.normal(0, 0.01),
            0.05,
            1.0
        )

        # Slow vaccination increase
        vaccination_rate[i] = np.clip(
            vaccination_rate[i] + np.random.normal(0.0005, 0.0003),
            0,
            1
        )

        vaccine_effect = 1 - vaccination_rate[i] * 0.70

        # Effective transmission
        effective_R = (
            Rt[i]
            * mobility_index[i]
            * vaccine_effect
            * climate_factor
        )

        density_factor = 1 + (population_density[i] / 2000) * 0.25

        expected_cases = (
            current_cases[i] * effective_R * density_factor
            + spatial_pressure * 0.015
            + imported_cases
        )

        # Stochastic epidemic process
        new_cases = np.random.poisson(max(expected_cases, 1))

        # Susceptible saturation
        susceptible_fraction = max(0.05, 1 - current_cases[i] / population[i])
        new_cases = int(new_cases * susceptible_fraction)

        new_cases = max(1, new_cases)

        # Observed cases
        detection_probability = np.clip(
            0.3 + testing_rate[i] * 0.6,
            0.1,
            0.95
        )

        observed_cases = np.random.binomial(new_cases, detection_probability)

        new_cases_global[i] = new_cases

        incidence_per_100k = (observed_cases / population[i]) * 100_000

        healthcare_pressure = (
            observed_cases / (healthcare_capacity[i] * 1000)
        )

        records.append([
            region,
            date,
            latitudes[i],
            longitudes[i],
            population[i],
            population_density[i],
            observed_cases,
            new_cases,
            Rt[i],
            effective_R,
            spatial_pressure,
            imported_cases,
            temperature,
            humidity,
            rainfall,
            mobility_index[i],
            vaccination_rate[i],
            testing_rate[i],
            healthcare_capacity[i],
            healthcare_pressure,
            incidence_per_100k,
            intervention_active[i]
        ])

    cases = new_cases_global.copy()

# DATAFRAME
df = pd.DataFrame(records, columns=[
    "region",
    "date",
    "latitude",
    "longitude",
    "population",
    "population_density",
    "reported_cases",
    "true_cases",
    "Rt",
    "effective_R",
    "spatial_pressure",
    "imported_cases",
    "temperature",
    "humidity",
    "rainfall",
    "mobility_index",
    "vaccination_rate",
    "testing_rate",
    "healthcare_capacity",
    "healthcare_pressure",
    "incidence_per_100k",
    "intervention_active"
])

df = df.sort_values(["region", "date"]).reset_index(drop=True)

# TEMPORAL FEATURES
for lag in [1, 3, 7, 14]:
    df[f"reported_cases_lag_{lag}"] = (
        df.groupby("region")["reported_cases"].shift(lag)
    )

for window in [7, 14]:
    df[f"reported_cases_rolling_mean_{window}"] = (
        df.groupby("region")["reported_cases"]
        .rolling(window)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df[f"reported_cases_rolling_std_{window}"] = (
        df.groupby("region")["reported_cases"]
        .rolling(window)
        .std()
        .reset_index(level=0, drop=True)
    )

df["growth_rate_7d"] = (
    df["reported_cases"] / (df["reported_cases_lag_7"] + 1)
)

df["incidence_growth_7d"] = (
    df.groupby("region")["incidence_per_100k"].pct_change(periods=7)
)

# FUTURE TARGETS
df["future_reported_cases_7d"] = (
    df.groupby("region")["reported_cases"].shift(-7)
)

df["future_incidence_7d"] = (
    df.groupby("region")["incidence_per_100k"].shift(-7)
)

df["future_effective_R_7d"] = (
    df.groupby("region")["effective_R"].shift(-7)
)

df["outbreak_7d"] = (
    (df["future_reported_cases_7d"] > 150) |
    (df["future_incidence_7d"] > 25) |
    (df["future_effective_R_7d"] > 1.25)
).astype(int)

# CLEANING
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna().reset_index(drop=True)

# SAVE
df.to_csv(OUTPUT_PATH, index=False)

print("Dataset generated successfully")
print(f"Shape: {df.shape}")
print(f"Saved to: {OUTPUT_PATH}")
print()
print("Outbreak distribution:")
print(df["outbreak_7d"].value_counts(normalize=True))
print()
print(df.head())
