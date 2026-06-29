import numpy as np
import pandas as pd
from pathlib import Path

# =========================
# CONFIGURATION
# =========================
SEED = 42
N_REGIONS = 20
N_DAYS = 365
START_DATE = "2023-01-01"

np.random.seed(SEED)

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = DATA_DIR / "epiwatch_epidemiological_surveillance_dataset.csv"

# REGIONS AND DATES
regions = [f"Region_{i}" for i in range(N_REGIONS)]
dates = pd.date_range(START_DATE, periods=N_DAYS)

# =========================
# SPATIAL ADJACENCY MATRIX
adj_matrix = np.random.uniform(0, 1, (N_REGIONS, N_REGIONS))
np.fill_diagonal(adj_matrix, 0)
adj_matrix = adj_matrix / adj_matrix.sum(axis=1, keepdims=True)

# =========================
# INITIAL CONDITIONS
# =========================
cases = np.random.randint(5, 30, size=N_REGIONS).astype(float)
Rt = np.random.uniform(1.0, 1.5, size=N_REGIONS)

population = np.random.randint(100_000, 5_000_000, size=N_REGIONS)
density = np.random.uniform(50, 2000, size=N_REGIONS)

vaccination = np.random.uniform(0.4, 0.7, size=N_REGIONS)
mobility = np.random.uniform(0.3, 0.7, size=N_REGIONS)

# Region-specific climate baselines
temperature_baseline = np.random.normal(18, 5, size=N_REGIONS)
humidity_baseline = np.random.uniform(40, 80, size=N_REGIONS)

data = []

# =========================
# SIMULATION LOOP
# =========================
for t, date in enumerate(dates):

    new_cases_global = np.zeros(N_REGIONS)
    spatial_pressures = np.zeros(N_REGIONS)

    # Seasonal effect
    seasonal_component = np.sin(2 * np.pi * t / 365)

    for i, region in enumerate(regions):

        # =========================
        # REGION-SPECIFIC CLIMATE
        # =========================
        temperature = temperature_baseline[i] + 8 * seasonal_component + np.random.normal(0, 2)
        humidity = humidity_baseline[i] - 10 * seasonal_component + np.random.normal(0, 5)
        rainfall = max(0, np.random.gamma(shape=2, scale=2))

        humidity = np.clip(humidity, 20, 100)

        # =========================
        # SPATIAL INFECTION PRESSURE
        # =========================
        neighbor_pressure = np.sum(adj_matrix[i] * cases)
        spatial_pressures[i] = neighbor_pressure

        # =========================
        # DYNAMIC RT
        # =========================
        Rt[i] = max(0.5, Rt[i] * np.random.normal(1.0, 0.03))

        # =========================
        # CLIMATE EFFECT
        # =========================
        climate_factor = 1.0

        if temperature < 10:
            climate_factor += 0.20

        if humidity > 80:
            climate_factor += 0.10

        if rainfall > 8:
            climate_factor += 0.05

        # =========================
        # INTERVENTION EFFECT
        # =========================
        intervention = 1.0

        if cases[i] > 200:
            intervention *= 0.70
            vaccination[i] = min(1.0, vaccination[i] + 0.002)

        mobility[i] = np.clip(mobility[i] * intervention + np.random.normal(0, 0.01), 0.05, 1.0)

        vaccine_effect = 1 - (vaccination[i] * 0.70)

        # =========================
        # EFFECTIVE TRANSMISSION
        # =========================
        effective_R = Rt[i] * mobility[i] * vaccine_effect * climate_factor

        spatial_spread = neighbor_pressure * 0.01

        # =========================
        # EPIDEMIC DYNAMICS
        # =========================
        noise = np.random.normal(0, 5)

        new_cases = cases[i] * effective_R + spatial_spread + noise

        # Saturation effect
        susceptible_fraction = max(0.05, 1 - cases[i] / population[i])
        new_cases *= susceptible_fraction

        new_cases = max(1, int(new_cases))
        new_cases_global[i] = new_cases

        # =========================
        # STORE DAILY RECORD
        # =========================
        incidence_per_100k = (new_cases / population[i]) * 100_000

        data.append([
            region,
            date,
            new_cases,
            population[i],
            density[i],
            temperature,
            humidity,
            rainfall,
            mobility[i],
            vaccination[i],
            Rt[i],
            neighbor_pressure,
            incidence_per_100k,
            effective_R
        ])

    # Simultaneous update
    cases = new_cases_global.copy()

# =========================
# DATAFRAME
# =========================
df = pd.DataFrame(data, columns=[
    "region",
    "date",
    "cases",
    "population",
    "population_density",
    "temperature",
    "humidity",
    "rainfall",
    "mobility_index",
    "vaccination_rate",
    "Rt",
    "spatial_pressure",
    "incidence_per_100k",
    "effective_R"
])

# =========================
# TEMPORAL FEATURES
# =========================
df = df.sort_values(["region", "date"]).reset_index(drop=True)

df["cases_lag_1"] = df.groupby("region")["cases"].shift(1)
df["cases_lag_7"] = df.groupby("region")["cases"].shift(7)
df["cases_lag_14"] = df.groupby("region")["cases"].shift(14)

df["cases_rolling_mean_7"] = (
    df.groupby("region")["cases"]
    .rolling(window=7)
    .mean()
    .reset_index(level=0, drop=True)
)

df["cases_rolling_std_7"] = (
    df.groupby("region")["cases"]
    .rolling(window=7)
    .std()
    .reset_index(level=0, drop=True)
)

df["cases_growth_7d"] = df["cases"] / (df["cases_lag_7"] + 1)

# =========================
# FUTURE OUTBREAK TARGET
# =========================
df["future_cases_7d"] = df.groupby("region")["cases"].shift(-7)

df["outbreak_7d"] = (
    (df["future_cases_7d"] > 150) |
    (df.groupby("region")["Rt"].shift(-7) > 1.8)
).astype(int)

# Remove rows with missing lag/future values
df = df.dropna().reset_index(drop=True)

# =========================
# SAVE DATASET
# =========================
df.to_csv(OUTPUT_PATH, index=False)

print("Dataset generated successfully")
print(f"Shape: {df.shape}")
print(f"Saved to: {OUTPUT_PATH}")
print(df.head())
