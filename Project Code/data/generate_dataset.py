"""
generate_dataset.py
--------------------
Generates a synthetic flood-prediction dataset that mirrors the structure
described in the "Rising Waters" project spec (annual rainfall, cloud
visibility, temperature, humidity, seasonal rainfall -> flood class).

Why synthetic data?
The original project references a public Kaggle dataset
(kaggle.com/arbethi/rainfall-dataset). Kaggle requires authenticated,
browser-based download and is not reachable from this build environment,
so this script produces a statistically realistic stand-in with the same
columns, ranges, and a genuine (noisy, non-trivial) relationship between
weather features and flood outcome. Swap this file out for the real
Kaggle CSV at any time -- the rest of the pipeline (train_model.py, app.py)
only cares about the column names below.

Run:
    python generate_dataset.py
Output:
    flood_dataset.csv  (in this same folder)
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_ROWS = 1500

rng = np.random.default_rng(RANDOM_SEED)


def generate():
    # --- Base feature generation -------------------------------------------------
    annual_rainfall = rng.normal(1500, 450, N_ROWS).clip(200, 4500)          # mm/year
    seasonal_rainfall = (annual_rainfall * rng.uniform(0.45, 0.75, N_ROWS))   # Jun-Sep mm
    cloud_visibility = rng.normal(6, 2.5, N_ROWS).clip(0.2, 15)               # km
    temperature = rng.normal(28, 4, N_ROWS).clip(10, 45)                      # deg C
    humidity = rng.normal(70, 12, N_ROWS).clip(20, 100)                       # %

    # --- Flood-risk score (latent, then thresholded with noise) ------------------
    # Higher rainfall & humidity, lower visibility => higher flood risk.
    score = (
        0.0028 * annual_rainfall
        + 0.0035 * seasonal_rainfall
        - 0.18 * cloud_visibility
        + 0.03 * humidity
        - 0.01 * temperature
        + rng.normal(0, 1.1, N_ROWS)  # noise so it's not perfectly separable
    )

    threshold = np.percentile(score, 60)  # ~40% flood-positive class
    flood_class = (score > threshold).astype(int)

    df = pd.DataFrame(
        {
            "AnnualRainfall": annual_rainfall.round(1),
            "CloudVisibility": cloud_visibility.round(2),
            "Temperature": temperature.round(1),
            "Humidity": humidity.round(1),
            "SeasonalRainfall": seasonal_rainfall.round(1),
            "class": flood_class,
        }
    )

    # --- Sprinkle in some realistic missing values & light outliers --------------
    for col in ["AnnualRainfall", "CloudVisibility", "Humidity"]:
        miss_idx = rng.choice(df.index, size=int(0.015 * N_ROWS), replace=False)
        df.loc[miss_idx, col] = np.nan

    outlier_idx = rng.choice(df.index, size=15, replace=False)
    df.loc[outlier_idx, "AnnualRainfall"] *= rng.uniform(1.8, 2.4, size=15)

    return df


if __name__ == "__main__":
    dataset = generate()
    out_path = "flood_dataset.csv"
    dataset.to_csv(out_path, index=False)
    print(f"Saved {len(dataset)} rows to {out_path}")
    print(dataset["class"].value_counts(normalize=True).rename("class_ratio"))
