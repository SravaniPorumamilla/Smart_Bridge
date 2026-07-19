"""
train_model.py
---------------
End-to-end pipeline for the "Rising Waters" flood-prediction project.

Epic 2 - Visualizing & Analysing the Data
Epic 3 - Data Pre-processing
Epic 4 - Model Building (Decision Tree, Random Forest, KNN, XGBoost)

Outputs (written to this folder):
    eda_distribution.png     - univariate distribution plots
    eda_boxplots.png         - box plots (outlier view)
    eda_heatmap.png          - multivariate correlation heat map
    floods.save              - best trained model   (joblib)
    transform.save           - fitted StandardScaler (joblib)
    model_comparison.txt     - accuracy / reports for all 4 models

Run:
    python train_model.py
"""

import os

import joblib
import matplotlib

matplotlib.use("Agg")  # headless plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost
from sklearn import ensemble, neighbors, tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "flood_dataset.csv")
FEATURES = ["AnnualRainfall", "CloudVisibility", "Temperature", "Humidity", "SeasonalRainfall"]
TARGET = "class"


# ---------------------------------------------------------------------------
# Epic 1 / Story 1 - Data Collection
# ---------------------------------------------------------------------------
def load_data():
    df = pd.read_csv(DATA_PATH)
    print("Shape:", df.shape)
    print(df.head())
    print(df.info())
    print(df.describe())
    return df


# ---------------------------------------------------------------------------
# Epic 2 - Visualizing & Analysing the Data
# ---------------------------------------------------------------------------
def run_eda(df: pd.DataFrame):
    # Univariate: distribution plots
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, col in zip(axes.flat, FEATURES):
        sns.histplot(df[col].dropna(), kde=True, ax=ax, color="#2b6cb0")
        ax.set_title(f"Distribution: {col}")
    axes.flat[-1].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(BASE_DIR, "eda_distribution.png"), dpi=110)
    plt.close(fig)

    # Univariate: box plots (outlier detection)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df[FEATURES], ax=ax, palette="Set2")
    ax.set_title("Box Plots - Outlier Detection")
    fig.tight_layout()
    fig.savefig(os.path.join(BASE_DIR, "eda_boxplots.png"), dpi=110)
    plt.close(fig)

    # Multivariate: correlation heat map
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[FEATURES + [TARGET]].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Feature Correlation Heat Map")
    fig.tight_layout()
    fig.savefig(os.path.join(BASE_DIR, "eda_heatmap.png"), dpi=110)
    plt.close(fig)

    print("EDA plots saved: eda_distribution.png, eda_boxplots.png, eda_heatmap.png")


# ---------------------------------------------------------------------------
# Epic 3 - Data Pre-processing
# ---------------------------------------------------------------------------
def cap_outliers_iqr(series: pd.Series) -> pd.Series:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return series.clip(lower=lower, upper=upper)


def preprocess(df: pd.DataFrame):
    df = df.copy()

    # Story 1: handle missing values (median impute - robust to outliers/skew)
    print("\nMissing values before imputation:\n", df.isnull().sum())
    for col in FEATURES:
        df[col] = df[col].fillna(df[col].median())

    # Story 2: handle outliers via IQR capping
    for col in FEATURES:
        df[col] = cap_outliers_iqr(df[col])

    # Story 3: categorical encoding - not required here (all-numeric weather
    # features), included for completeness / template purposes:
    # from sklearn.preprocessing import LabelEncoder
    # df['some_category'] = LabelEncoder().fit_transform(df['some_category'])

    # Story 4: independent / dependent split
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Story 5: feature scaling (fit on train, applied to both)
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, sc


# ---------------------------------------------------------------------------
# Epic 4 - Model Building
# ---------------------------------------------------------------------------
def evaluate(name, model, X_test, y_test, log_lines):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    log_lines.append(f"\n===== {name} =====")
    log_lines.append(f"Accuracy: {acc * 100:.2f}%")
    log_lines.append(f"Confusion Matrix:\n{cm}")
    log_lines.append(f"Classification Report:\n{report}")

    print(f"[{name}] Accuracy: {acc * 100:.2f}%")
    return acc, y_pred


def decisiontree(X_train, X_test, y_train, y_test, log_lines):
    dtree = tree.DecisionTreeClassifier(random_state=42)
    dtree.fit(X_train, y_train)
    acc, _ = evaluate("Decision Tree", dtree, X_test, y_test, log_lines)
    return dtree, acc


def randomForest(X_train, X_test, y_train, y_test, log_lines):
    rf = ensemble.RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    acc, _ = evaluate("Random Forest", rf, X_test, y_test, log_lines)
    return rf, acc


def KNN(X_train, X_test, y_train, y_test, log_lines):
    knn = neighbors.KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    acc, _ = evaluate("K-Nearest Neighbors", knn, X_test, y_test, log_lines)
    return knn, acc


def xgboost_model(X_train, X_test, y_train, y_test, log_lines):
    xgb = xgboost.XGBClassifier(eval_metric="logloss", random_state=42)
    xgb.fit(X_train, y_train)
    acc, _ = evaluate("XGBoost", xgb, X_test, y_test, log_lines)
    return xgb, acc


def compareModel(results, log_lines):
    log_lines.append("\n===== Model Comparison =====")
    print("\n===== Model Comparison =====")
    for name, acc in results.items():
        line = f"{name}: {acc * 100:.2f}% accuracy"
        log_lines.append(line)
        print(line)
    best_name = max(results, key=results.get)
    log_lines.append(f"\nBest model selected: {best_name}")
    print(f"\nBest model selected: {best_name}")
    return best_name


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    log_lines = []

    df = load_data()
    run_eda(df)
    X_train, X_test, y_train, y_test, scaler = preprocess(df)

    models = {}
    results = {}

    models["Decision Tree"], results["Decision Tree"] = decisiontree(
        X_train, X_test, y_train, y_test, log_lines
    )
    models["Random Forest"], results["Random Forest"] = randomForest(
        X_train, X_test, y_train, y_test, log_lines
    )
    models["K-Nearest Neighbors"], results["K-Nearest Neighbors"] = KNN(
        X_train, X_test, y_train, y_test, log_lines
    )
    models["XGBoost"], results["XGBoost"] = xgboost_model(
        X_train, X_test, y_train, y_test, log_lines
    )

    best_name = compareModel(results, log_lines)

    # Prefer XGBoost when it is tied with / close to the top score, matching
    # the project spec's rationale (generalization + stability), otherwise
    # fall back to whichever model truly scored highest.
    best_acc = results[best_name]
    if results["XGBoost"] >= best_acc - 1e-9:
        best_name = "XGBoost"

    best_model = models[best_name]

    joblib.dump(best_model, os.path.join(BASE_DIR, "floods.save"))
    joblib.dump(scaler, os.path.join(BASE_DIR, "transform.save"))

    with open(os.path.join(BASE_DIR, "model_comparison.txt"), "w") as f:
        f.write("\n".join(log_lines))

    print(f"\nSaved model -> floods.save   (final model: {best_name})")
    print("Saved scaler -> transform.save")


if __name__ == "__main__":
    main()
