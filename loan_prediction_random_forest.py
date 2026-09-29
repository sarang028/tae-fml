"""Loan Approval Prediction using Random Forest."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_FILE = Path("loan_data.csv")
RANDOM_STATE = 42


def create_demo_dataset(size=1000):
    """Create a reproducible demo dataset when no CSV is supplied."""
    rng = np.random.default_rng(RANDOM_STATE)

    data = pd.DataFrame({
        "Gender": rng.choice(["Male", "Female"], size),
        "Married": rng.choice(["Yes", "No"], size),
        "Dependents": rng.choice(["0", "1", "2", "3+"], size),
        "Education": rng.choice(["Graduate", "Not Graduate"], size),
        "Self_Employed": rng.choice(["Yes", "No"], size),
        "ApplicantIncome": rng.integers(2000, 15000, size),
        "CoapplicantIncome": rng.integers(0, 8000, size),
        "LoanAmount": rng.integers(50, 500, size),
        "Loan_Amount_Term": rng.choice([120, 180, 240, 300, 360], size),
        "Credit_History": rng.choice([0, 1], size, p=[0.18, 0.82]),
        "Property_Area": rng.choice(["Urban", "Semiurban", "Rural"], size),
    })

    score = (
        2.5 * data["Credit_History"]
        + data["ApplicantIncome"] / 15000
        + data["CoapplicantIncome"] / 12000
        - data["LoanAmount"] / 500
        + (data["Education"] == "Graduate").astype(int) * 0.3
        + (data["Property_Area"] == "Semiurban").astype(int) * 0.2
    )
    probability = 1 / (1 + np.exp(-(score - 1.8)))
    data["Loan_Status"] = (rng.random(size) < probability).astype(int)

    return data


def load_data():
    """Load CSV data when available, otherwise use demo data."""
    if DATA_FILE.exists():
        print(f"Using dataset: {DATA_FILE}")
        return pd.read_csv(DATA_FILE)

    print("loan_data.csv not found. Using a reproducible demo dataset.")
    return create_demo_dataset()


def preprocess_data(data):
    """Clean data and encode categorical features."""
    data = data.copy()
    data.columns = [column.strip().replace(" ", "_") for column in data.columns]

    for column in data.columns:
        if data[column].dtype == "object":
            data[column] = data[column].fillna(data[column].mode()[0])
        else:
            data[column] = data[column].fillna(data[column].median())

    target_candidates = ["Loan_Status", "Loan_Approved", "loan_status"]
    target = next(
        (column for column in target_candidates if column in data.columns),
        None,
    )

    if target is None:
        raise ValueError(
            "Target column not found. Use Loan_Status or Loan_Approved."
        )

    if data[target].dtype == "object":
        mapping = {
            "Y": 1, "N": 0, "Yes": 1, "No": 0,
            "Approved": 1, "Rejected": 0,
        }
        data[target] = data[target].astype(str).str.strip().map(mapping)

    data = data.dropna(subset=[target])

    X = data.drop(columns=[target])
    y = data[target].astype(int)

    for column in X.select_dtypes(include=["object"]).columns:
        encoder = LabelEncoder()
        X[column] = encoder.fit_transform(X[column].astype(str))

    return X, y


def main():
    data = load_data()
    X, y = preprocess_data(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=4,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("=" * 60)
    print("RANDOM FOREST BASED LOAN PREDICTION")
    print("=" * 60)
    print(f"Training samples : {len(X_train)}")
    print(f"Testing samples  : {len(X_test)}")
    print(f"Features         : {X.shape[1]}")
    print(f"Accuracy         : {accuracy:.4f}")
    print()
    print(classification_report(
        y_test,
        predictions,
        target_names=["Loan Rejected", "Loan Approved"],
        digits=4,
    ))

    matrix = confusion_matrix(y_test, predictions)
    print("Confusion Matrix:")
    print(matrix)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Rejected", "Approved"],
    )
    display.plot(values_format="d")
    display.ax_.set_title("Random Forest - Loan Prediction")
    plt.tight_layout()
    plt.savefig("loan_prediction_confusion_matrix.png", dpi=180)
    plt.close()

    importance = pd.Series(
        model.feature_importances_,
        index=X.columns,
    ).sort_values(ascending=False)

    print()
    print("Feature Importance:")
    print(importance)

    importance.sort_values().plot(
        kind="barh",
        figsize=(9, 6),
        title="Random Forest Feature Importance",
    )
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=180)
    plt.close()

    print("\nGenerated:")
    print("- loan_prediction_confusion_matrix.png")
    print("- feature_importance.png")
    print("\nEducational project only; not a real lending decision system.")


if __name__ == "__main__":
    main()
