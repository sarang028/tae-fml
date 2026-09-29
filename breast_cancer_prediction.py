"""Breast Cancer Prediction using Machine Learning and Boosting Algorithms."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
OUTPUT_DIR = Path("outputs")


def load_data():
    data = load_breast_cancer(as_frame=True)
    return data.data, data.target


def build_models():
    return {
        "Decision Tree": Pipeline([
            ("model", DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE))
        ]),
        "Random Forest": Pipeline([
            ("model", RandomForestClassifier(
                n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
            ))
        ]),
        "AdaBoost": Pipeline([
            ("model", AdaBoostClassifier(
                n_estimators=150, learning_rate=0.8, random_state=RANDOM_STATE
            ))
        ]),
        "Gradient Boosting": Pipeline([
            ("model", GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=2,
                random_state=RANDOM_STATE,
            ))
        ]),
        "HistGradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("model", HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.08,
                max_leaf_nodes=15,
                random_state=RANDOM_STATE,
            ))
        ]),
    }


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    results = []

    print("=" * 70)
    print("BREAST CANCER PREDICTION USING ML AND BOOSTING")
    print("=" * 70)
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")
    print(f"Features        : {X.shape[1]}")
    print()

    for name, model in build_models().items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, predictions)
        roc_auc = roc_auc_score(y_test, probabilities)

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "ROC-AUC": roc_auc,
        })

        print(f"--- {name} ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC-AUC : {roc_auc:.4f}")
        print(classification_report(
            y_test,
            predictions,
            target_names=["Malignant", "Benign"],
            digits=4,
        ))

        matrix = confusion_matrix(y_test, predictions)
        print("Confusion Matrix:")
        print(matrix)
        print()

        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=["Malignant", "Benign"],
        )
        display.plot(values_format="d")
        display.ax_.set_title(f"{name} - Confusion Matrix")
        display.figure_.tight_layout()
        display.figure_.savefig(
            OUTPUT_DIR / f"{name.lower().replace(' ', '_')}_confusion_matrix.png",
            dpi=180,
        )
        plt.close(display.figure_)

    comparison = pd.DataFrame(results)
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    ax = comparison.set_index("Model")[["Accuracy", "ROC-AUC"]].plot(
        kind="bar",
        figsize=(10, 6),
        ylim=(0.80, 1.00),
        rot=20,
    )
    ax.set_ylabel("Score")
    ax.set_title("Breast Cancer Model Comparison")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=180)
    plt.close()

    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(comparison.to_string(index=False))
    print(f"\nResults saved in: {OUTPUT_DIR.resolve()}")
    print("\nEducational project only; not a clinical diagnostic tool.")


if __name__ == "__main__":
    main()
