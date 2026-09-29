"""Example loan approval prediction using Random Forest."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from loan_prediction_random_forest import (
    DATA_FILE,
    RANDOM_STATE,
    create_demo_dataset,
    preprocess_data,
)


def main():
    data = pd.read_csv(DATA_FILE) if DATA_FILE.exists() else create_demo_dataset()
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

    sample = X_test.iloc[[0]]
    prediction = int(model.predict(sample)[0])
    probability = float(model.predict_proba(sample)[0][prediction])

    result = "APPROVED" if prediction == 1 else "REJECTED"

    print("Loan Prediction")
    print("----------------")
    print(f"Prediction : {result}")
    print(f"Confidence : {probability:.2%}")


if __name__ == "__main__":
    main()
