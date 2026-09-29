"""Example single-sample prediction using Gradient Boosting."""

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def main():
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=2,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    sample = X_test.iloc[[0]]
    prediction = int(model.predict(sample)[0])
    probability = float(model.predict_proba(sample)[0][prediction])

    labels = {0: "Malignant", 1: "Benign"}

    print("Breast Cancer Prediction")
    print("------------------------")
    print(f"Actual     : {labels[int(y_test.iloc[0])]}")
    print(f"Prediction : {labels[prediction]}")
    print(f"Confidence : {probability:.2%}")


if __name__ == "__main__":
    main()
