import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve
)

from imblearn.over_sampling import SMOTE


DATA_PATH = "data/processed/merged_fraud_data.csv"
MODEL_PATH = "models/fraud_model.pkl"


def find_target_column(df):

    possible_targets = [
        "FraudIndicator",
        "Fraud",
        "IsFraud",
        "is_fraud",
        "fraud_indicator",
        "Class",
        "class"
    ]

    for column in possible_targets:
        if column in df.columns:
            return column

    raise ValueError(
        f"Fraud target column not found.\n"
        f"Available columns: {list(df.columns)}"
    )


def prepare_target(y):

    # Convert common fraud labels into 0 and 1
    if y.dtype == "object":

        y = (
            y.astype(str)
            .str.strip()
            .str.lower()
        )

        fraud_values = [
            "1",
            "fraud",
            "fraudulent",
            "yes",
            "true"
        ]

        y = y.isin(fraud_values).astype(int)

    else:

        y = pd.to_numeric(
            y,
            errors="coerce"
        )

        y = y.fillna(0).astype(int)

    return y


def train_model():

    print("Loading processed dataset...")

    df = pd.read_csv(DATA_PATH)

    print("Dataset Shape:", df.shape)

    target_column = find_target_column(df)

    print("Target Column:", target_column)

    y = prepare_target(
        df[target_column]
    )

    print("\nFraud Class Distribution:")

    print(
        y.value_counts()
    )

    numeric_columns = (
        df.select_dtypes(
            include=["number"]
        )
        .columns
        .tolist()
    )

    feature_columns = [
        column
        for column in numeric_columns
        if column != target_column
    ]

    X = df[feature_columns].copy()

    X = X.fillna(
        X.median()
    )

    print(
        f"\nTotal Features: "
        f"{len(feature_columns)}"
    )

    # Train-test split
    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
    )

    print(
        "\nOriginal Training Distribution:"
    )

    print(
        y_train.value_counts()
    )

    # Apply SMOTE
    smote = SMOTE(
        random_state=42,
        k_neighbors=3
    )

    X_train_balanced, y_train_balanced = (
        smote.fit_resample(
            X_train,
            y_train
        )
    )

    print(
        "\nBalanced Training Distribution:"
    )

    print(
        y_train_balanced.value_counts()
    )

    # Train model
    print("\nTraining Random Forest Model...")

    model = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=2,
        n_jobs=-1
    )

    model.fit(
        X_train_balanced,
        y_train_balanced
    )

    # Get probabilities
    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    # -----------------------------------
    # FIND BETTER FRAUD THRESHOLD
    # -----------------------------------

    best_threshold = 0.5
    best_f1 = 0

    thresholds = [
        round(x, 2)
        for x in [
            0.10,
            0.15,
            0.20,
            0.25,
            0.30,
            0.35,
            0.40,
            0.45,
            0.50,
            0.55,
            0.60
        ]
    ]

    print("\nThreshold Results:")

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        current_f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        current_recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        print(
            f"Threshold: {threshold:.2f} | "
            f"F1: {current_f1:.4f} | "
            f"Recall: {current_recall:.4f}"
        )

        if current_f1 > best_f1:

            best_f1 = current_f1
            best_threshold = threshold

    # Final predictions using best threshold
    predictions = (
        probabilities >= best_threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    print("\n" + "=" * 50)

    print("FINAL MODEL PERFORMANCE")

    print("=" * 50)

    print(f"Best Threshold: {best_threshold:.2f}")
    print(f"Accuracy:       {accuracy:.4f}")
    print(f"Precision:      {precision:.4f}")
    print(f"Recall:         {recall:.4f}")
    print(f"F1 Score:       {f1:.4f}")

    print("=" * 50)

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    os.makedirs(
        "models",
        exist_ok=True
    )

    # Save everything
    joblib.dump(
        {
            "model": model,
            "features": feature_columns,
            "target_column": target_column,
            "threshold": best_threshold,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        },
        MODEL_PATH
    )

    print("\nModel saved successfully!")
    print(f"Location: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()