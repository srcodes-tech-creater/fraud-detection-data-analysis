import os
import pandas as pd


BASE_PATH = os.path.join("data", "raw", "Data")


def load_and_merge_data():
    # Transaction data
    transaction_records = pd.read_csv(
        os.path.join(
            BASE_PATH,
            "Transaction Data",
            "transaction_records.csv"
        )
    )

    transaction_metadata = pd.read_csv(
        os.path.join(
            BASE_PATH,
            "Transaction Data",
            "transaction_metadata.csv"
        )
    )

    # Transaction amounts
    amount_data = pd.read_csv(
        os.path.join(
            BASE_PATH,
            "Transaction Amounts",
            "amount_data.csv"
        )
    )

    anomaly_scores = pd.read_csv(
        os.path.join(
            BASE_PATH,
            "Transaction Amounts",
            "anomaly_scores.csv"
        )
    )

    # Fraud information
    fraud_indicators = pd.read_csv(
        os.path.join(
            BASE_PATH,
            "Fraudulent Patterns",
            "fraud_indicators.csv"
        )
    )

    # Suspicious activity
    suspicious_activity = pd.read_csv(
        os.path.join(
            BASE_PATH,
            "Fraudulent Patterns",
            "suspicious_activity.csv"
        )
    )

    # Customer data
    customer_data = pd.read_csv(
        os.path.join(
            BASE_PATH,
            "Customer Profiles",
            "customer_data.csv"
        )
    )

    account_activity = pd.read_csv(
        os.path.join(
            BASE_PATH,
            "Customer Profiles",
            "account_activity.csv"
        )
    )

    # Merchant data
    merchant_data = pd.read_csv(
        os.path.join(
            BASE_PATH,
            "Merchant Information",
            "merchant_data.csv"
        )
    )

    transaction_categories = pd.read_csv(
        os.path.join(
            BASE_PATH,
            "Merchant Information",
            "transaction_category_labels.csv"
        )
    )

    # Merge transaction-based data
    df = transaction_records.merge(
        transaction_metadata,
        on="TransactionID",
        how="left"
    )

    df = df.merge(
        amount_data,
        on="TransactionID",
        how="left"
    )

    df = df.merge(
        anomaly_scores,
        on="TransactionID",
        how="left"
    )

    df = df.merge(
        fraud_indicators,
        on="TransactionID",
        how="left"
    )

    df = df.merge(
        transaction_categories,
        on="TransactionID",
        how="left"
    )

    # Merge customer data
    df = df.merge(
        customer_data,
        on="CustomerID",
        how="left"
    )

    df = df.merge(
        account_activity,
        on="CustomerID",
        how="left"
    )

    df = df.merge(
        suspicious_activity,
        on="CustomerID",
        how="left"
    )

    # Merge merchant data
    df = df.merge(
        merchant_data,
        on="MerchantID",
        how="left"
    )

    return df


def clean_data(df):
    df = df.copy()

    # Convert timestamp to datetime
    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce"
    )

    # Create time-based features
    df["TransactionHour"] = df["Timestamp"].dt.hour
    df["TransactionDay"] = df["Timestamp"].dt.day
    df["TransactionMonth"] = df["Timestamp"].dt.month

    # Remove unnecessary personal columns
    columns_to_drop = [
        "Name",
        "Address",
        "MerchantName",
        "Location",
        "LastLogin"
    ]

    df = df.drop(
        columns=[
            column for column in columns_to_drop
            if column in df.columns
        ]
    )

    # Fill missing numerical values
    numeric_columns = df.select_dtypes(
        include=["number"]
    ).columns

    for column in numeric_columns:
        df[column] = df[column].fillna(
            df[column].median()
        )

    # Fill missing categorical values
    categorical_columns = df.select_dtypes(
        include=["object"]
    ).columns

    for column in categorical_columns:
        df[column] = df[column].fillna("Unknown")

    return df


def save_processed_data(df):
    output_path = os.path.join(
        "data",
        "processed",
        "merged_fraud_data.csv"
    )

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(f"Processed data saved to: {output_path}")


if __name__ == "__main__":
    data = load_and_merge_data()

    print("Original Data Shape:")
    print(data.shape)

    cleaned_data = clean_data(data)

    print("\nCleaned Data Shape:")
    print(cleaned_data.shape)

    print("\nColumns:")
    print(cleaned_data.columns.tolist())

    save_processed_data(cleaned_data)