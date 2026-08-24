import matplotlib.pyplot as plt
import seaborn as sns


def fraud_distribution_chart(df):

    fig, ax = plt.subplots()

    sns.countplot(
        data=df,
        x="FraudIndicator",
        ax=ax
    )

    ax.set_title(
        "Fraud vs Legitimate Transactions"
    )

    ax.set_xlabel(
        "Fraud Indicator"
    )

    ax.set_ylabel(
        "Number of Transactions"
    )

    return fig


def transaction_amount_chart(df):

    fig, ax = plt.subplots()

    sns.histplot(
        data=df,
        x="TransactionAmount",
        hue="FraudIndicator",
        bins=30,
        kde=True,
        ax=ax
    )

    ax.set_title(
        "Transaction Amount Distribution"
    )

    return fig


def anomaly_score_chart(df):

    fig, ax = plt.subplots()

    sns.boxplot(
        data=df,
        x="FraudIndicator",
        y="AnomalyScore",
        ax=ax
    )

    ax.set_title(
        "Anomaly Score by Fraud Status"
    )

    return fig