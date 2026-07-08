import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Transaction Risk Scoring Dashboard", layout="wide")

st.title("🏦 Transaction Risk Scoring Dashboard")
st.markdown(
    "Simulates a bank transaction monitoring system: generates synthetic transactions, "
    "scores each one for anomaly risk, and flags suspicious activity — the kind of "
    "workflow used in real-time fraud/AML monitoring."
)

# ---------- Sidebar controls ----------
st.sidebar.header("Simulation Settings")
n_transactions = st.sidebar.slider("Number of transactions", 500, 5000, 2000, step=500)
anomaly_rate = st.sidebar.slider("Injected anomaly rate (%)", 1, 15, 5) / 100
seed = st.sidebar.number_input("Random seed", value=42)

np.random.seed(seed)

# ---------- Generate synthetic data ----------
def generate_transactions(n, anomaly_rate):
    n_anomalies = int(n * anomaly_rate)
    n_normal = n - n_anomalies

    normal = pd.DataFrame({
        "amount": np.random.gamma(shape=2, scale=150, size=n_normal),
        "hour": np.random.normal(14, 4, n_normal).clip(0, 23),
        "txn_frequency_24h": np.random.poisson(3, n_normal),
        "is_new_beneficiary": np.random.binomial(1, 0.05, n_normal),
        "label": 0
    })

    anomalies = pd.DataFrame({
        "amount": np.random.gamma(shape=2, scale=150, size=n_anomalies) *
                  np.random.choice([0.05, 8, 15], n_anomalies, p=[0.3, 0.4, 0.3]),
        "hour": np.random.choice([1, 2, 3, 4, 23], n_anomalies),
        "txn_frequency_24h": np.random.poisson(15, n_anomalies),
        "is_new_beneficiary": np.random.binomial(1, 0.7, n_anomalies),
        "label": 1
    })

    df = pd.concat([normal, anomalies], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    df["transaction_id"] = ["TXN" + str(i).zfill(6) for i in range(len(df))]
    return df

df = generate_transactions(n_transactions, anomaly_rate)

col1, col2, col3 = st.columns(3)
col1.metric("Total Transactions", f"{len(df):,}")
col2.metric("Ground-Truth Anomalies", f"{int(df['label'].sum()):,}")
col3.metric("Anomaly Rate", f"{df['label'].mean()*100:.2f}%")

st.subheader("Transaction Amount Distribution")
fig, ax = plt.subplots()
sns.histplot(data=df, x="amount", hue="label", bins=50, log_scale=(True, False), ax=ax)
ax.set_xlabel("Amount (log scale)")
st.pyplot(fig)

if st.button("Run Risk Scoring Model"):
    with st.spinner("Scoring transactions with Isolation Forest..."):
        features = df[["amount", "hour", "txn_frequency_24h", "is_new_beneficiary"]]
        model = IsolationForest(contamination=anomaly_rate, random_state=seed)
        df["risk_score"] = -model.fit(features).score_samples(features)
        df["flagged"] = model.predict(features) == -1

    st.subheader("Model Results")
    detected = int(df["flagged"].sum())
    true_positives = int(((df["flagged"]) & (df["label"] == 1)).sum())
    precision = true_positives / detected if detected else 0
    recall = true_positives / df["label"].sum() if df["label"].sum() else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Flagged as Risky", f"{detected:,}")
    c2.metric("Precision", f"{precision*100:.1f}%")
    c3.metric("Recall", f"{recall*100:.1f}%")

    st.subheader("Risk Score Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(data=df, x="risk_score", hue="label", bins=50, ax=ax2)
    st.pyplot(fig2)

    st.subheader("Top 20 Highest-Risk Transactions")
    top_risk = df.sort_values("risk_score", ascending=False).head(20)
    st.dataframe(
        top_risk[["transaction_id", "amount", "hour", "txn_frequency_24h",
                  "is_new_beneficiary", "risk_score", "flagged"]],
        use_container_width=True
    )
else:
    st.info("Click 'Run Risk Scoring Model' to score the generated transactions.")
