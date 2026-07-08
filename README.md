# Transaction Risk Scoring Dashboard

A transaction risk scoring dashboard that simulates bank transaction monitoring, using an
Isolation Forest model to detect anomalous transactions based on amount, timing, frequency,
and beneficiary patterns. Built with Streamlit and scikit-learn.

## Features

- Generates synthetic transaction data with configurable volume and anomaly rate
- Scores each transaction for anomaly risk using Isolation Forest
- Displays precision/recall against injected ground-truth anomalies
- Visualizes transaction amount distribution and risk score distribution
- Surfaces the top 20 highest-risk transactions in a sortable table

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Live demo

[Add your Streamlit Cloud link here once deployed]

## Tech stack

Python, Streamlit, scikit-learn, pandas, numpy, matplotlib, seaborn
