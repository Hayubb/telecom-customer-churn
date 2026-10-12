from pathlib import Path

import pandas as pd
import streamlit as st

DATA_PATH = Path(__file__).parent / "data" / "telco_churn.csv"


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return None
    df = pd.read_csv(DATA_PATH)
    for c in ["tenure", "MonthlyCharges", "TotalCharges"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Churn"] = df["Churn"].astype(str).str.strip().str.title()
    df["ChurnFlag"] = (df["Churn"] == "Yes").astype(int)
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        [-1, 12, 24, 48, float("inf")],
        labels=["0–12 months", "13–24 months", "25–48 months", "49+ months"],
    )
    return df


@st.cache_data
def churn_by(df, col):
    return (
        df.groupby(col, observed=True)["ChurnFlag"]
        .agg(Customers="size", Churned="sum", ChurnRate="mean")
        .reset_index()
        .assign(ChurnRate=lambda d: d["ChurnRate"] * 100)
        .sort_values("ChurnRate", ascending=False)
    )
