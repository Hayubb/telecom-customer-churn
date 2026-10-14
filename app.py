import base64
import joblib
from pathlib import Path

import streamlit as st
from PIL import Image

from data import load_data
from views import insights, overview, prediction

BASE_DIR = Path(__file__).parent
ASSETS = BASE_DIR / "assets"  # put favicon.png and logo.png in this folder


MODEL_PATH = Path(__file__).parent / "models" / "logistic_regression_bundle.joblib"


@st.cache_resource
def load_model_bundle():
    return joblib.load(MODEL_PATH)


bundle = load_model_bundle()

model = bundle["model"]
preprocessor = bundle["preprocessor"]
feature_columns = bundle["feature_columns"]
model_name = bundle["model_name"]
target_classes = bundle["target_classes"]
feature_engineering = bundle["feature_engineering"]


# 1. Browser tab icon (must be the first Streamlit call)
st.set_page_config(
    page_title="Telecom Churn Intelligence",
    page_icon=Image.open(ASSETS / "favicon.png"),
    layout="wide",
)


css = (BASE_DIR / "styles.css").read_text()
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

df = load_data()
if df is None:
    st.error("Dataset not found. Place telco_churn.csv inside the data folder.")
    st.stop()

pg = st.navigation(
    [
        st.Page(
            lambda: overview.render(df),
            title="Overview",
            icon="📊",
            default=True,
            url_path="overview",
        ),
        st.Page(
            prediction.render,
            title="Churn Prediction",
            icon="🎯",
            url_path="prediction",
        ),
        st.Page(
            lambda: insights.render(df),
            title="Customer Insights",
            icon="🔍",
            url_path="insights",
        ),
    ]
)


@st.cache_data
def logo_data_uri():
    return base64.b64encode((ASSETS / "logo.png").read_bytes()).decode()


with st.container(key="app_header"):
    st.markdown(
        f"""
        <div class="app-header">
            <div class="brand-wrap">
                <img src="data:image/png;base64,{logo_data_uri()}" class="brand-logo">
                <div>
                    <div class="brand">Churn Intelligence</div>
                    <div class="tagline">Telecom customer retention analytics</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

pg.run()
