import streamlit as st

from charts import churn_bar
from data import churn_by

SEGMENTS = {
    "Contract type": "Contract",
    "Internet service": "InternetService",
    "Payment method": "PaymentMethod",
    "Tenure group": "TenureGroup",
}


def render(df):
    st.markdown(
        '<p class="section-label">EXPLORATORY ANALYTICS</p>', unsafe_allow_html=True
    )
    st.title("Customer Insights")
    st.markdown(
        '<p class="page-subtitle">Explore customer segments and the factors associated with churn.</p>',
        unsafe_allow_html=True,
    )

    st.subheader("Explore churn by customer segment")
    label = st.selectbox("Choose a customer attribute", list(SEGMENTS))
    col = SEGMENTS[label]
    seg = churn_by(df, col)

    m1, m2, m3 = st.columns(3)
    m1.metric("Segments compared", len(seg))
    m2.metric("Highest segment churn", f"{seg.iloc[0]['ChurnRate']:.1f}%")
    m3.metric("Customers in dataset", f"{len(df):,}")

    fig = churn_bar(
        seg, col, height=400, labels={col: label}, hover_data=["Customers", "Churned"]
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Segment breakdown")
    st.dataframe(
        seg.rename(columns={col: label}),
        use_container_width=True,
        hide_index=True,
        column_config={
            "ChurnRate": st.column_config.NumberColumn("Churn Rate (%)", format="%.1f")
        },
    )
