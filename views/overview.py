import plotly.express as px
import streamlit as st

from charts import TRANSPARENT, churn_bar
from data import churn_by


def render(df):
    st.markdown('<p class="section-label">OVERVIEW</p>', unsafe_allow_html=True)
    st.title("Customer Churn Dashboard")
    st.markdown(
        '<p class="page-subtitle">Monitor customer retention and explore churn behaviour.</p>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Customers", f"{len(df):,}")
    c2.metric("Churn Rate", f"{df['ChurnFlag'].mean() * 100:.1f}%")
    c3.metric("Monthly Charges", f"${df['MonthlyCharges'].sum():,.0f}")

    st.subheader("Customer Retention Overview")
    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            st.markdown("**Customer Churn Distribution**")
            st.caption("Compare customers who stayed with those who left.")
            counts = (
                df["Churn"]
                .value_counts()
                .reindex(["No", "Yes"], fill_value=0)
                .rename_axis("Churn")
                .reset_index(name="Customers")
            )
            fig = px.pie(
                counts,
                names="Churn",
                values="Customers",
                hole=0.65,
                color="Churn",
                color_discrete_map={"No": "#2563EB", "Yes": "#F59E0B"},
            )
            fig.update_traces(
                textinfo="label+percent",
                hovertemplate="%{label}: %{value:,} customers<extra></extra>",
            )
            fig.update_layout(
                height=340,
                legend=dict(orientation="h", y=-0.12),
                margin=dict(t=15, b=25, l=10, r=10),
                paper_bgcolor=TRANSPARENT,
                plot_bgcolor=TRANSPARENT,
            )
            st.plotly_chart(fig, use_container_width=True)

    with right:
        with st.container(border=True):
            st.markdown("**Churn Rate by Contract Type**")
            st.caption("See how churn differs across contract durations.")
            st.plotly_chart(
                churn_bar(churn_by(df, "Contract"), "Contract"),
                use_container_width=True,
            )

    st.subheader("Customer Records")
    search = st.text_input(
        "Search customer records", placeholder="Enter a customer ID..."
    )
    cols = [
        "customerID",
        "Contract",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Churn",
    ]
    records = df[[c for c in cols if c in df.columns]]
    if search.strip():
        records = records[
            records["customerID"]
            .astype(str)
            .str.contains(search.strip(), case=False, na=False)
        ]

    st.dataframe(
        records.head(100),
        use_container_width=True,
        hide_index=True,
        column_config={
            "MonthlyCharges": st.column_config.NumberColumn("Monthly", format="$%.2f"),
            "TotalCharges": st.column_config.NumberColumn("Total", format="$%.2f"),
        },
    )
    st.caption(
        f"Showing {min(len(records), 100):,} of {len(records):,} matching records."
    )
