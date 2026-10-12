import streamlit as st


def render():
    st.markdown(
        '<p class="section-label">PREDICTIVE ANALYTICS</p>', unsafe_allow_html=True
    )
    st.title("Churn Risk Assessment")
    st.markdown(
        '<p class="page-subtitle">Enter a customer profile to prepare a churn risk assessment.</p>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown("**Assess a customer profile**")
        st.caption(
            "Provide the customer's service and account details. "
            "The trained machine learning model will be connected here later."
        )

    with st.form("customer_prediction_form"):
        st.subheader("Customer information")
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior citizen", ["No", "Yes"])
            partner = st.selectbox("Has a partner?", ["Yes", "No"])
            dependents = st.selectbox("Has dependents?", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)

        with col2:
            internet_service = st.selectbox(
                "Internet service", ["DSL", "Fiber optic", "No"]
            )
            contract = st.selectbox(
                "Contract type", ["Month-to-month", "One year", "Two year"]
            )
            payment_method = st.selectbox(
                "Payment method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
            monthly_charges = st.number_input(
                "Monthly charges ($)",
                min_value=0.0,
                max_value=200.0,
                value=70.0,
                step=5.0,
            )
            paperless_billing = st.selectbox("Paperless billing", ["Yes", "No"])

        submitted = st.form_submit_button(
            "Assess churn risk", use_container_width=True, type="primary"
        )

    if submitted:
        st.subheader("Assessment status")
        st.info(
            "Your customer profile has been captured. "
            "The trained prediction model has not been connected yet, "
            "so no churn probability is being reported."
        )
        st.caption(
            "This form is a UI prototype. A real assessment will be available "
            "after integration with the team's saved preprocessing pipeline "
            "and trained model."
        )
