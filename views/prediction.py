from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


@st.cache_resource
def load_model_bundle():
    """Find and load the saved model bundle."""
    for parent in Path(__file__).resolve().parents:
        bundle_path = parent / "models" / "logistic_regression_bundle.joblib"
        if bundle_path.exists():
            return joblib.load(bundle_path)

    raise FileNotFoundError("Could not find models/logistic_regression_bundle.joblib")


def render():
    st.markdown(
        '<p class="section-label">PREDICTIVE ANALYTICS</p>',
        unsafe_allow_html=True,
    )
    st.title("Churn Risk Assessment")
    st.markdown(
        '<p class="page-subtitle">'
        "Enter a customer profile to estimate churn risk."
        "</p>",
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown("**Assess a customer profile**")
        st.caption(
            "Enter the customer's account, billing, and service details "
            "to generate a prediction from the trained Logistic Regression model."
        )

    with st.form("customer_prediction_form"):
        st.subheader("Customer information")
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior citizen", ["No", "Yes"])
            partner = st.selectbox("Has a partner?", ["Yes", "No"])
            dependents = st.selectbox("Has dependents?", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)

            phone_service = st.selectbox(
                "Phone service",
                ["Yes", "No"],
                help="Does this customer subscribe to telephone service?",
            )

            if phone_service == "No":
                multiple_lines = "No phone service"
                st.selectbox(
                    "Multiple lines",
                    ["No phone service"],
                    disabled=True,
                    help="Not applicable because the customer has no phone service.",
                )
            else:
                multiple_lines = st.selectbox(
                    "Multiple lines",
                    ["No", "Yes"],
                    help="Does the customer have more than one phone line?",
                )

        with col2:
            internet_service = st.selectbox(
                "Internet service",
                ["DSL", "Fiber optic", "No"],
            )
            contract = st.selectbox(
                "Contract type",
                ["Month-to-month", "One year", "Two year"],
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
            total_charges = st.number_input(
                "Total charges ($)",
                min_value=0.0,
                value=840.0,
                step=10.0,
            )

        average_monthly_spend = (
            total_charges / tenure if tenure > 0 else monthly_charges
        )

        st.metric(
            "Average monthly spend",
            f"${average_monthly_spend:,.2f}",
            help=(
                "Total charges divided by tenure in months. "
                "When tenure is zero, monthly charges are used."
            ),
        )

        paperless_billing = st.selectbox("Paperless billing", ["Yes", "No"])

        st.subheader("Additional services")

        service_col1, service_col2, service_col3 = st.columns(3)

        # These options match the original Telco churn dataset.
        with service_col1:
            online_security = st.selectbox(
                "Online security",
                ["No", "Yes", "No internet service"],
            )
            online_backup = st.selectbox(
                "Online backup",
                ["No", "Yes", "No internet service"],
            )
            device_protection = st.selectbox(
                "Device protection",
                ["No", "Yes", "No internet service"],
            )

        with service_col2:
            tech_support = st.selectbox(
                "Tech support",
                ["No", "Yes", "No internet service"],
            )
            streaming_tv = st.selectbox(
                "Streaming TV",
                ["No", "Yes", "No internet service"],
            )

        with service_col3:
            streaming_movies = st.selectbox(
                "Streaming movies",
                ["No", "Yes", "No internet service"],
            )

        submitted = st.form_submit_button(
            "Assess churn risk",
            use_container_width=True,
            type="primary",
        )

    if submitted:
        try:
            bundle = load_model_bundle()
            model = bundle["model"]
            preprocessor = bundle["preprocessor"]

            # Convert the senior-citizen selection to the training format.
            senior_citizen_value = 1 if senior_citizen == "Yes" else 0

            # Supply all 20 original model input features.
            customer = {
                "gender": gender,
                "SeniorCitizen": senior_citizen_value,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless_billing,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
                "AverageMonthlySpend": average_monthly_spend,
            }

            # Respect the exact feature order saved with the model.
            input_df = pd.DataFrame(
                [customer],
                columns=bundle["feature_columns"],
            )

            # Apply the original fitted preprocessing pipeline.
            processed_input = preprocessor.transform(input_df)

            prediction = model.predict(processed_input)[0]
            probabilities = model.predict_proba(processed_input)[0]

            classes = list(model.classes_)
            churn_index = classes.index("Yes")
            churn_probability = float(probabilities[churn_index])

            st.divider()
            st.subheader("Assessment result")

            result_col1, result_col2, result_col3 = st.columns(3)

            with result_col1:
                st.metric(
                    "Predicted outcome",
                    "Likely to churn" if prediction == "Yes" else "Likely to stay",
                )

            with result_col2:
                st.metric(
                    "Churn probability",
                    f"{churn_probability:.1%}",
                )

            with result_col3:
                st.metric(
                    "Average monthly spend",
                    f"${average_monthly_spend:,.2f}",
                )

            st.progress(churn_probability)

            if prediction == "Yes":
                st.warning(
                    "The model predicts that this customer will churn. "
                    "Consider reviewing their contract, service experience, "
                    "and available retention options."
                )
            else:
                st.success("The model predicts that this customer will stay.")

            st.caption(
                "This is a model-based estimate, not a certainty. "
                "Predictions depend on the information entered and the "
                "patterns learned from the training dataset."
            )

        except FileNotFoundError as error:
            st.error(str(error))
        except Exception as error:
            st.error(
                "The prediction could not be completed. " f"Error details: {error}"
            )
