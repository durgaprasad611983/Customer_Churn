"""
Streamlit App: Customer Churn Predictor
-----------------------------------------
A very simple, beginner-friendly web app that loads the model trained
in `Customer_Churn_Classification.ipynb` and lets a user predict whether
a customer is likely to churn.

Run with:
    streamlit run app.py
"""

import pickle
import numpy as np
import pandas as pd
import streamlit as st
#import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# -----------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📡",
    layout="wide",
)

# -----------------------------------------------------------------
# Load model artifact (cached so it only loads once)
# -----------------------------------------------------------------
@st.cache_resource
def load_artifact():
    with open("C:/Users/User/Downloads/churn_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    return pd.read_csv("C:/Users/User/Downloads/customer_churn_dataset.csv")

artifact = load_artifact()

model = artifact["model"]
model_name = artifact["model_name"]
scaler = artifact["scaler"]
all_feature_names = artifact["all_feature_names"]
selected_features = artifact["selected_features"]
target_encoder = artifact["target_encoder"]
results_table = artifact["results_table"]

# -----------------------------------------------------------------
# Title & intro
# -----------------------------------------------------------------
st.title("📡 Customer Churn Predictor")
st.caption("A simple end-to-end machine learning app — predict whether a telecom customer will churn.")

st.info(
    f"🏆 Currently serving predictions using **{model_name}**, the best model "
    f"found during training (out of 8 models compared)."
)

tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Model Performance", "🗂️ Explore Dataset"])

# ===================================================================
# TAB 1: PREDICTION
# ===================================================================
with tab1:
    st.subheader("Enter customer details")
    st.write("Fill in the details below and click **Predict** to see if this customer is likely to churn.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.slider("Age", 18, 70, 35)
            tenure = st.slider("Tenure (months with company)", 0, 72, 12)
            monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 200.0, 70.0, step=1.0)

        with col2:
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 800.0, step=10.0)
            num_support_calls = st.slider("Number of Support Calls", 0, 15, 2)
            contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

        with col3:
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            tech_support = st.radio("Has Tech Support?", ["Yes", "No"], horizontal=True)
            payment_method = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
            )

        submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True)

    if submitted:
        with st.spinner("Analyzing customer profile..."):
            # Build a raw input row matching the ORIGINAL training columns
            raw_input = pd.DataFrame([{
                "Age": age,
                "Tenure_Months": tenure,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
                "NumSupportCalls": num_support_calls,
                "ContractType": contract_type,
                "InternetService": internet_service,
                "TechSupport": tech_support,
                "PaymentMethod": payment_method,
            }])

            # One-hot encode the same way as training
            raw_input = pd.get_dummies(
                raw_input,
                columns=["ContractType", "InternetService", "TechSupport", "PaymentMethod"],
            )

            # Add any missing dummy columns (set to 0) and order columns correctly
            for col in all_feature_names:
                if col not in raw_input.columns:
                    raw_input[col] = 0
            raw_input = raw_input[all_feature_names]

            # Scale, then select the same features used at training time
            scaled = scaler.transform(raw_input)
            scaled_df = pd.DataFrame(scaled, columns=all_feature_names)
            final_input = scaled_df[selected_features]

            prediction = model.predict(final_input)[0]
            probability = model.predict_proba(final_input)[0][1]

        st.divider()
        result_label = target_encoder.inverse_transform([prediction])[0]

        if result_label == "Yes":
            st.error(f"⚠️ This customer is **likely to churn** (probability: {probability:.1%})")
        else:
            st.success(f"✅ This customer is **likely to stay** (churn probability: {probability:.1%})")

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Churn Probability", f"{probability:.1%}")
        col_b.metric("Prediction", result_label)
        col_c.metric("Model Used", model_name)

        st.progress(min(max(probability, 0.0), 1.0))
        st.caption("The bar above shows the churn risk level — closer to full means higher risk.")

# ===================================================================
# TAB 2: MODEL PERFORMANCE
# ===================================================================
with tab2:
    st.subheader("How well do our models perform?")
    st.write(
        "We trained **8 different classification algorithms** and compared them. "
        "Here's how they stack up on unseen test data:"
    )

    st.dataframe(
        results_table.style.highlight_max(
            subset=["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"], color="lightgreen"
        ),
        use_container_width=True,
    )

    best_row = results_table.iloc[0]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{best_row['Accuracy']:.1%}")
    m2.metric("Precision", f"{best_row['Precision']:.1%}")
    m3.metric("Recall", f"{best_row['Recall']:.1%}")
    m4.metric("F1 Score", f"{best_row['F1 Score']:.1%}")
    m5.metric("ROC-AUC", f"{best_row['ROC-AUC']:.1%}")

    #st.divider()
    #st.subheader("Accuracy Comparison Across Models")
    #fig, ax = plt.subplots(figsize=(8, 4))
    #sns.barplot(data=results_table, x="Accuracy", y="Model", palette="mako", ax=ax)
    #ax.set_xlim(0, 1)
    #st.pyplot(fig)

    with st.expander("ℹ️ What do these metrics mean?"):
        st.markdown("""
- **Accuracy** — Overall, what % of predictions were correct?
- **Precision** — Of the customers we *predicted* would churn, how many actually did?
- **Recall** — Of the customers who *actually* churned, how many did we correctly catch?
- **F1 Score** — A balance between Precision and Recall.
- **ROC-AUC** — How well the model separates churners from non-churners (closer to 1.0 is better).
        """)

# ===================================================================
# TAB 3: EXPLORE DATASET
# ===================================================================
with tab3:
    st.subheader("Take a peek at the training dataset")
    df = load_data()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", len(df))
    c2.metric("Churned", int((df["Churn"] == "Yes").sum()))
    c3.metric("Stayed", int((df["Churn"] == "No").sum()))
    c4.metric("Features", df.shape[1] - 2)

    st.dataframe(df.head(20), use_container_width=True)

    #st.divider()
    #col_x, col_y = st.columns(1)

    #with col_x:
        #st.write("**Churn by Contract Type**")
        #fig1, ax1 = plt.subplots()
        #sns.countplot(data=df, x="ContractType", hue="Churn", palette="Set2", ax=ax1)
        #plt.xticks(rotation=15)
        #st.pyplot(fig1)

    #with col_y:
        #st.write("**Monthly Charges Distribution**")
        #fig2, ax2 = plt.subplots()
        #sns.histplot(data=df, x="MonthlyCharges", hue="Churn", kde=True, palette="Set2", ax=ax2)
        #st.pyplot(fig2)

    st.success("Tip: Head to the **Predict** tab to try the model on your own custom customer profile!")

# -----------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ About this app")
    st.write(
        "This is a simple end-to-end machine learning demo: a telecom company's "
        "customer data is used to predict which customers are likely to churn."
    )
    st.markdown("**Pipeline steps covered:**")
    st.markdown("""
    - Data loading & cleaning
    - Missing value & outlier handling
    - Feature engineering & selection
    - Training 8 ML models
    - Evaluating & picking the best one
    - Deploying with Streamlit
    """)
    st.divider()
    st.caption("Built with ❤️ using Streamlit + scikit-learn")
