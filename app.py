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
#@st.cache_resource
#def load_artifact():
    #with open("C:/Users/User/Downloads/churn_model.pkl", "rb") as f:
        #return pickle.load(f)

#@st.cache_data
#def load_data():
    #return pd.read_csv("C:/Users/User/Downloads/customer_churn_dataset.csv")

#artifact = load_artifact()

#model = artifact["model"]
#model_name = artifact["model_name"]
#scaler = artifact["scaler"]
#all_feature_names = artifact["all_feature_names"]
#selected_features = artifact["selected_features"]
#target_encoder = artifact["target_encoder"]
#results_table = artifact["results_table"]


# **Customer Churn Prediction — An End-to-End ML Project**

**Goal (in plain English):** A telecom company wants to know **which customers are likely to leave (churn)**, so it can try to keep them.

We will go through the **complete machine learning pipeline**, step by step, explaining every action in simple terms:

#import streamlit as st
#st.title("📡 Customer Churn Predictor")
#st.caption("A simple end-to-end machine learning app — predict whether a telecom customer will churn.")

1. Import libraries
2. Load the dataset
3. Basic information about the dataset
4. Exploratory Data Analysis (EDA) — missing values, duplicates, outliers
5. Feature Engineering — scaling, importance, selection
6. Train-test split
7. Model training (8 different algorithms)
8. Model evaluation (accuracy, confusion matrix, precision, recall, F1, ROC-AUC)
"""

"""## **Step 1: Import all the Required Libraries**

Think of this like laying out all your tools on the table before starting a DIY project.
- `pandas` / `numpy` → for handling data (like a super-powered Excel)
- `matplotlib` / `seaborn` → for drawing charts
- `sklearn` → the main machine learning toolbox
- `xgboost`, `lightgbm` → two extra powerful ML algorithms

"""

# Data handling
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# UI
# import streamlit as st

# Preprocessing & utilities
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
#from sklearn.feature_selection import SelectKBest, f_classif

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.feature_selection import SelectKBest, f_classif

# Evaluation
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score,
    roc_curve, roc_auc_score, RocCurveDisplay
)

#import warnings
#warnings.filterwarnings("ignore")

# Make charts look nicer
#sns.set_style("whitegrid")
#plt.rcParams["figure.figsize"] = (7, 4)

print("✅ All libraries imported successfully!")


"""## **Step 2: Load the Dataset**

We are using a **Customer Churn dataset** — data about telecom customers, and whether they left the company (`Churn = Yes`) or stayed (`Churn = No`).

> 📁 Make sure `customer_churn_dataset.csv` is in the same folder as this notebook.

"""

df = pd.read_csv("/customer_churn_dataset.csv")
df.head()

"""## **Step 3: Basic Information about the Dataset**

Before doing anything fancy, let's get to know our data:
- How many rows & columns?
- What are the column names and data types?
- Any statistical summary?

"""

print("Shape of dataset (rows, columns):", df.shape)

df.info()

df.describe(include="all").T

print("Target variable distribution:")
df["Churn"].value_counts()

#sns.countplot(data=df, x="Churn", palette="Set2")
#plt.title("How many customers churned vs stayed?")
#plt.show()

"""## **Step 4: Exploratory Data Analysis (EDA)**

EDA simply means **"getting familiar with messy real-world data and cleaning it up"**. Real data is almost never perfect. it usually has missing values, duplicate rows, and weird extreme values (outliers). We fix these before training a model.

### **4.1 Handling Missing Values**

Missing values are like blank cells in an Excel sheet — the information just wasn't recorded. We first **find** them, then **fill them in (impute)** using sensible values (like the average or the most common value).
"""

# Step A: Find missing values
missing = df.isnull().sum()
missing_percent = (missing / len(df)) * 100
missing_summary = pd.DataFrame({"Missing Count": missing, "Missing %": missing_percent})
missing_summary[missing_summary["Missing Count"] > 0]

# Step B: Fill missing values
# Numeric columns -> fill with the median (middle value, robust to outliers)
# Categorical columns -> fill with the mode (most frequent value)

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(include="object").columns.tolist()

for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

print("✅ Missing values remaining:", df.isnull().sum().sum())

"""## **4.2 Handling Duplicate Values**

Duplicate rows are exact copies of another row — they can happen due to data entry mistakes or system glitches. They can unfairly bias the model, so we remove them.

"""

print("Number of duplicate rows found:", df.duplicated().sum())

df = df.drop_duplicates()

print("✅ Duplicates removed. New shape:", df.shape)

"""## **4.3 Handling Outliers**

Outliers are extreme, unusual values — e.g. a customer's monthly bill of \$500 when almost everyone pays \$20-150. They can confuse the model. We use the **IQR (Interquartile Range) method**: any value far outside the "normal" range is capped (clipped) to a reasonable boundary instead of deleted, so we don't lose data.

"""

def cap_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data[column] = data[column].clip(lower=lower_bound, upper=upper_bound)
    return data

# Visualize before
#fig, axes = plt.subplots(1, 2, figsize=(12, 4))
#sns.boxplot(data=df, y="MonthlyCharges", ax=axes[0], color="salmon")
#axes[0].set_title("MonthlyCharges - Before")

#sns.boxplot(data=df, y="TotalCharges", ax=axes[1], color="salmon")
#axes[1].set_title("TotalCharges - Before")
#plt.tight_layout()
#plt.show()

# Cap outliers for key numeric columns
for col in ["MonthlyCharges", "TotalCharges", "Age"]:
    df = cap_outliers_iqr(df, col)

# Visualize after
#fig, axes = plt.subplots(1, 2, figsize=(12, 4))
#sns.boxplot(data=df, y="MonthlyCharges", ax=axes[0], color="lightgreen")
#axes[0].set_title("MonthlyCharges - After")
#sns.boxplot(data=df, y="TotalCharges", ax=axes[1], color="lightgreen")
#axes[1].set_title("TotalCharges - After")
#plt.tight_layout()
#plt.show()

print("✅ Outliers handled (capped to a reasonable range).")

"""## **Step 5: Feature Engineering**

"Feature Engineering" just means **preparing the columns (features) so the model can understand them best**.

### **5.1 Encoding categorical columns**

Machine learning models only understand numbers, not text like `"Yes"` or `"Fiber optic"`. So we convert text categories into numbers.
"""

# Drop the ID column - it's just a label, not useful for prediction
df_model = df.drop(columns=["CustomerID"])

# Encode the target column: Yes -> 1, No -> 0
le_target = LabelEncoder()
df_model["Churn"] = le_target.fit_transform(df_model["Churn"])  # Yes=1, No=0

# One-hot encode the remaining categorical columns
categorical_features = df_model.select_dtypes(include="object").columns.tolist()
df_model = pd.get_dummies(df_model, columns=categorical_features, drop_first=True)

"""### **5.2 Feature Scaling**

Some columns (like `TotalCharges`, in the thousands) are on a much bigger scale than others (like `Age`, in tens). Many models get confused by this size difference, so we **scale everything to a similar range** using `StandardScaler`.

"""

X = df_model.drop(columns=["Churn"])
y = df_model["Churn"]

feature_names = X.columns.tolist()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_names)

X_scaled.head()

"""## **5.3 Feature Importance**

Not all features matter equally for predicting churn. We use a quick **Random Forest** model to rank which features matter most — this is just for understanding the data, not our final model yet.

"""

rf_importance = RandomForestClassifier(n_estimators=200, random_state=42)
rf_importance.fit(X_scaled, y)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf_importance.feature_importances_
}).sort_values(by="Importance", ascending=False)

#plt.figure(figsize=(8, 6))
#sns.barplot(data=importance_df, x="Importance", y="Feature", palette="viridis")
##plt.title("Which features matter most for predicting churn?")
#plt.show()

importance_df

"""### **5.4 Feature Selection**

To keep our model simple and fast, we select only the **top most useful features** (using a statistical test called ANOVA F-test) instead of using all of them.

"""

K = 10  # number of top features to keep
selector = SelectKBest(score_func=f_classif, k=min(K, X_scaled.shape[1]))
X_selected = selector.fit_transform(X_scaled, y)

selected_features = X_scaled.columns[selector.get_support()].tolist()
X_final = pd.DataFrame(X_selected, columns=selected_features)

print("✅ Selected top features:")
print(selected_features)

"""## **Step 6: Train-Test Split**

We split our data into two parts:
- **Training set (80%)** → the model learns patterns from this
- **Test set (20%)** → used to check how well the model performs on data it has never seen (like a final exam)

"""

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42, stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

"""## **Step 7: Model Training**

Now the fun part! We train **8 different classification algorithms** and compare them. Think of this like asking 8 different experts to each predict churn, so we can see which expert is the most accurate.

"""

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
    "LightGBM": LGBMClassifier(random_state=42, verbose=-1),
}

trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model
    print(f"✅ {name} trained successfully!")

"""## **Step 8: Model Evaluation**

For each model, we check:
- **Accuracy** → % of correct predictions overall
- **Precision** → of all customers we *predicted* would churn, how many actually did?
- **Recall** → of all customers who *actually* churned, how many did we catch?
- **F1 Score** → a balance between precision and recall
- **Confusion Matrix** → a table showing correct vs incorrect predictions
- **ROC Curve & AUC** → how well the model separates churners from non-churners

"""

results = []

for name, model in trained_models.items():
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    results.append({
        "Model": name,
        "Accuracy": round(acc, 3),
        "Precision": round(prec, 3),
        "Recall": round(rec, 3),
        "F1 Score": round(f1, 3),
        "ROC-AUC": round(auc, 3),
    })

results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
results_df

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
        ##plt.xticks(rotation=15)
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
