import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(
    page_title="B2B Client Risk Intelligence Dashboard",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Styling
# -----------------------------

st.markdown("""
<style>

body {
    font-family: 'Segoe UI', sans-serif;
}

.kpi-card{
    background-color:#F5F7FA;
    padding:20px;
    border-radius:10px;
    text-align:center;
    border:1px solid #E0E0E0;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv("B2B_Client_Churn_5000.csv")

# -----------------------------
# Risk Score Logic
# -----------------------------

def calculate_risk(row):

    score = 0

    if row["Payment_Delay_Days"] > 30:
        score += 3
    elif row["Payment_Delay_Days"] > 15:
        score += 2
    else:
        score += 1

    if row["Monthly_Usage_Score"] < 30:
        score += 3
    elif row["Monthly_Usage_Score"] < 60:
        score += 2
    else:
        score += 1

    if row["Contract_Length_Months"] <= 6:
        score += 3
    elif row["Contract_Length_Months"] <= 12:
        score += 2
    else:
        score += 1

    if row["Support_Tickets_Last30Days"] > 10:
        score += 3
    elif row["Support_Tickets_Last30Days"] > 5:
        score += 2
    else:
        score += 1

    return score


df["Risk_Score"] = df.apply(calculate_risk, axis=1)


def categorize_risk(score):

    if score >= 10:
        return "High Risk"
    elif score >= 7:
        return "Medium Risk"
    else:
        return "Low Risk"


df["Risk_Category"] = df["Risk_Score"].apply(categorize_risk)

# -----------------------------
# Sidebar Filters
# -----------------------------

st.sidebar.title("Filters")

region_filter = st.sidebar.multiselect(
    "Region",
    df["Region"].unique(),
    default=df["Region"].unique()
)

industry_filter = st.sidebar.multiselect(
    "Industry",
    df["Industry"].unique(),
    default=df["Industry"].unique()
)

risk_filter = st.sidebar.multiselect(
    "Risk Category",
    df["Risk_Category"].unique(),
    default=df["Risk_Category"].unique()
)

filtered_df = df[
    (df["Region"].isin(region_filter)) &
    (df["Industry"].isin(industry_filter)) &
    (df["Risk_Category"].isin(risk_filter))
]

# -----------------------------
# KPIs
# -----------------------------

total_clients = len(filtered_df)

high_risk = len(filtered_df[filtered_df["Risk_Category"] == "High Risk"])

avg_revenue = filtered_df["Monthly_Revenue_USD"].mean()

churn_rate = (filtered_df["Renewal_Status"] == "No").mean() * 100

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="kpi-card">
    <h4>Total Clients</h4>
    <h2>{total_clients}</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card">
    <h4>High Risk Clients</h4>
    <h2>{high_risk}</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card">
    <h4>Predicted Churn Rate</h4>
    <h2>{churn_rate:.2f}%</h2>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="kpi-card">
    <h4>Avg Revenue / Client</h4>
    <h2>${avg_revenue:.0f}</h2>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# -----------------------------
# Charts
# -----------------------------

col1, col2 = st.columns(2)

with col1:

    st.subheader("Risk Category Distribution")

    fig, ax = plt.subplots()

    filtered_df["Risk_Category"].value_counts().plot(
        kind="bar",
        ax=ax
    )

    st.pyplot(fig)


with col2:

    st.subheader("Industry Wise Risk")

    industry_risk = pd.crosstab(
        filtered_df["Industry"],
        filtered_df["Risk_Category"]
    )

    fig, ax = plt.subplots()

    industry_risk.plot(kind="bar", stacked=True, ax=ax)

    st.pyplot(fig)

# -----------------------------
# Scatter Plot
# -----------------------------

st.subheader("Revenue vs Risk")

fig, ax = plt.subplots()

ax.scatter(
    filtered_df["Risk_Score"],
    filtered_df["Monthly_Revenue_USD"]
)

ax.set_xlabel("Risk Score")
ax.set_ylabel("Monthly Revenue")

st.pyplot(fig)

# -----------------------------
# Machine Learning Model
# -----------------------------

st.divider()
st.header("Churn Prediction Model")

model_df = df.copy()

model_df["Renewal_Status"] = model_df["Renewal_Status"].map({"Yes":1,"No":0})

features = [
    "Monthly_Usage_Score",
    "Payment_Delay_Days",
    "Contract_Length_Months",
    "Support_Tickets_Last30Days",
    "Monthly_Revenue_USD"
]

X = model_df[features]
y = model_df["Renewal_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.3,random_state=42
)

model = DecisionTreeClassifier(max_depth=5)

model.fit(X_train,y_train)

pred = model.predict(X_test)

accuracy = accuracy_score(y_test,pred)

st.write("Model Accuracy:", round(accuracy,3))

cm = confusion_matrix(y_test,pred)

st.write("Confusion Matrix")

st.write(cm)

# Feature Importance

importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=False)

st.subheader("Feature Importance")

fig, ax = plt.subplots()

importance.plot(kind="bar", ax=ax)

st.pyplot(fig)

# -----------------------------
# Top High Risk Clients
# -----------------------------

st.divider()
st.header("Top 20 High Risk Clients")

high_risk_table = filtered_df[
    filtered_df["Risk_Category"] == "High Risk"
].sort_values(
    by="Risk_Score",
    ascending=False
).head(20)

st.dataframe(high_risk_table)

# -----------------------------
# Retention Strategy Generator
# -----------------------------

st.divider()

st.header("AI Retention Strategy")

if st.button("Generate Retention Strategy"):

    st.success("Recommended Actions")

    st.write("• Offer discount plans to clients with payment delays above 30 days")

    st.write("• Assign dedicated account managers to high revenue clients")

    st.write("• Provide contract renewal incentives for clients with short contracts")

    st.write("• Improve support response for clients with high ticket counts")

    st.write("• Conduct engagement campaigns for low usage customers")

# -----------------------------
# Responsible AI Section
# -----------------------------

st.divider()

st.header("Responsible AI Considerations")

st.write("""

Predictive churn models can introduce several ethical concerns.

Bias in Data  
If historical data reflects bias (for example certain industries receiving worse service),
the model may unfairly label them as high-risk.

Client Labeling Risk  
Labeling a client as "High Risk" may influence company behavior toward them, which could
damage long-term relationships.

Data Privacy  
Companies must ensure that sensitive client data used for prediction is handled securely
and complies with data protection regulations.

Responsible Decision Making  
AI predictions should support human decision making rather than replace it. Managers should
interpret predictions carefully before taking action.

""")
