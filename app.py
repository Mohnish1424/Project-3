import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="Client Risk Dashboard", layout="wide")

# ===== Background Styling =====
st.markdown("""
<style>
.stApp {
    background-color: #0E1117;
}
.block-container {
    background-color: #111827;
    padding: 2rem;
    border-radius: 10px;
}
[data-testid="stSidebar"] {
    background-color: #111827;
}
</style>
""", unsafe_allow_html=True)

st.title("B2B Client Risk & Churn Prediction Dashboard")

# ===== Fancy Header =====
st.markdown("""
<style>
.big-title {
    font-size:28px;
    font-weight:700;
    background: linear-gradient(90deg, #00C9FF, #92FE9D);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
</style>
<div class="big-title">AI-Powered Client Risk Intelligence System</div>
""", unsafe_allow_html=True)

st.markdown("### 📊 Monitor risk, predict churn, and prioritize high-value customers")

# ===== LOAD DATA =====
data = pd.read_csv("B2B_Client_Churn_5000.csv")
data.columns = data.columns.str.strip().str.replace(" ", "_")
data['Renewal_Status'] = data['Renewal_Status'].map({'Yes':1,'No':0})

# ===== RISK SCORE =====
def calculate_risk(row):
    risk = 0
    if row['Payment_Delay_Days'] > 30:
        risk += 2
    if row['Monthly_Usage_Score'] < 50:
        risk += 2
    if row['Contract_Length_Months'] < 12:
        risk += 2
    if row['Support_Tickets_Last30Days'] > 5:
        risk += 2
    return risk

data['Risk_Score'] = data.apply(calculate_risk, axis=1)

def risk_label(score):
    if score <= 2:
        return "Low Risk"
    elif score <= 5:
        return "Medium Risk"
    else:
        return "High Risk"

data['Risk_Category'] = data['Risk_Score'].apply(risk_label)

# ===== SIDEBAR FILTERS =====
st.sidebar.header("Filters")

region = st.sidebar.multiselect("Region", data['Region'].unique())
industry = st.sidebar.multiselect("Industry", data['Industry'].unique())
risk = st.sidebar.multiselect("Risk Category", data['Risk_Category'].unique())

filtered = data.copy()

if region:
    filtered = filtered[filtered['Region'].isin(region)]
if industry:
    filtered = filtered[filtered['Industry'].isin(industry)]
if risk:
    filtered = filtered[filtered['Risk_Category'].isin(risk)]

# ===== KPI METRICS =====
st.markdown("---")
st.subheader("Key Metrics")
st.caption("⚠ High Risk clients require immediate retention attention.")

total_clients = len(filtered)
high_risk = (filtered['Risk_Category']=="High Risk").sum()
churn_rate = round(filtered['Renewal_Status'].mean()*100,2)
avg_revenue = round(filtered['Monthly_Revenue_USD'].mean(),2)

col1, col2, col3, col4 = st.columns(4)
col1.metric("👥 Total Clients", total_clients)
col2.metric("⚠ High Risk", high_risk)
col3.metric("📉 Churn Rate %", churn_rate)
col4.metric("💰 Avg Revenue", avg_revenue)

# ===== CHURN GAUGE =====
st.subheader("Overall Churn Risk")
st.progress(churn_rate/100)
st.write(f"Churn Risk Level: {churn_rate}%")

# ===== RISK DISTRIBUTION (smaller) =====
st.markdown("---")
st.subheader("Risk Distribution")

risk_counts = filtered['Risk_Category'].value_counts()
fig, ax = plt.subplots(figsize=(5,3))
ax.bar(risk_counts.index, risk_counts.values)
st.pyplot(fig)

# ===== SIDE-BY-SIDE ANALYTICS =====
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Industry-wise Risk")
    st.bar_chart(filtered.groupby('Industry')['Risk_Score'].mean(), height=250)

with col2:
    st.subheader("Revenue vs Risk")
    st.scatter_chart(filtered, x='Monthly_Revenue_USD', y='Risk_Score', height=250)

# ===== MACHINE LEARNING MODEL =====
st.markdown("---")
st.subheader("Churn Prediction Model")

features = [
    'Monthly_Usage_Score',
    'Payment_Delay_Days',
    'Contract_Length_Months',
    'Support_Tickets_Last30Days',
    'Monthly_Revenue_USD'
]

X = data[features]
y = data['Renewal_Status']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)

st.write("Model Accuracy:", round(accuracy_score(y_test, pred),3))

fig, ax = plt.subplots(figsize=(4,3))
ax.matshow(confusion_matrix(y_test, pred))
st.pyplot(fig)

# ===== FEATURE IMPORTANCE =====
importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
})

st.subheader("Feature Importance")
st.bar_chart(importance.set_index('Feature'), height=250)

# ===== EXTRA ANALYTICS =====
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Churn Rate by Industry")
    st.bar_chart(data.groupby('Industry')['Renewal_Status'].mean(), height=250)

with col2:
    st.subheader("Payment Delay vs Churn")
    st.line_chart(data.groupby('Payment_Delay_Days')['Renewal_Status'].mean(), height=250)

st.subheader("High Revenue Clients at Risk")
high_value_risk = filtered[
    (filtered['Risk_Category']=="High Risk") &
    (filtered['Monthly_Revenue_USD'] > filtered['Monthly_Revenue_USD'].median())
]
st.dataframe(high_value_risk.head(10))

# ===== TOP HIGH RISK CLIENTS =====
st.markdown("---")
st.subheader("Top 20 High Risk Clients")

top20 = filtered.sort_values(by='Risk_Score', ascending=False).head(20)

def highlight_risk(row):
    if row['Risk_Category'] == "High Risk":
        return ['background-color:#ff4b4b;color:white'] * len(row)
    elif row['Risk_Category'] == "Medium Risk":
        return ['background-color:#ffa600'] * len(row)
    else:
        return ['background-color:#2ecc71'] * len(row)

st.dataframe(top20.style.apply(highlight_risk, axis=1))

# ===== RETENTION STRATEGY =====
st.markdown("---")
if st.button("Generate Retention Strategy"):
    st.write("### Recommended Actions")
    st.write("• Offer discounts to clients with payment delays > 30 days")
    st.write("• Assign dedicated account managers to high complaint clients")
    st.write("• Provide incentives for long-term contracts")
    st.write("• Improve response time for support tickets")
    st.write("• Provide onboarding/training for low usage clients")

# ===== ETHICAL AI =====
st.markdown("---")
st.subheader("Ethical Implications of Predicting Client Churn")

st.write("""
• Predictive models may contain bias from historical data.  
• Labeling clients as high-risk can affect relationships unfairly.  
• Client data must be protected and handled securely.  
• AI predictions should support human decisions, not replace them.
""")