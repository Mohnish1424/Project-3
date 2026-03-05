import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="Client Risk Dashboard", layout="wide")

# ===== PROFESSIONAL UI STYLE =====
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #F4F6F8;
}

.block-container {
    background-color: #FFFFFF;
    padding: 2rem 3rem;
    border-radius: 10px;
    box-shadow: 0px 1px 6px rgba(0,0,0,0.08);
}

[data-testid="stSidebar"] {
    background-color: #ECEFF3;
}

h1, h2, h3 {
    color: #2C3E50;
    font-weight: 600;
}

p, li {
    color: #444;
    font-size: 14px;
}

[data-testid="metric-container"] {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    padding: 12px;
    border-radius: 8px;
}

.stButton button {
    background-color: #3B6EA8;
    color: white;
    border-radius: 6px;
    border: none;
}

.stButton button:hover {
    background-color: #2E5C8F;
}

</style>
""", unsafe_allow_html=True)

# ===== HEADER =====
col1, col2 = st.columns([1,8])

with col1:
    st.image("logo.png", width=70)

with col2:
    st.title("B2B Client Risk & Churn Prediction Dashboard")
    st.caption("Group-2 • Rhinos • BBA Semester 4 • Woxsen University")

# ===== TEAM BUTTON =====
if st.button("👥 View Team Members"):
    st.info("""
Group-2 — Rhinos

Mohnish Singh Patwal  
Shreyas Kandi  
Akash Krishna  
Nihal Talampally  
""")

st.markdown("### Monitor risk, predict churn, and prioritize high-value customers")

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

# ===== MACHINE LEARNING MODEL =====
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

model_accuracy = round(accuracy_score(y_test, pred),3)
predicted_churn_rate = round(pred.mean()*100,2)

# ===== KPI METRICS =====
st.divider()
st.subheader("Key Metrics")

total_clients = len(filtered)
high_risk = (filtered['Risk_Category']=="High Risk").sum()
avg_revenue = round(filtered['Monthly_Revenue_USD'].mean(),2)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Clients", total_clients)
col2.metric("High Risk Clients", high_risk)
col3.metric("Predicted Churn %", predicted_churn_rate)
col4.metric("Average Revenue", avg_revenue)

# ===== RISK LOGIC CARD =====
st.markdown("""
<div style="
background-color:#EEF2F7;
padding:14px;
border-radius:8px;
border:1px solid #D8E0EA;
">

<b>Risk Score Logic</b>

• Payment delay > 30 days → +2 risk  
• Usage score < 50 → +2 risk  
• Contract length < 12 months → +2 risk  
• Support tickets > 5 → +2 risk  

</div>
""", unsafe_allow_html=True)

# ===== RISK DISTRIBUTION =====
st.divider()
st.subheader("Risk Category Distribution")

risk_counts = filtered['Risk_Category'].value_counts()

fig, ax = plt.subplots(figsize=(4,2.3))

ax.bar(
    risk_counts.index,
    risk_counts.values,
    color="#4C78A8"
)

ax.spines[['top','right']].set_visible(False)
ax.set_ylabel("Clients")

st.pyplot(fig)

# ===== ANALYTICS =====
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Industry-wise Risk")
    st.bar_chart(filtered.groupby('Industry')['Risk_Score'].mean(), height=250)

with col2:
    st.subheader("Revenue vs Risk")
    st.scatter_chart(filtered, x='Monthly_Revenue_USD', y='Risk_Score', height=250)

# ===== CONTRACT LENGTH VS CHURN =====
st.divider()
st.subheader("Contract Length vs Churn")

st.line_chart(
    data.groupby('Contract_Length_Months')['Renewal_Status'].mean(),
    height=250
)

# ===== MODEL PERFORMANCE =====
st.divider()
st.subheader("Churn Prediction Model")

st.write("Model Accuracy:", model_accuracy)

cm = confusion_matrix(y_test, pred)

fig, ax = plt.subplots(figsize=(3,2.4))
ax.imshow(cm, cmap="Blues")

for i in range(len(cm)):
    for j in range(len(cm)):
        ax.text(j, i, cm[i, j], ha="center", va="center")

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

st.pyplot(fig)

# ===== FEATURE IMPORTANCE =====
importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
})

st.subheader("Feature Importance")
st.bar_chart(importance.set_index('Feature'), height=250)

# ===== EXTRA ANALYTICS =====
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Churn Rate by Industry")
    st.bar_chart(data.groupby('Industry')['Renewal_Status'].mean(), height=250)

with col2:
    st.subheader("Payment Delay vs Churn")
    st.line_chart(data.groupby('Payment_Delay_Days')['Renewal_Status'].mean(), height=250)

# ===== HIGH VALUE CLIENTS =====
st.divider()
st.subheader("High Revenue Clients at Risk")

high_value_risk = filtered[
    (filtered['Risk_Category']=="High Risk") &
    (filtered['Monthly_Revenue_USD'] > filtered['Monthly_Revenue_USD'].median())
]

st.dataframe(high_value_risk.head(10))

# ===== TOP HIGH RISK CLIENTS =====
st.divider()
st.subheader("Top 20 High Risk Clients")

top20 = filtered.sort_values(by='Risk_Score', ascending=False).head(20)

st.dataframe(top20)

# ===== RETENTION STRATEGY =====
st.divider()

if st.button("Generate Retention Strategy"):
    st.write("### Recommended Actions")

    st.write("• Offer discounts to clients with payment delays > 30 days")
    st.write("• Assign dedicated account managers to high complaint clients")
    st.write("• Provide incentives for long-term contracts")
    st.write("• Improve support response time")
    st.write("• Provide onboarding for low usage clients")

# ===== ETHICAL AI =====
st.divider()

st.subheader("Ethical Implications of Predicting Client Churn")

st.write("""
• Predictive models may contain bias from historical data.  
• Labeling clients as high-risk can affect relationships unfairly.  
• Client data must be protected and handled securely.  
• AI predictions should support human decisions, not replace them.
""")
