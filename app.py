import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="Client Risk Dashboard", layout="wide")

# ===== THEME =====
st.markdown("""
<style>
.stApp {background-color: #0E1117;}
.block-container {
    background-color: #111827;
    padding: 2rem;
    border-radius: 12px;
}
[data-testid="stSidebar"] {background-color: #111827;}
</style>
""", unsafe_allow_html=True)

# ===== HEADER WITH LOGO =====
col1, col2 = st.columns([1.2,6])

with col1:
    st.image("logo.png", width=110)

with col2:
    st.title("B2B Client Risk & Churn Prediction Dashboard")
    st.caption("Group-2 • Rhinos • BBA-Sem:4")

# ===== TEAM POPUP =====
if st.button("👥 View Team Members"):
    st.info("""
**Group-2 — Rhinos**

• Mohnish Singh Patwal  
• Shreyas Kandi  
• Akash Krishna  
• Nihal Talampally  

Woxsen University  
BBA Semester 4
""")

st.caption("AI-powered risk intelligence & churn prediction")

# ===== LOAD DATA =====
data = pd.read_csv("B2B_Client_Churn_5000.csv")
data.columns = data.columns.str.strip().str.replace(" ", "_")
data['Renewal_Status'] = data['Renewal_Status'].map({'Yes':1,'No':0})

# ===== SIDEBAR =====
st.sidebar.subheader("Risk Settings")
usage_threshold = st.sidebar.slider("Low Usage Threshold", 20, 80, 50)
ticket_threshold = st.sidebar.slider("Support Ticket Risk Level", 1, 10, 5)

st.sidebar.subheader("Filters")
region = st.sidebar.multiselect("Region", data['Region'].unique())
industry = st.sidebar.multiselect("Industry", data['Industry'].unique())

st.sidebar.markdown("---")
st.sidebar.subheader("👥 Team — Group 2")
st.sidebar.write("""
**Rhinos**

• Mohnish Singh Patwal  
• Shreyas Kandi  
• Akash Krishna  
• Nihal Talampally  
""")

# ===== PRESENTATION MODE =====
st.sidebar.markdown("---")
presentation_mode = st.sidebar.checkbox("📽 Presentation Mode")
st.sidebar.info("Turn ON for clean screenshots")

if presentation_mode:
    st.markdown(
        "<style>[data-testid='stSidebar'] {display:none;}</style>",
        unsafe_allow_html=True
    )

# ===== RISK SCORE =====
def calculate_risk(row):
    risk = 0
    if row['Payment_Delay_Days'] > 30:
        risk += 2
    if row['Monthly_Usage_Score'] < usage_threshold:
        risk += 2
    if row['Contract_Length_Months'] < 12:
        risk += 2
    if row['Support_Tickets_Last30Days'] > ticket_threshold:
        risk += 2
    return risk

data['Risk_Score'] = data.apply(calculate_risk, axis=1)

def risk_label(score):
    if score <= 2: return "Low Risk"
    elif score <= 5: return "Medium Risk"
    else: return "High Risk"

data['Risk_Category'] = data['Risk_Score'].apply(risk_label)

filtered = data.copy()
if region:
    filtered = filtered[filtered['Region'].isin(region)]
if industry:
    filtered = filtered[filtered['Industry'].isin(industry)]

# ===== KPI METRICS =====
st.markdown("---")
if not presentation_mode:
    st.subheader("Key Metrics")

total_clients = len(filtered)
high_risk = (filtered['Risk_Category']=="High Risk").sum()
churn_rate = round(filtered['Renewal_Status'].mean()*100,2)
avg_revenue = round(filtered['Monthly_Revenue_USD'].mean(),2)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Clients", total_clients)
c2.metric("High Risk", high_risk)
c3.metric("Churn %", churn_rate)
c4.metric("Avg Revenue", avg_revenue)

st.progress(churn_rate/100)

# ===== RISK DISTRIBUTION =====
st.markdown("---")
if not presentation_mode:
    st.subheader("Risk Distribution")

chart_type = st.radio("Chart Type", ["Bar", "Pie"], horizontal=True)
risk_counts = filtered['Risk_Category'].value_counts()

if chart_type == "Bar":
    st.bar_chart(risk_counts, height=220)
else:
    fig, ax = plt.subplots(figsize=(3,3), facecolor='none')
    ax.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%')
    st.pyplot(fig)

# ===== ANALYTICS =====
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    if not presentation_mode:
        st.subheader("Industry Risk")
    st.bar_chart(filtered.groupby('Industry')['Risk_Score'].mean(), height=220)

with col2:
    if not presentation_mode:
        st.subheader("Revenue vs Risk")
    show_density = st.checkbox("Show Density", value=False)
    if show_density:
        st.scatter_chart(filtered.sample(1000), x='Monthly_Revenue_USD', y='Risk_Score', height=260)
    else:
        st.scatter_chart(filtered, x='Monthly_Revenue_USD', y='Risk_Score', height=260)

# ===== MACHINE LEARNING =====
st.markdown("---")
if not presentation_mode:
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

st.write("Accuracy:", round(accuracy_score(y_test, pred),3))

# ===== CONFUSION MATRIX (FIXED) =====
fig, ax = plt.subplots(figsize=(3.5,3))
cm = confusion_matrix(y_test, pred)
cax = ax.imshow(cm, cmap="Blues")

for i in range(len(cm)):
    for j in range(len(cm[0])):
        ax.text(j, i, cm[i, j], ha='center', va='center')

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Model Performance")

fig.colorbar(cax)
st.pyplot(fig)

# ===== FEATURE IMPORTANCE =====
importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
})

if not presentation_mode:
    st.subheader("Feature Importance")

st.bar_chart(importance.set_index('Feature'), height=220)

# ===== HIGH VALUE CLIENTS =====
st.markdown("---")
if not presentation_mode:
    st.subheader("High Revenue Clients at Risk")

high_value = filtered[
    (filtered['Risk_Category']=="High Risk") &
    (filtered['Monthly_Revenue_USD'] > filtered['Monthly_Revenue_USD'].median())
]

st.dataframe(high_value.head(10))

# ===== TOP RISK CLIENTS =====
st.markdown("---")
if not presentation_mode:
    st.subheader("Top High Risk Clients")

sort_order = st.selectbox("Sort By", ["Risk Score", "Revenue"])

if sort_order == "Revenue":
    top20 = filtered.sort_values(by='Monthly_Revenue_USD', ascending=False).head(20)
else:
    top20 = filtered.sort_values(by='Risk_Score', ascending=False).head(20)

st.dataframe(top20)

# ===== RETENTION STRATEGIES =====
st.markdown("---")
if st.button("Generate Retention Strategy"):
    st.success("""
• Offer discounts for delayed payments  
• Assign account managers to high complaint clients  
• Encourage long-term contracts  
• Provide training for low usage customers  
""")

# ===== ETHICS =====
st.markdown("---")
if not presentation_mode:
    st.subheader("Responsible AI Considerations")

st.write("""
• Avoid bias in model predictions  
• Protect customer data privacy  
• Use predictions to support decisions, not replace them  
""")

# ===== FOOTER =====
st.markdown("---")
st.caption("Developed by Group-2 • Rhinos • BBA Semester 4")
