import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Client Risk Dashboard", layout="wide")

# ---------- DATA ----------

@st.cache_data
def load_data():

    rng = np.random.default_rng(42)

    df = pd.DataFrame({
        "Client":[f"Client-{i}" for i in range(200)],
        "Region":rng.choice(["North","South","East","West"],200),
        "Industry":rng.choice(["Finance","Healthcare","Retail","Tech"],200),
        "Revenue":rng.integers(2000,50000,200),
        "Usage":rng.integers(10,100,200),
        "Payment_Delay":rng.integers(0,90,200),
        "Contract":rng.choice([3,6,12,24],200),
        "Tickets":rng.integers(0,15,200)
    })

    def risk(row):

        score=0

        if row["Payment_Delay"]>30: score+=2
        if row["Usage"]<50: score+=2
        if row["Contract"]<12: score+=2
        if row["Tickets"]>5: score+=2

        return score

    df["Risk_Score"]=df.apply(risk,axis=1)

    def label(x):

        if x<=2: return "Low Risk"
        elif x<=5: return "Medium Risk"
        else: return "High Risk"

    df["Risk_Level"]=df["Risk_Score"].apply(label)

    df["Renewal"]=(rng.random(200)>df["Risk_Score"]/10).astype(int)

    return df


data=load_data()

# ---------- HEADER ----------

st.title("B2B Client Risk & Churn Prediction Dashboard")
st.caption("Group-2 • Rhinos • Woxsen University")

# ---------- SIDEBAR ----------

st.sidebar.header("Filters")

region=st.sidebar.multiselect("Region",data["Region"].unique())
industry=st.sidebar.multiselect("Industry",data["Industry"].unique())

filtered=data.copy()

if region:
    filtered=filtered[filtered["Region"].isin(region)]

if industry:
    filtered=filtered[filtered["Industry"].isin(industry)]

# ---------- KPIs ----------

st.subheader("Key Metrics")

c1,c2,c3,c4=st.columns(4)

total=len(filtered)
high=(filtered["Risk_Level"]=="High Risk").sum()
churn=round((filtered["Renewal"]==0).mean()*100,1)
avg=round(filtered["Revenue"].mean(),0)

c1.metric("Total Clients",total)
c2.metric("High Risk",high)
c3.metric("Churn Rate",f"{churn}%")
c4.metric("Avg Revenue",f"${avg}")

st.divider()

# ---------- CHARTS ----------

col1,col2=st.columns(2)

with col1:

    st.subheader("Risk Distribution")

    risk_counts=filtered["Risk_Level"].value_counts()

    fig,ax=plt.subplots()

    ax.bar(risk_counts.index,risk_counts.values)

    st.pyplot(fig)

with col2:

    st.subheader("Industry Risk")

    ind=filtered.groupby("Industry")["Risk_Score"].mean()

    fig,ax=plt.subplots()

    ax.bar(ind.index,ind.values)

    st.pyplot(fig)

st.divider()

col3,col4=st.columns(2)

with col3:

    st.subheader("Revenue vs Risk")

    fig,ax=plt.subplots()

    ax.scatter(filtered["Revenue"],filtered["Risk_Score"])

    st.pyplot(fig)

with col4:

    st.subheader("Contract vs Renewal")

    renew=data.groupby("Contract")["Renewal"].mean()*100

    fig,ax=plt.subplots()

    ax.plot(renew.index,renew.values)

    st.pyplot(fig)

st.divider()

# ---------- HIGH VALUE CLIENTS ----------

st.subheader("High Revenue Clients at Risk")

median=filtered["Revenue"].median()

risk_clients=filtered[
(filtered["Risk_Level"]=="High Risk") &
(filtered["Revenue"]>median)
]

st.dataframe(risk_clients.head(10))

st.divider()

# ---------- RETENTION STRATEGY ----------

st.subheader("Retention Strategy")

if st.button("Generate Recommendations"):

    st.success("""
• Offer payment flexibility for delayed clients  
• Assign account managers to high complaint customers  
• Provide incentives for long contracts  
• Improve onboarding for low usage clients
""")
