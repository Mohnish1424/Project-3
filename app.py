import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

st.set_page_config(
    page_title="B2B Client Risk Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ DARK THEME COLORS ------------------

BG       = "#0F172A"
SURFACE  = "#111827"
BORDER   = "#1F2937"
TEXT     = "#E5E7EB"
MUTED    = "#9CA3AF"
ACCENT   = "#2563EB"
ACCENT_L = "#1E3A8A"
DANGER   = "#EF4444"
WARN     = "#F59E0B"
SUCCESS  = "#10B981"

RISK_COLORS = {"High Risk": DANGER, "Medium Risk": WARN, "Low Risk": SUCCESS}

# ------------------ GLOBAL CSS ------------------

st.markdown(f"""
<style>

.stApp {{
    background-color: {BG};
}}

.main .block-container {{
    background-color: {SURFACE};
    padding: 2rem 2.5rem 3rem 2.5rem;
    border-radius: 10px;
}}

[data-testid="stSidebar"] > div:first-child {{
    background-color: {SURFACE};
}}

h1,h2,h3,h4,h5 {{
    color:{TEXT};
}}

p,span,label {{
    color:{TEXT};
}}

hr {{
border-color:{BORDER};
}}

[data-testid="metric-container"] {{
background-color:{SURFACE};
border:1px solid {BORDER};
border-radius:10px;
}}

</style>
""", unsafe_allow_html=True)

# ------------------ MATPLOTLIB DARK STYLE ------------------

rcParams.update({
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "axes.edgecolor": BORDER,
    "axes.labelcolor": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": TEXT,
    "axes.grid": True,
    "grid.color": BORDER
})

# ------------------ DATASET ------------------

REGIONS    = ["North", "South", "East", "West", "Central"]
INDUSTRIES = ["Finance", "Healthcare", "Retail", "Manufacturing", "Tech", "Logistics"]

@st.cache_data
def load_data():

    try:
        df = pd.read_csv("B2B_Client_Churn_5000.csv")
        df.columns = df.columns.str.strip().str.replace(" ", "_")
        return df

    except:
        rng = np.random.default_rng(42)

        df = pd.DataFrame({
            "Company":[f"Client-{i}" for i in range(200)],
            "Region":rng.choice(REGIONS,200),
            "Industry":rng.choice(INDUSTRIES,200),
            "Monthly_Revenue_USD":rng.integers(2000,50000,200),
            "Monthly_Usage_Score":rng.integers(10,100,200),
            "Payment_Delay_Days":rng.integers(0,90,200),
            "Contract_Length_Months":rng.choice([3,6,12,24],200),
            "Support_Tickets_Last30Days":rng.integers(0,15,200)
        })

        return df

data = load_data()

# ------------------ HEADER ------------------

col1,col2 = st.columns([1,8])

with col1:
    st.image("logo.png", width=80)

with col2:
    st.title("B2B Client Risk & Churn Prediction Dashboard")
    st.caption("Group-2 • Rhinos • BBA Semester 4 • Woxsen University")

if st.button("👥 View Team Members"):
    st.info("""
Group-2 – Rhinos

Mohnish Singh Patwal  
Shreyas Kandi  
Akash Krishna  
Nihal Talampally
""")

st.markdown("Monitor risk, predict churn, and prioritize high-value customers")

st.divider()

# ------------------ RISK CALCULATION ------------------

def calc_risk(row):
    r=0
    if row["Payment_Delay_Days"]>30: r+=2
    if row["Monthly_Usage_Score"]<50: r+=2
    if row["Contract_Length_Months"]<12: r+=2
    if row["Support_Tickets_Last30Days"]>5: r+=2
    return r

data["Risk_Score"]=data.apply(calc_risk,axis=1)

def label(x):
    if x<=2:return "Low Risk"
    elif x<=5:return "Medium Risk"
    else:return "High Risk"

data["Risk_Category"]=data["Risk_Score"].apply(label)

# ------------------ SIDEBAR ------------------

with st.sidebar:

    st.subheader("Filters")

    region = st.multiselect("Region", data["Region"].unique())
    industry = st.multiselect("Industry", data["Industry"].unique())
    risk = st.multiselect("Risk Level", ["High Risk","Medium Risk","Low Risk"])

# ------------------ FILTER DATA ------------------

filtered=data.copy()

if region:
    filtered=filtered[filtered["Region"].isin(region)]

if industry:
    filtered=filtered[filtered["Industry"].isin(industry)]

if risk:
    filtered=filtered[filtered["Risk_Category"].isin(risk)]

# ------------------ KPIs ------------------

st.subheader("Key Metrics")

total=len(filtered)
high=(filtered["Risk_Category"]=="High Risk").sum()
churn=round((filtered.get("Renewal_Status",0)==0).mean()*100 if "Renewal_Status" in filtered else 0,1)
avg=round(filtered["Monthly_Revenue_USD"].mean(),2)

c1,c2,c3,c4=st.columns(4)

c1.metric("Total Clients",total)
c2.metric("High Risk Clients",high)
c3.metric("Churn Rate",f"{churn}%")
c4.metric("Avg Revenue",f"${avg:,.0f}")

st.divider()

# ------------------ RISK DISTRIBUTION ------------------

colA,colB=st.columns(2)

with colA:

    st.subheader("Risk Distribution")

    risk_counts=filtered["Risk_Category"].value_counts()

    fig,ax=plt.subplots(figsize=(4,2.5))

    ax.bar(risk_counts.index,risk_counts.values,
           color=[RISK_COLORS[i] for i in risk_counts.index])

    st.pyplot(fig)

with colB:

    st.subheader("Revenue vs Risk")

    fig,ax=plt.subplots(figsize=(4,2.5))

    ax.scatter(filtered["Monthly_Revenue_USD"],
               filtered["Risk_Score"],
               color=ACCENT)

    ax.set_xlabel("Revenue")
    ax.set_ylabel("Risk Score")

    st.pyplot(fig)

st.divider()

# ------------------ INDUSTRY RISK ------------------

st.subheader("Industry Risk")

fig,ax=plt.subplots(figsize=(6,2.5))

ind=filtered.groupby("Industry")["Risk_Score"].mean()

ax.bar(ind.index,ind.values,color=ACCENT)

plt.xticks(rotation=30)

st.pyplot(fig)

st.divider()

# ------------------ HIGH RISK CLIENTS ------------------

st.subheader("High Revenue Clients At Risk")

med=filtered["Monthly_Revenue_USD"].median()

risk_clients=filtered[
(filtered["Risk_Category"]=="High Risk") &
(filtered["Monthly_Revenue_USD"]>med)
]

st.dataframe(risk_clients.head(10),use_container_width=True)

st.divider()

# ------------------ FOOTER ------------------

st.markdown(
f"""
<div style="text-align:center;margin-top:40px;color:{MUTED};font-size:12px">
B2B Client Risk Dashboard | Group-2 Rhinos | BBA Semester 4 | Woxsen University
</div>
""",
unsafe_allow_html=True
)
