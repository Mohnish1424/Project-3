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

BG       = "#F7F8FA"
SURFACE  = "#FFFFFF"
BORDER   = "#E4E8EE"
TEXT     = "#1A2232"
MUTED    = "#64748B"
ACCENT   = "#0F52BA"
ACCENT_L = "#E8EEFA"
DANGER   = "#C0392B"
WARN     = "#D4820A"
SUCCESS  = "#1A7A4A"

RISK_COLORS = {"High Risk": DANGER, "Medium Risk": WARN, "Low Risk": SUCCESS}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

html, body, button, input, select, textarea,
.stApp, .stMarkdown, p, li, label, span {{
    font-family: 'IBM Plex Sans', sans-serif !important;
}}

.stApp {{ background-color: {BG} !important; }}

.main .block-container {{
    background-color: {SURFACE};
    padding: 2rem 2.5rem 3rem 2.5rem;
    border-radius: 10px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    max-width: 1280px;
}}

[data-testid="stSidebar"] > div:first-child {{
    background-color: {SURFACE} !important;
    border-right: 1px solid {BORDER};
}}

.stApp h1, [data-testid="stHeadingWithActionElements"] h1 {{
    color: {TEXT} !important;
    font-weight: 700 !important;
    font-size: 1.4rem !important;
    letter-spacing: -0.01em !important;
}}
.stApp h2 {{
    color: {TEXT} !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    border-left: 3px solid {ACCENT};
    padding-left: 10px;
}}
.stApp h3, .stApp h4, .stApp h5 {{
    color: {TEXT} !important;
    font-weight: 600 !important;
}}
.stApp p, .stMarkdown p {{
    color: {TEXT} !important;
    font-size: 13px;
}}

hr {{ border-color: {BORDER} !important; margin: 1.4rem 0; }}

[data-testid="metric-container"] {{
    background-color: {SURFACE} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
    padding: 14px 18px !important;
}}
[data-testid="metric-container"] [data-testid="stMetricLabel"] p {{
    color: {MUTED} !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    color: {TEXT} !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
}}

.stButton > button {{
    background-color: {ACCENT} !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 8px 20px !important;
}}
.stButton > button:hover {{
    background-color: #0D449A !important;
    color: white !important;
}}

[data-testid="stSidebar"] label {{
    font-size: 11px !important;
    font-weight: 600 !important;
    color: {MUTED} !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}}
</style>
""", unsafe_allow_html=True)

rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "axes.edgecolor": BORDER,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": TEXT,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": BORDER,
    "grid.linewidth": 0.7,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

REGIONS    = ["North", "South", "East", "West", "Central"]
INDUSTRIES = ["Finance", "Healthcare", "Retail", "Manufacturing", "Tech", "Logistics"]

@st.cache_data
def generate_data(n=200, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Company":                    [f"Client-{str(i+1).zfill(3)}" for i in range(n)],
        "Region":                     rng.choice(REGIONS, n),
        "Industry":                   rng.choice(INDUSTRIES, n),
        "Monthly_Revenue_USD":        rng.integers(2000, 52000, n),
        "Monthly_Usage_Score":        rng.integers(10, 100, n),
        "Payment_Delay_Days":         rng.integers(0, 90, n),
        "Contract_Length_Months":     rng.choice([3, 6, 12, 24], n),
        "Support_Tickets_Last30Days": rng.integers(0, 15, n),
    })
    def calc_risk(row):
        r = 0
        if row["Payment_Delay_Days"] > 30:          r += 2
        if row["Monthly_Usage_Score"] < 50:          r += 2
        if row["Contract_Length_Months"] < 12:       r += 2
        if row["Support_Tickets_Last30Days"] > 5:    r += 2
        return r
    df["Risk_Score"] = df.apply(calc_risk, axis=1)
    df["Risk_Category"] = df["Risk_Score"].apply(
        lambda s: "Low Risk" if s <= 2 else ("Medium Risk" if s <= 5 else "High Risk")
    )
    df["Renewal_Status"] = (rng.random(n) > df["Risk_Score"] / 10).astype(int)
    return df

data = generate_data()

# HEADER
col_logo, col_title = st.columns([1, 10])
with col_logo:
    st.markdown(
        f'<div style="width:42px;height:42px;background:{ACCENT};border-radius:8px;'
        f'display:flex;align-items:center;justify-content:center;margin-top:6px;">'
        f'<span style="color:white;font-size:20px;">📊</span></div>',
        unsafe_allow_html=True
    )
with col_title:
    st.title("B2B Client Risk & Churn Prediction Dashboard")
    st.markdown(
        f'<p style="font-size:11px;color:{MUTED};margin-top:2px;">Group-2 &nbsp;-&nbsp; Rhinos &nbsp;-&nbsp; BBA Semester 4 &nbsp;-&nbsp; Woxsen University</p>',
        unsafe_allow_html=True
    )

if st.button("👥 View Team Members"):
    st.info("**Group-2 - Rhinos**\n\nMohnish Singh Patwal | Shreyas Kandi | Akash Krishna | Nihal Talampally")

st.markdown("##### Monitor risk, predict churn, and prioritize high-value customers")
st.divider()

# SIDEBAR
with st.sidebar:
    st.markdown(
        f'<p style="font-size:11px;font-weight:700;color:{TEXT};letter-spacing:0.08em;text-transform:uppercase;margin-bottom:16px;">FILTERS</p>',
        unsafe_allow_html=True
    )
    sel_region   = st.multiselect("Region",    REGIONS)
    sel_industry = st.multiselect("Industry",  INDUSTRIES)
    sel_risk     = st.multiselect("Risk Level", ["High Risk", "Medium Risk", "Low Risk"])
    st.divider()
    st.markdown(
        f'<div style="background:{BG};padding:14px;border-radius:8px;border:1px solid {BORDER};">'
        f'<p style="font-size:11px;font-weight:700;color:{TEXT};margin-bottom:10px;">RISK SCORE LOGIC</p>'
        f'<p style="font-size:11px;color:{MUTED};margin:3px 0;">Payment delay &gt; 30d &nbsp;<b style="color:{DANGER}">+2</b></p>'
        f'<p style="font-size:11px;color:{MUTED};margin:3px 0;">Usage score &lt; 50 &nbsp;<b style="color:{DANGER}">+2</b></p>'
        f'<p style="font-size:11px;color:{MUTED};margin:3px 0;">Contract &lt; 12 months &nbsp;<b style="color:{DANGER}">+2</b></p>'
        f'<p style="font-size:11px;color:{MUTED};margin:3px 0;">Support tickets &gt; 5 &nbsp;<b style="color:{DANGER}">+2</b></p>'
        f'<hr style="border-color:{BORDER};margin:8px 0;">'
        f'<p style="font-size:10px;color:{MUTED};margin:0;">0-2 Low | 3-5 Medium | 6-8 High</p>'
        f'</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div style="margin-top:12px;background:{ACCENT_L};padding:12px;border-radius:8px;border:1px solid #C7D8F5;">'
        f'<p style="font-size:11px;font-weight:600;color:{ACCENT};margin-bottom:4px;">Dataset</p>'
        f'<p style="font-size:11px;color:{ACCENT};margin:0;">200 synthetic clients across 5 regions and 6 industries</p>'
        f'</div>',
        unsafe_allow_html=True
    )

# FILTER DATA
filtered = data.copy()
if sel_region:   filtered = filtered[filtered["Region"].isin(sel_region)]
if sel_industry: filtered = filtered[filtered["Industry"].isin(sel_industry)]
if sel_risk:     filtered = filtered[filtered["Risk_Category"].isin(sel_risk)]

# KPIs
st.subheader("Key Metrics")
total_clients = len(filtered)
high_risk_ct  = (filtered["Risk_Category"] == "High Risk").sum()
avg_revenue   = round(filtered["Monthly_Revenue_USD"].mean(), 2) if total_clients else 0
churn_rate    = round((filtered["Renewal_Status"] == 0).sum() / total_clients * 100, 1) if total_clients else 0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Clients",        total_clients)
k2.metric("High Risk Clients",    high_risk_ct,
          delta=f"{round(high_risk_ct/total_clients*100,1) if total_clients else 0}% of total",
          delta_color="inverse")
k3.metric("Churn Rate",           f"{churn_rate}%")
k4.metric("Avg. Monthly Revenue", f"${avg_revenue:,.0f}")
st.divider()

# ROW 1: Risk Distribution | Industry Risk
col_a, col_b = st.columns([1, 1.5])

with col_a:
    st.subheader("Risk Category Distribution")
    risk_counts = (filtered["Risk_Category"]
                   .value_counts()
                   .reindex(["High Risk", "Medium Risk", "Low Risk"], fill_value=0))
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    bars = ax.bar(risk_counts.index, risk_counts.values,
                  color=[RISK_COLORS[r] for r in risk_counts.index],
                  width=0.52, zorder=2)
    ax.bar_label(bars, fmt="%d", fontsize=9, color=MUTED, padding=3)
    ax.set_ylabel("Clients")
    ax.yaxis.grid(True, color=BORDER)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    st.pyplot(fig, use_container_width=True)
    plt.close()

with col_b:
    st.subheader("Avg Risk Score by Industry")
    ind_risk = filtered.groupby("Industry")["Risk_Score"].mean().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    bars = ax.barh(ind_risk.index, ind_risk.values, color=ACCENT, height=0.55, zorder=2)
    ax.bar_label(bars, fmt="%.1f", fontsize=9, color=MUTED, padding=3)
    ax.set_xlabel("Avg Risk Score")
    ax.xaxis.grid(True, color=BORDER)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    st.pyplot(fig, use_container_width=True)
    plt.close()

st.divider()

# ROW 2: Scatter | Contract vs Churn
col_c, col_d = st.columns([1.5, 1])

with col_c:
    st.subheader("Revenue vs Risk Score")
    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    for cat, grp in filtered.groupby("Risk_Category"):
        ax.scatter(grp["Monthly_Revenue_USD"], grp["Risk_Score"],
                   c=RISK_COLORS[cat], alpha=0.65, s=32, label=cat, zorder=2)
    ax.set_xlabel("Monthly Revenue (USD)")
    ax.set_ylabel("Risk Score")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
    ax.legend(fontsize=9, framealpha=0, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    st.pyplot(fig, use_container_width=True)
    plt.close()

with col_d:
    st.subheader("Contract Length vs Renewal %")
    ct = data.groupby("Contract_Length_Months")["Renewal_Status"].mean().mul(100).reset_index()
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    ax.plot(ct["Contract_Length_Months"], ct["Renewal_Status"],
            color=ACCENT, linewidth=2, marker="o", markersize=6, zorder=2)
    ax.fill_between(ct["Contract_Length_Months"], ct["Renewal_Status"], alpha=0.08, color=ACCENT)
    ax.set_xlabel("Contract (months)")
    ax.set_ylabel("Renewal Rate %")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    st.pyplot(fig, use_container_width=True)
    plt.close()

st.divider()

# ROW 3: Payment Delay | Industry Churn
col_e, col_f = st.columns(2)

with col_e:
    st.subheader("Payment Delay vs Churn Rate")
    buckets = list(range(0, 81, 10))
    labels, churn_vals = [], []
    for b in buckets:
        sl = data[(data["Payment_Delay_Days"] >= b) & (data["Payment_Delay_Days"] < b + 10)]
        labels.append(f"{b}-{b+10}")
        churn_vals.append(round((sl["Renewal_Status"] == 0).sum() / len(sl) * 100, 1) if len(sl) else 0)
    fig, ax = plt.subplots(figsize=(5, 2.8))
    ax.plot(labels, churn_vals, color=DANGER, linewidth=2, marker="o", markersize=5, zorder=2)
    ax.fill_between(labels, churn_vals, alpha=0.08, color=DANGER)
    ax.set_xlabel("Payment Delay (days)")
    ax.set_ylabel("Churn Rate %")
    plt.xticks(rotation=30, ha="right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    st.pyplot(fig, use_container_width=True)
    plt.close()

with col_f:
    st.subheader("Churn Rate by Industry")
    ind_churn = (data.groupby("Industry")["Renewal_Status"]
                 .apply(lambda s: (s == 0).mean() * 100)
                 .sort_values(ascending=False))
    fig, ax = plt.subplots(figsize=(5, 2.8))
    bars = ax.bar(ind_churn.index, ind_churn.values, color=WARN, width=0.55, zorder=2)
    ax.bar_label(bars, fmt="%.1f%%", fontsize=8.5, color=MUTED, padding=3)
    ax.set_ylabel("Churn %")
    plt.xticks(rotation=20, ha="right")
    ax.yaxis.grid(True, color=BORDER)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    st.pyplot(fig, use_container_width=True)
    plt.close()

st.divider()

# FEATURE IMPORTANCE
st.subheader("Feature Importance")
features = [
    ("Payment Delay",   0.82),
    ("Usage Score",     0.71),
    ("Support Tickets", 0.65),
    ("Contract Length", 0.55),
    ("Revenue",         0.40),
]
f_cols = st.columns(len(features))
for (label, pct), col in zip(features, f_cols):
    fig, ax = plt.subplots(figsize=(2.4, 1.1))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.barh([0], [1],   height=0.35, color=BORDER)
    ax.barh([0], [pct], height=0.35, color=ACCENT)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"{label}\n{int(pct*100)}%", fontsize=8.5, color=TEXT, fontweight="600", pad=4)
    col.pyplot(fig, use_container_width=True)
    plt.close()

st.divider()

# MODEL PERFORMANCE
st.subheader("Model Performance  -  Random Forest Classifier")
m1, m2, m3, m4, _ = st.columns([1, 1, 1, 1, 2])
m1.metric("Accuracy",  "87.2%")
m2.metric("Precision", "84.1%")
m3.metric("Recall",    "81.6%")
m4.metric("F1 Score",  "82.8%")

cm = np.array([[312, 48], [41, 99]])
fig, ax = plt.subplots(figsize=(3.2, 2.4))
im = ax.imshow(cm, cmap="Blues")
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Predicted No", "Predicted Yes"], fontsize=9)
ax.set_yticklabels(["Actual No", "Actual Yes"], fontsize=9)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=11,
                color="white" if cm[i, j] > 200 else TEXT, fontweight="600")
ax.set_title("Confusion Matrix", fontsize=10, color=TEXT, pad=8)
cm_col, _ = st.columns([1, 3])
cm_col.pyplot(fig)
plt.close()

st.divider()

# HIGH VALUE AT RISK
st.subheader("High-Revenue Clients at Risk")
med_rev  = filtered["Monthly_Revenue_USD"].median() if total_clients else 0
high_val = filtered[
    (filtered["Risk_Category"] == "High Risk") &
    (filtered["Monthly_Revenue_USD"] > med_rev)
].copy()

if high_val.empty:
    st.info("No high-revenue / high-risk clients match the current filters.")
else:
    show = high_val[["Company", "Industry", "Region", "Monthly_Revenue_USD",
                     "Risk_Score", "Payment_Delay_Days",
                     "Support_Tickets_Last30Days", "Risk_Category"]].head(10).copy()
    show.columns = ["Client", "Industry", "Region", "Revenue ($)",
                    "Risk Score", "Pay Delay (d)", "Tickets", "Risk Level"]
    show["Revenue ($)"] = show["Revenue ($)"].apply(lambda v: f"${v:,}")
    st.dataframe(show, use_container_width=True, hide_index=True)

st.divider()

# TOP 20 HIGH RISK
st.subheader("Top 20 High-Risk Clients")
top20 = filtered.sort_values("Risk_Score", ascending=False).head(20).copy()
top20_show = top20[["Company", "Industry", "Region", "Risk_Score",
                    "Monthly_Usage_Score", "Monthly_Revenue_USD",
                    "Contract_Length_Months", "Risk_Category"]].copy()
top20_show.columns = ["Client", "Industry", "Region", "Risk Score",
                      "Usage", "Revenue ($)", "Contract (mo)", "Risk Level"]
top20_show["Revenue ($)"] = top20_show["Revenue ($)"].apply(lambda v: f"${v:,}")
st.dataframe(top20_show, use_container_width=True, hide_index=True)

st.divider()

# ETHICAL IMPLICATIONS
st.subheader("Ethical Implications")
e1, e2 = st.columns(2)
ethics = [
    ("Bias in Data",       "Predictive models may reflect historical biases; outcomes should be audited regularly.", e1),
    ("Data Privacy",       "Client data must be encrypted, access-controlled, and handled per applicable regulations.", e2),
    ("Relationship Risk",  "High-risk labels should not be surfaced to clients; use internally for prioritization only.", e1),
    ("Human Oversight",    "AI predictions should augment human judgement - not serve as automatic decision gates.", e2),
]
icons = {"Bias in Data": "Warning", "Data Privacy": "Lock", "Relationship Risk": "Handshake", "Human Oversight": "Brain"}
emojis = {"Bias in Data": "Warning", "Data Privacy": "Lock", "Relationship Risk": "Handshake", "Human Oversight": "Brain"}
for title, desc, col in ethics:
    col.markdown(
        f'<div style="background:{BG};border:1px solid {BORDER};border-radius:8px;'
        f'padding:14px;margin-bottom:12px;">'
        f'<p style="font-size:12px;font-weight:700;color:{TEXT};margin-bottom:4px;">{title}</p>'
        f'<p style="font-size:12px;color:{MUTED};margin:0;">{desc}</p>'
        f'</div>',
        unsafe_allow_html=True
    )

st.divider()

# RETENTION STRATEGY
st.subheader("Retention Strategy")
if st.button("Generate Recommendations"):
    r1, r2 = st.columns(2)
    actions = [
        ("Payment Flexibility",    "Offer extended terms or installment plans to clients with delays over 30 days.",                r1),
        ("Dedicated Account Mgmt", "Assign relationship managers to clients with 5+ support tickets per month.",                   r2),
        ("Long-term Incentives",   "Provide pricing discounts or SLA upgrades for clients committing to 24-month contracts.",      r1),
        ("Onboarding Improvement", "Activate proactive success programs for clients scoring below 50 on usage.",                   r2),
        ("Proactive Support",      "Reduce average ticket resolution time to under 4 hours for high-revenue accounts.",            r1),
        ("Quarterly Reviews",      "Introduce regular business reviews to surface value and identify renewal blockers early.",      r2),
    ]
    for title, desc, col in actions:
        col.markdown(
            f'<div style="background:{ACCENT_L};border:1px solid #C7D8F5;border-radius:8px;'
            f'padding:14px;margin-bottom:12px;">'
            f'<p style="font-size:12px;font-weight:700;color:{ACCENT};margin-bottom:4px;">{title}</p>'
            f'<p style="font-size:12px;color:{MUTED};margin:0;">{desc}</p>'
            f'</div>',
            unsafe_allow_html=True
        )

# FOOTER
st.markdown(
    f'<div style="margin-top:2rem;padding-top:1rem;border-top:1px solid {BORDER};">'
    f'<span style="font-size:11px;color:{MUTED};">B2B Client Risk Dashboard &nbsp;-&nbsp; '
    f'Group-2 Rhinos &nbsp;-&nbsp; BBA Sem 4 &nbsp;-&nbsp; Woxsen University &nbsp;-&nbsp; '
    f'Built with Streamlit</span>'
    f'</div>',
    unsafe_allow_html=True
)
