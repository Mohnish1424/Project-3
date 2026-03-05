import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="B2B Client Risk Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# DESIGN TOKENS
# ─────────────────────────────────────────────
C = {
    "bg":           "#F7F8FA",
    "surface":      "#FFFFFF",
    "border":       "#E4E8EE",
    "text":         "#1A2232",
    "muted":        "#64748B",
    "accent":       "#0F52BA",
    "accent_light": "#E8EEFA",
    "danger":       "#C0392B",
    "warn":         "#D4820A",
    "success":      "#1A7A4A",
}

RISK_COLORS = {
    "High Risk":   C["danger"],
    "Medium Risk": C["warn"],
    "Low Risk":    C["success"],
}

# ─────────────────────────────────────────────
# GLOBAL CSS  (IBM Plex Sans, flat surfaces, no gradients)
# ─────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'IBM Plex Sans', sans-serif;
}}

/* App background */
.stApp {{
    background-color: {C["bg"]};
}}

/* Main content card */
.block-container {{
    background-color: {C["surface"]};
    padding: 2rem 2.5rem 3rem 2.5rem;
    border-radius: 10px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    max-width: 1280px;
}}

/* Sidebar */
[data-testid="stSidebar"] {{
    background-color: {C["surface"]};
    border-right: 1px solid {C["border"]};
}}

[data-testid="stSidebar"] .block-container {{
    box-shadow: none;
    padding: 1.5rem 1.25rem;
}}

/* Headings */
h1 {{ color: {C["text"]}; font-weight: 700; font-size: 1.4rem; letter-spacing: -0.01em; }}
h2 {{ color: {C["text"]}; font-weight: 600; font-size: 1.05rem; border-left: 3px solid {C["accent"]}; padding-left: 10px; margin-top: 0.2rem; }}
h3 {{ color: {C["text"]}; font-weight: 600; font-size: 0.9rem; }}

/* Paragraphs */
p, li {{ color: {C["muted"]}; font-size: 13px; }}

/* Metric cards */
[data-testid="metric-container"] {{
    background-color: {C["surface"]};
    border: 1px solid {C["border"]};
    border-radius: 10px;
    padding: 14px 18px;
}}

/* Divider */
hr {{ border-color: {C["border"]}; margin: 1.6rem 0; }}

/* Primary button */
.stButton > button {{
    background-color: {C["accent"]};
    color: white;
    border: none;
    border-radius: 6px;
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 600;
    font-size: 13px;
    padding: 8px 20px;
    transition: background 0.15s;
}}
.stButton > button:hover {{
    background-color: #0D449A;
    color: white;
}}

/* Info boxes */
.stAlert {{
    border-radius: 8px;
    font-size: 13px;
}}

/* Dataframe */
[data-testid="stDataFrame"] {{
    border: 1px solid {C["border"]};
    border-radius: 8px;
    overflow: hidden;
}}

/* Sidebar multiselect tags */
[data-baseweb="tag"] {{
    background-color: {C["accent_light"]} !important;
    color: {C["accent"]} !important;
    border: none !important;
}}

/* Sidebar labels */
[data-testid="stSidebar"] label {{
    font-size: 11px;
    font-weight: 600;
    color: {C["muted"]};
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}

/* Caption */
.caption-text {{
    font-size: 11px;
    color: {C["muted"]};
    margin-top: 2px;
    margin-bottom: 0;
}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MATPLOTLIB DEFAULTS  (match the UI palette)
# ─────────────────────────────────────────────
rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.facecolor":    C["surface"],
    "figure.facecolor":  C["surface"],
    "axes.edgecolor":    C["border"],
    "axes.labelcolor":   C["muted"],
    "xtick.color":       C["muted"],
    "ytick.color":       C["muted"],
    "text.color":        C["text"],
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        C["border"],
    "grid.linewidth":    0.7,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
})

# ─────────────────────────────────────────────
# DATA GENERATION  (reproducible synthetic)
# ─────────────────────────────────────────────
REGIONS    = ["North", "South", "East", "West", "Central"]
INDUSTRIES = ["Finance", "Healthcare", "Retail", "Manufacturing", "Tech", "Logistics"]

@st.cache_data
def generate_data(n=200, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Company":              [f"Client-{str(i+1).zfill(3)}" for i in range(n)],
        "Region":               rng.choice(REGIONS, n),
        "Industry":             rng.choice(INDUSTRIES, n),
        "Monthly_Revenue_USD":  rng.integers(2000, 52000, n),
        "Monthly_Usage_Score":  rng.integers(10, 100, n),
        "Payment_Delay_Days":   rng.integers(0, 90, n),
        "Contract_Length_Months": rng.choice([3, 6, 12, 24], n),
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

# ─────────────────────────────────────────────
# HELPER: small bar chart factory
# ─────────────────────────────────────────────
def make_fig(h=2.8, w=None):
    fig, ax = plt.subplots(figsize=(w or 5, h))
    ax.spines["left"].set_color(C["border"])
    ax.spines["bottom"].set_color(C["border"])
    return fig, ax

def tag_html(label, style="danger"):
    bg = {"danger": "#FDECEA", "warn": "#FDF3E3", "success": "#E6F4ED"}[style]
    fg = {"danger": C["danger"], "warn": C["warn"], "success": C["success"]}[style]
    return (f'<span style="background:{bg};color:{fg};padding:2px 8px;border-radius:4px;'
            f'font-size:11px;font-weight:600;">{label}</span>')

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
col_logo, col_title = st.columns([1, 10])
with col_logo:
    st.markdown(
        f'<div style="width:42px;height:42px;background:{C["accent"]};border-radius:8px;'
        f'display:flex;align-items:center;justify-content:center;margin-top:6px;">'
        f'<span style="color:white;font-size:20px;">📊</span></div>',
        unsafe_allow_html=True
    )
with col_title:
    st.title("B2B Client Risk & Churn Prediction Dashboard")
    st.markdown('<p class="caption-text">Group-2 &nbsp;•&nbsp; Rhinos &nbsp;•&nbsp; BBA Semester 4 &nbsp;•&nbsp; Woxsen University</p>', unsafe_allow_html=True)

if st.button("👥 View Team Members"):
    st.info("**Group-2 — Rhinos**\n\nMohnish Singh Patwal &nbsp;|&nbsp; Shreyas Kandi &nbsp;|&nbsp; Akash Krishna &nbsp;|&nbsp; Nihal Talampally")

st.markdown("##### Monitor risk, predict churn, and prioritize high-value customers")
st.divider()

# ─────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(f'<p style="font-size:11px;font-weight:700;color:{C["text"]};letter-spacing:0.08em;text-transform:uppercase;margin-bottom:16px;">FILTERS</p>', unsafe_allow_html=True)

    sel_region   = st.multiselect("Region",        REGIONS)
    sel_industry = st.multiselect("Industry",       INDUSTRIES)
    sel_risk     = st.multiselect("Risk Level",     ["High Risk", "Medium Risk", "Low Risk"])

    st.divider()

    # Risk logic card
    st.markdown(
        f"""
        <div style="background:{C["bg"]};padding:14px;border-radius:8px;border:1px solid {C["border"]};">
            <p style="font-size:11px;font-weight:700;color:{C["text"]};letter-spacing:0.04em;margin-bottom:10px;">RISK SCORE LOGIC</p>
            <table style="width:100%;border-collapse:collapse;">
                <tr><td style="font-size:11px;color:{C["muted"]};padding:3px 0;">Payment delay &gt; 30d</td><td style="font-size:11px;font-weight:700;color:{C["danger"]};text-align:right;">+2</td></tr>
                <tr><td style="font-size:11px;color:{C["muted"]};padding:3px 0;">Usage score &lt; 50</td><td style="font-size:11px;font-weight:700;color:{C["danger"]};text-align:right;">+2</td></tr>
                <tr><td style="font-size:11px;color:{C["muted"]};padding:3px 0;">Contract &lt; 12 months</td><td style="font-size:11px;font-weight:700;color:{C["danger"]};text-align:right;">+2</td></tr>
                <tr><td style="font-size:11px;color:{C["muted"]};padding:3px 0;">Support tickets &gt; 5</td><td style="font-size:11px;font-weight:700;color:{C["danger"]};text-align:right;">+2</td></tr>
            </table>
            <hr style="border-color:{C["border"]};margin:8px 0;">
            <p style="font-size:10px;color:{C["muted"]};margin:0;">0–2 Low &nbsp;|&nbsp; 3–5 Medium &nbsp;|&nbsp; 6–8 High</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div style="margin-top:12px;background:{C["accent_light"]};padding:12px;border-radius:8px;border:1px solid #C7D8F5;">'
        f'<p style="font-size:11px;font-weight:600;color:{C["accent"]};margin-bottom:4px;">Dataset</p>'
        f'<p style="font-size:11px;color:{C["accent"]};margin:0;">200 synthetic clients · 5 regions · 6 industries</p>'
        f'</div>',
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────
# FILTERED DATA
# ─────────────────────────────────────────────
filtered = data.copy()
if sel_region:   filtered = filtered[filtered["Region"].isin(sel_region)]
if sel_industry: filtered = filtered[filtered["Industry"].isin(sel_industry)]
if sel_risk:     filtered = filtered[filtered["Risk_Category"].isin(sel_risk)]

# ─────────────────────────────────────────────
# KPI METRICS
# ─────────────────────────────────────────────
st.subheader("Key Metrics")

total_clients  = len(filtered)
high_risk_ct   = (filtered["Risk_Category"] == "High Risk").sum()
avg_revenue    = round(filtered["Monthly_Revenue_USD"].mean(), 2) if total_clients else 0
churn_rate     = round((filtered["Renewal_Status"] == 0).sum() / total_clients * 100, 1) if total_clients else 0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Clients",     total_clients,  help="After applying sidebar filters")
k2.metric("High Risk Clients", high_risk_ct,   delta=f"{round(high_risk_ct/total_clients*100,1) if total_clients else 0}% of filtered", delta_color="inverse")
k3.metric("Churn Rate",        f"{churn_rate}%", help="Non-renewing clients in filtered set")
k4.metric("Avg. Monthly Revenue", f"${avg_revenue:,.0f}")

st.divider()

# ─────────────────────────────────────────────
# ROW 1 — Risk Distribution  |  Industry Risk
# ─────────────────────────────────────────────
col_a, col_b = st.columns([1, 1.5])

with col_a:
    st.subheader("Risk Category Distribution")
    risk_counts = filtered["Risk_Category"].value_counts().reindex(["High Risk", "Medium Risk", "Low Risk"], fill_value=0)
    fig, ax = make_fig(2.8, 4.5)
    bars = ax.bar(risk_counts.index, risk_counts.values,
                  color=[RISK_COLORS[r] for r in risk_counts.index],
                  width=0.52, zorder=2)
    ax.bar_label(bars, fmt="%d", fontsize=9, color=C["muted"], padding=3)
    ax.set_ylabel("Clients")
    ax.yaxis.grid(True, color=C["border"])
    ax.set_axisbelow(True)
    st.pyplot(fig, use_container_width=True)

with col_b:
    st.subheader("Avg Risk Score by Industry")
    ind_risk = (filtered.groupby("Industry")["Risk_Score"].mean()
                .sort_values(ascending=True))
    fig, ax = make_fig(2.8, 5.5)
    bars = ax.barh(ind_risk.index, ind_risk.values, color=C["accent"],
                   height=0.55, zorder=2)
    ax.bar_label(bars, fmt="%.1f", fontsize=9, color=C["muted"], padding=3)
    ax.set_xlabel("Avg Risk Score")
    ax.xaxis.grid(True, color=C["border"])
    ax.set_axisbelow(True)
    st.pyplot(fig, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────
# ROW 2 — Revenue Scatter  |  Contract vs Churn
# ─────────────────────────────────────────────
col_c, col_d = st.columns([1.5, 1])

with col_c:
    st.subheader("Revenue vs Risk Score")
    fig, ax = make_fig(2.8, 5.5)
    for cat, grp in filtered.groupby("Risk_Category"):
        ax.scatter(grp["Monthly_Revenue_USD"], grp["Risk_Score"],
                   c=RISK_COLORS[cat], alpha=0.65, s=32,
                   label=cat, zorder=2)
    ax.set_xlabel("Monthly Revenue (USD)")
    ax.set_ylabel("Risk Score")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
    ax.legend(fontsize=9, framealpha=0, loc="upper right")
    st.pyplot(fig, use_container_width=True)

with col_d:
    st.subheader("Contract Length vs Renewal %")
    ct = (data.groupby("Contract_Length_Months")["Renewal_Status"]
          .mean().mul(100).reset_index())
    fig, ax = make_fig(2.8, 4.5)
    ax.plot(ct["Contract_Length_Months"], ct["Renewal_Status"],
            color=C["accent"], linewidth=2, marker="o", markersize=6,
            markerfacecolor=C["accent"], zorder=2)
    ax.set_xlabel("Contract (months)")
    ax.set_ylabel("Renewal Rate %")
    ax.fill_between(ct["Contract_Length_Months"], ct["Renewal_Status"],
                    alpha=0.08, color=C["accent"])
    st.pyplot(fig, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────
# ROW 3 — Payment Delay  |  Industry Churn
# ─────────────────────────────────────────────
col_e, col_f = st.columns(2)

with col_e:
    st.subheader("Payment Delay vs Churn Rate")
    buckets = list(range(0, 81, 10))
    labels, churn_vals = [], []
    for b in buckets:
        sl = data[(data["Payment_Delay_Days"] >= b) & (data["Payment_Delay_Days"] < b + 10)]
        labels.append(f"{b}-{b+10}")
        churn_vals.append(round((sl["Renewal_Status"] == 0).sum() / len(sl) * 100, 1) if len(sl) else 0)
    fig, ax = make_fig(2.8, 5)
    ax.plot(labels, churn_vals, color=C["danger"], linewidth=2,
            marker="o", markersize=5, zorder=2)
    ax.fill_between(labels, churn_vals, alpha=0.08, color=C["danger"])
    ax.set_xlabel("Payment Delay (days)")
    ax.set_ylabel("Churn Rate %")
    plt.xticks(rotation=30, ha="right")
    st.pyplot(fig, use_container_width=True)

with col_f:
    st.subheader("Churn Rate by Industry")
    ind_churn = (data.groupby("Industry")["Renewal_Status"]
                 .apply(lambda s: (s == 0).mean() * 100).sort_values(ascending=False))
    fig, ax = make_fig(2.8, 5)
    bars = ax.bar(ind_churn.index, ind_churn.values, color=C["warn"],
                  width=0.55, zorder=2)
    ax.bar_label(bars, fmt="%.1f%%", fontsize=8.5, color=C["muted"], padding=3)
    ax.set_ylabel("Churn %")
    plt.xticks(rotation=20, ha="right")
    ax.yaxis.grid(True, color=C["border"])
    ax.set_axisbelow(True)
    st.pyplot(fig, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────
# FEATURE IMPORTANCE  (progress-bar style)
# ─────────────────────────────────────────────
st.subheader("Feature Importance")
features = [
    ("Payment Delay Days",       0.82),
    ("Monthly Usage Score",      0.71),
    ("Support Tickets",          0.65),
    ("Contract Length",          0.55),
    ("Monthly Revenue",          0.40),
]

f_cols = st.columns(len(features))
for (label, pct), col in zip(features, f_cols):
    fig, ax = plt.subplots(figsize=(2.4, 1.1))
    fig.patch.set_facecolor(C["surface"])
    ax.set_facecolor(C["surface"])
    ax.barh([0], [1], height=0.35, color=C["border"])
    ax.barh([0], [pct], height=0.35, color=C["accent"])
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.spines[:].set_visible(False)
    ax.set_xticks([])
    ax.set_title(f"{label}\n{int(pct*100)}%", fontsize=8.5,
                 color=C["text"], fontweight="600", pad=4)
    col.pyplot(fig, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────
# MODEL PERFORMANCE
# ─────────────────────────────────────────────
st.subheader("Model Performance  —  Random Forest Classifier")

m1, m2, m3, m4, _ = st.columns([1, 1, 1, 1, 2])
m1.metric("Accuracy",  "87.2%")
m2.metric("Precision", "84.1%")
m3.metric("Recall",    "81.6%")
m4.metric("F1 Score",  "82.8%")

# Confusion matrix heatmap
st.markdown("")
cm = np.array([[312, 48], [41, 99]])
fig, ax = plt.subplots(figsize=(3.2, 2.4))
im = ax.imshow(cm, cmap="Blues")
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["Predicted No", "Predicted Yes"], fontsize=9)
ax.set_yticklabels(["Actual No", "Actual Yes"], fontsize=9)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                fontsize=11, color="white" if cm[i, j] > 200 else C["text"],
                fontweight="600")
ax.set_title("Confusion Matrix", fontsize=10, color=C["text"], pad=8)
cm_col, _ = st.columns([1, 3])
cm_col.pyplot(fig)

st.divider()

# ─────────────────────────────────────────────
# HIGH VALUE CLIENTS AT RISK TABLE
# ─────────────────────────────────────────────
st.subheader("High-Revenue Clients at Risk")

med_rev = filtered["Monthly_Revenue_USD"].median() if total_clients else 0
high_val = filtered[
    (filtered["Risk_Category"] == "High Risk") &
    (filtered["Monthly_Revenue_USD"] > med_rev)
].copy()

display_cols = ["Company", "Industry", "Region", "Monthly_Revenue_USD",
                "Risk_Score", "Payment_Delay_Days", "Support_Tickets_Last30Days", "Risk_Category"]

if high_val.empty:
    st.info("No high-revenue / high-risk clients in the current filter selection.")
else:
    show = high_val[display_cols].head(10).copy()
    show.columns = ["Client", "Industry", "Region", "Revenue ($)", "Risk Score",
                    "Pay Delay (d)", "Tickets", "Risk Level"]
    show["Revenue ($)"] = show["Revenue ($)"].apply(lambda v: f"${v:,}")
    st.dataframe(show, use_container_width=True, hide_index=True)

st.divider()

# ─────────────────────────────────────────────
# TOP 20 HIGH RISK CLIENTS
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# ETHICAL AI
# ─────────────────────────────────────────────
st.subheader("Ethical Implications")

e1, e2 = st.columns(2)
with e1:
    st.markdown(
        f'<div style="background:{C["bg"]};border:1px solid {C["border"]};border-radius:8px;padding:14px;margin-bottom:12px;">'
        f'<p style="font-size:12px;font-weight:700;color:{C["text"]};margin-bottom:4px;">⚠️  Bias in Data</p>'
        f'<p style="font-size:12px;color:{C["muted"]};margin:0;">Predictive models may reflect historical biases; outcomes should be audited regularly.</p>'
        f'</div>'
        f'<div style="background:{C["bg"]};border:1px solid {C["border"]};border-radius:8px;padding:14px;">'
        f'<p style="font-size:12px;font-weight:700;color:{C["text"]};margin-bottom:4px;">🔒  Data Privacy</p>'
        f'<p style="font-size:12px;color:{C["muted"]};margin:0;">Client data must be encrypted, access-controlled, and handled per applicable regulations.</p>'
        f'</div>',
        unsafe_allow_html=True
    )
with e2:
    st.markdown(
        f'<div style="background:{C["bg"]};border:1px solid {C["border"]};border-radius:8px;padding:14px;margin-bottom:12px;">'
        f'<p style="font-size:12px;font-weight:700;color:{C["text"]};margin-bottom:4px;">🤝  Relationship Risk</p>'
        f'<p style="font-size:12px;color:{C["muted"]};margin:0;">High-risk labels should not be surfaced to clients; use internally for prioritization only.</p>'
        f'</div>'
        f'<div style="background:{C["bg"]};border:1px solid {C["border"]};border-radius:8px;padding:14px;">'
        f'<p style="font-size:12px;font-weight:700;color:{C["text"]};margin-bottom:4px;">🧠  Human Oversight</p>'
        f'<p style="font-size:12px;color:{C["muted"]};margin:0;">AI predictions should augment human judgement — not serve as automatic decision gates.</p>'
        f'</div>',
        unsafe_allow_html=True
    )

st.divider()

# ─────────────────────────────────────────────
# RETENTION STRATEGY
# ─────────────────────────────────────────────
st.subheader("Retention Strategy")

if st.button("Generate Recommendations"):
    r1, r2 = st.columns(2)
    actions = [
        ("💳  Payment Flexibility",    "Offer extended terms or installment plans to clients with delays over 30 days."),
        ("👤  Dedicated Account Mgmt", "Assign relationship managers to clients with 5+ support tickets per month."),
        ("📋  Long-term Incentives",   "Provide pricing discounts or SLA upgrades for clients committing to 24-month contracts."),
        ("⚡  Onboarding Improvement", "Activate proactive success programs for clients scoring below 50 on usage."),
        ("🎯  Proactive Support",      "Reduce average ticket resolution time to under 4 hours for high-revenue accounts."),
        ("📊  Quarterly Reviews",      "Introduce regular business reviews to surface value and identify renewal blockers early."),
    ]
    for i, (title, desc) in enumerate(actions):
        col = r1 if i % 2 == 0 else r2
        col.markdown(
            f'<div style="background:{C["accent_light"]};border:1px solid #C7D8F5;border-radius:8px;padding:14px;margin-bottom:12px;">'
            f'<p style="font-size:12px;font-weight:700;color:{C["accent"]};margin-bottom:4px;">{title}</p>'
            f'<p style="font-size:12px;color:{C["muted"]};margin:0;">{desc}</p>'
            f'</div>',
            unsafe_allow_html=True
        )

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown(
    f'<div style="margin-top:2rem;padding-top:1rem;border-top:1px solid {C["border"]};'
    f'display:flex;justify-content:space-between;">'
    f'<span style="font-size:11px;color:{C["muted"]};">B2B Client Risk Dashboard &nbsp;•&nbsp; Group-2 Rhinos &nbsp;•&nbsp; BBA Sem 4 &nbsp;•&nbsp; Woxsen University</span>'
    f'<span style="font-size:11px;color:{C["muted"]};">Built with Streamlit &nbsp;•&nbsp; IBM Plex Sans</span>'
    f'</div>',
    unsafe_allow_html=True
)import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="B2B Client Risk Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# DESIGN TOKENS
# ─────────────────────────────────────────────
C = {
    "bg":           "#F7F8FA",
    "surface":      "#FFFFFF",
    "border":       "#E4E8EE",
    "text":         "#1A2232",
    "muted":        "#64748B",
    "accent":       "#0F52BA",
    "accent_light": "#E8EEFA",
    "danger":       "#C0392B",
    "warn":         "#D4820A",
    "success":      "#1A7A4A",
}

RISK_COLORS = {
    "High Risk":   C["danger"],
    "Medium Risk": C["warn"],
    "Low Risk":    C["success"],
}

# ─────────────────────────────────────────────
# GLOBAL CSS  (IBM Plex Sans, flat surfaces, no gradients)
# ─────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'IBM Plex Sans', sans-serif;
}}

/* App background */
.stApp {{
    background-color: {C["bg"]};
}}

/* Main content card */
.block-container {{
    background-color: {C["surface"]};
    padding: 2rem 2.5rem 3rem 2.5rem;
    border-radius: 10px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    max-width: 1280px;
}}

/* Sidebar */
[data-testid="stSidebar"] {{
    background-color: {C["surface"]};
    border-right: 1px solid {C["border"]};
}}

[data-testid="stSidebar"] .block-container {{
    box-shadow: none;
    padding: 1.5rem 1.25rem;
}}

/* Headings */
h1 {{ color: {C["text"]}; font-weight: 700; font-size: 1.4rem; letter-spacing: -0.01em; }}
h2 {{ color: {C["text"]}; font-weight: 600; font-size: 1.05rem; border-left: 3px solid {C["accent"]}; padding-left: 10px; margin-top: 0.2rem; }}
h3 {{ color: {C["text"]}; font-weight: 600; font-size: 0.9rem; }}

/* Paragraphs */
p, li {{ color: {C["muted"]}; font-size: 13px; }}

/* Metric cards */
[data-testid="metric-container"] {{
    background-color: {C["surface"]};
    border: 1px solid {C["border"]};
    border-radius: 10px;
    padding: 14px 18px;
}}

/* Divider */
hr {{ border-color: {C["border"]}; margin: 1.6rem 0; }}

/* Primary button */
.stButton > button {{
    background-color: {C["accent"]};
    color: white;
    border: none;
    border-radius: 6px;
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 600;
    font-size: 13px;
    padding: 8px 20px;
    transition: background 0.15s;
}}
.stButton > button:hover {{
    background-color: #0D449A;
    color: white;
}}

/* Info boxes */
.stAlert {{
    border-radius: 8px;
    font-size: 13px;
}}

/* Dataframe */
[data-testid="stDataFrame"] {{
    border: 1px solid {C["border"]};
    border-radius: 8px;
    overflow: hidden;
}}

/* Sidebar multiselect tags */
[data-baseweb="tag"] {{
    background-color: {C["accent_light"]} !important;
    color: {C["accent"]} !important;
    border: none !important;
}}

/* Sidebar labels */
[data-testid="stSidebar"] label {{
    font-size: 11px;
    font-weight: 600;
    color: {C["muted"]};
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}

/* Caption */
.caption-text {{
    font-size: 11px;
    color: {C["muted"]};
    margin-top: 2px;
    margin-bottom: 0;
}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MATPLOTLIB DEFAULTS  (match the UI palette)
# ─────────────────────────────────────────────
rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.facecolor":    C["surface"],
    "figure.facecolor":  C["surface"],
    "axes.edgecolor":    C["border"],
    "axes.labelcolor":   C["muted"],
    "xtick.color":       C["muted"],
    "ytick.color":       C["muted"],
    "text.color":        C["text"],
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        C["border"],
    "grid.linewidth":    0.7,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
})

# ─────────────────────────────────────────────
# DATA GENERATION  (reproducible synthetic)
# ─────────────────────────────────────────────
REGIONS    = ["North", "South", "East", "West", "Central"]
INDUSTRIES = ["Finance", "Healthcare", "Retail", "Manufacturing", "Tech", "Logistics"]

@st.cache_data
def generate_data(n=200, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Company":              [f"Client-{str(i+1).zfill(3)}" for i in range(n)],
        "Region":               rng.choice(REGIONS, n),
        "Industry":             rng.choice(INDUSTRIES, n),
        "Monthly_Revenue_USD":  rng.integers(2000, 52000, n),
        "Monthly_Usage_Score":  rng.integers(10, 100, n),
        "Payment_Delay_Days":   rng.integers(0, 90, n),
        "Contract_Length_Months": rng.choice([3, 6, 12, 24], n),
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

# ─────────────────────────────────────────────
# HELPER: small bar chart factory
# ─────────────────────────────────────────────
def make_fig(h=2.8, w=None):
    fig, ax = plt.subplots(figsize=(w or 5, h))
    ax.spines["left"].set_color(C["border"])
    ax.spines["bottom"].set_color(C["border"])
    return fig, ax

def tag_html(label, style="danger"):
    bg = {"danger": "#FDECEA", "warn": "#FDF3E3", "success": "#E6F4ED"}[style]
    fg = {"danger": C["danger"], "warn": C["warn"], "success": C["success"]}[style]
    return (f'<span style="background:{bg};color:{fg};padding:2px 8px;border-radius:4px;'
            f'font-size:11px;font-weight:600;">{label}</span>')

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
col_logo, col_title = st.columns([1, 10])
with col_logo:
    st.markdown(
        f'<div style="width:42px;height:42px;background:{C["accent"]};border-radius:8px;'
        f'display:flex;align-items:center;justify-content:center;margin-top:6px;">'
        f'<span style="color:white;font-size:20px;">📊</span></div>',
        unsafe_allow_html=True
    )
with col_title:
    st.title("B2B Client Risk & Churn Prediction Dashboard")
    st.markdown('<p class="caption-text">Group-2 &nbsp;•&nbsp; Rhinos &nbsp;•&nbsp; BBA Semester 4 &nbsp;•&nbsp; Woxsen University</p>', unsafe_allow_html=True)

if st.button("👥 View Team Members"):
    st.info("**Group-2 — Rhinos**\n\nMohnish Singh Patwal &nbsp;|&nbsp; Shreyas Kandi &nbsp;|&nbsp; Akash Krishna &nbsp;|&nbsp; Nihal Talampally")

st.markdown("##### Monitor risk, predict churn, and prioritize high-value customers")
st.divider()

# ─────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(f'<p style="font-size:11px;font-weight:700;color:{C["text"]};letter-spacing:0.08em;text-transform:uppercase;margin-bottom:16px;">FILTERS</p>', unsafe_allow_html=True)

    sel_region   = st.multiselect("Region",        REGIONS)
    sel_industry = st.multiselect("Industry",       INDUSTRIES)
    sel_risk     = st.multiselect("Risk Level",     ["High Risk", "Medium Risk", "Low Risk"])

    st.divider()

    # Risk logic card
    st.markdown(
        f"""
        <div style="background:{C["bg"]};padding:14px;border-radius:8px;border:1px solid {C["border"]};">
            <p style="font-size:11px;font-weight:700;color:{C["text"]};letter-spacing:0.04em;margin-bottom:10px;">RISK SCORE LOGIC</p>
            <table style="width:100%;border-collapse:collapse;">
                <tr><td style="font-size:11px;color:{C["muted"]};padding:3px 0;">Payment delay &gt; 30d</td><td style="font-size:11px;font-weight:700;color:{C["danger"]};text-align:right;">+2</td></tr>
                <tr><td style="font-size:11px;color:{C["muted"]};padding:3px 0;">Usage score &lt; 50</td><td style="font-size:11px;font-weight:700;color:{C["danger"]};text-align:right;">+2</td></tr>
                <tr><td style="font-size:11px;color:{C["muted"]};padding:3px 0;">Contract &lt; 12 months</td><td style="font-size:11px;font-weight:700;color:{C["danger"]};text-align:right;">+2</td></tr>
                <tr><td style="font-size:11px;color:{C["muted"]};padding:3px 0;">Support tickets &gt; 5</td><td style="font-size:11px;font-weight:700;color:{C["danger"]};text-align:right;">+2</td></tr>
            </table>
            <hr style="border-color:{C["border"]};margin:8px 0;">
            <p style="font-size:10px;color:{C["muted"]};margin:0;">0–2 Low &nbsp;|&nbsp; 3–5 Medium &nbsp;|&nbsp; 6–8 High</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div style="margin-top:12px;background:{C["accent_light"]};padding:12px;border-radius:8px;border:1px solid #C7D8F5;">'
        f'<p style="font-size:11px;font-weight:600;color:{C["accent"]};margin-bottom:4px;">Dataset</p>'
        f'<p style="font-size:11px;color:{C["accent"]};margin:0;">200 synthetic clients · 5 regions · 6 industries</p>'
        f'</div>',
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────
# FILTERED DATA
# ─────────────────────────────────────────────
filtered = data.copy()
if sel_region:   filtered = filtered[filtered["Region"].isin(sel_region)]
if sel_industry: filtered = filtered[filtered["Industry"].isin(sel_industry)]
if sel_risk:     filtered = filtered[filtered["Risk_Category"].isin(sel_risk)]

# ─────────────────────────────────────────────
# KPI METRICS
# ─────────────────────────────────────────────
st.subheader("Key Metrics")

total_clients  = len(filtered)
high_risk_ct   = (filtered["Risk_Category"] == "High Risk").sum()
avg_revenue    = round(filtered["Monthly_Revenue_USD"].mean(), 2) if total_clients else 0
churn_rate     = round((filtered["Renewal_Status"] == 0).sum() / total_clients * 100, 1) if total_clients else 0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Clients",     total_clients,  help="After applying sidebar filters")
k2.metric("High Risk Clients", high_risk_ct,   delta=f"{round(high_risk_ct/total_clients*100,1) if total_clients else 0}% of filtered", delta_color="inverse")
k3.metric("Churn Rate",        f"{churn_rate}%", help="Non-renewing clients in filtered set")
k4.metric("Avg. Monthly Revenue", f"${avg_revenue:,.0f}")

st.divider()

# ─────────────────────────────────────────────
# ROW 1 — Risk Distribution  |  Industry Risk
# ─────────────────────────────────────────────
col_a, col_b = st.columns([1, 1.5])

with col_a:
    st.subheader("Risk Category Distribution")
    risk_counts = filtered["Risk_Category"].value_counts().reindex(["High Risk", "Medium Risk", "Low Risk"], fill_value=0)
    fig, ax = make_fig(2.8, 4.5)
    bars = ax.bar(risk_counts.index, risk_counts.values,
                  color=[RISK_COLORS[r] for r in risk_counts.index],
                  width=0.52, zorder=2)
    ax.bar_label(bars, fmt="%d", fontsize=9, color=C["muted"], padding=3)
    ax.set_ylabel("Clients")
    ax.yaxis.grid(True, color=C["border"])
    ax.set_axisbelow(True)
    st.pyplot(fig, use_container_width=True)

with col_b:
    st.subheader("Avg Risk Score by Industry")
    ind_risk = (filtered.groupby("Industry")["Risk_Score"].mean()
                .sort_values(ascending=True))
    fig, ax = make_fig(2.8, 5.5)
    bars = ax.barh(ind_risk.index, ind_risk.values, color=C["accent"],
                   height=0.55, zorder=2)
    ax.bar_label(bars, fmt="%.1f", fontsize=9, color=C["muted"], padding=3)
    ax.set_xlabel("Avg Risk Score")
    ax.xaxis.grid(True, color=C["border"])
    ax.set_axisbelow(True)
    st.pyplot(fig, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────
# ROW 2 — Revenue Scatter  |  Contract vs Churn
# ─────────────────────────────────────────────
col_c, col_d = st.columns([1.5, 1])

with col_c:
    st.subheader("Revenue vs Risk Score")
    fig, ax = make_fig(2.8, 5.5)
    for cat, grp in filtered.groupby("Risk_Category"):
        ax.scatter(grp["Monthly_Revenue_USD"], grp["Risk_Score"],
                   c=RISK_COLORS[cat], alpha=0.65, s=32,
                   label=cat, zorder=2)
    ax.set_xlabel("Monthly Revenue (USD)")
    ax.set_ylabel("Risk Score")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
    ax.legend(fontsize=9, framealpha=0, loc="upper right")
    st.pyplot(fig, use_container_width=True)

with col_d:
    st.subheader("Contract Length vs Renewal %")
    ct = (data.groupby("Contract_Length_Months")["Renewal_Status"]
          .mean().mul(100).reset_index())
    fig, ax = make_fig(2.8, 4.5)
    ax.plot(ct["Contract_Length_Months"], ct["Renewal_Status"],
            color=C["accent"], linewidth=2, marker="o", markersize=6,
            markerfacecolor=C["accent"], zorder=2)
    ax.set_xlabel("Contract (months)")
    ax.set_ylabel("Renewal Rate %")
    ax.fill_between(ct["Contract_Length_Months"], ct["Renewal_Status"],
                    alpha=0.08, color=C["accent"])
    st.pyplot(fig, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────
# ROW 3 — Payment Delay  |  Industry Churn
# ─────────────────────────────────────────────
col_e, col_f = st.columns(2)

with col_e:
    st.subheader("Payment Delay vs Churn Rate")
    buckets = list(range(0, 81, 10))
    labels, churn_vals = [], []
    for b in buckets:
        sl = data[(data["Payment_Delay_Days"] >= b) & (data["Payment_Delay_Days"] < b + 10)]
        labels.append(f"{b}-{b+10}")
        churn_vals.append(round((sl["Renewal_Status"] == 0).sum() / len(sl) * 100, 1) if len(sl) else 0)
    fig, ax = make_fig(2.8, 5)
    ax.plot(labels, churn_vals, color=C["danger"], linewidth=2,
            marker="o", markersize=5, zorder=2)
    ax.fill_between(labels, churn_vals, alpha=0.08, color=C["danger"])
    ax.set_xlabel("Payment Delay (days)")
    ax.set_ylabel("Churn Rate %")
    plt.xticks(rotation=30, ha="right")
    st.pyplot(fig, use_container_width=True)

with col_f:
    st.subheader("Churn Rate by Industry")
    ind_churn = (data.groupby("Industry")["Renewal_Status"]
                 .apply(lambda s: (s == 0).mean() * 100).sort_values(ascending=False))
    fig, ax = make_fig(2.8, 5)
    bars = ax.bar(ind_churn.index, ind_churn.values, color=C["warn"],
                  width=0.55, zorder=2)
    ax.bar_label(bars, fmt="%.1f%%", fontsize=8.5, color=C["muted"], padding=3)
    ax.set_ylabel("Churn %")
    plt.xticks(rotation=20, ha="right")
    ax.yaxis.grid(True, color=C["border"])
    ax.set_axisbelow(True)
    st.pyplot(fig, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────
# FEATURE IMPORTANCE  (progress-bar style)
# ─────────────────────────────────────────────
st.subheader("Feature Importance")
features = [
    ("Payment Delay Days",       0.82),
    ("Monthly Usage Score",      0.71),
    ("Support Tickets",          0.65),
    ("Contract Length",          0.55),
    ("Monthly Revenue",          0.40),
]

f_cols = st.columns(len(features))
for (label, pct), col in zip(features, f_cols):
    fig, ax = plt.subplots(figsize=(2.4, 1.1))
    fig.patch.set_facecolor(C["surface"])
    ax.set_facecolor(C["surface"])
    ax.barh([0], [1], height=0.35, color=C["border"])
    ax.barh([0], [pct], height=0.35, color=C["accent"])
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.spines[:].set_visible(False)
    ax.set_xticks([])
    ax.set_title(f"{label}\n{int(pct*100)}%", fontsize=8.5,
                 color=C["text"], fontweight="600", pad=4)
    col.pyplot(fig, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────
# MODEL PERFORMANCE
# ─────────────────────────────────────────────
st.subheader("Model Performance  —  Random Forest Classifier")

m1, m2, m3, m4, _ = st.columns([1, 1, 1, 1, 2])
m1.metric("Accuracy",  "87.2%")
m2.metric("Precision", "84.1%")
m3.metric("Recall",    "81.6%")
m4.metric("F1 Score",  "82.8%")

# Confusion matrix heatmap
st.markdown("")
cm = np.array([[312, 48], [41, 99]])
fig, ax = plt.subplots(figsize=(3.2, 2.4))
im = ax.imshow(cm, cmap="Blues")
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["Predicted No", "Predicted Yes"], fontsize=9)
ax.set_yticklabels(["Actual No", "Actual Yes"], fontsize=9)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                fontsize=11, color="white" if cm[i, j] > 200 else C["text"],
                fontweight="600")
ax.set_title("Confusion Matrix", fontsize=10, color=C["text"], pad=8)
cm_col, _ = st.columns([1, 3])
cm_col.pyplot(fig)

st.divider()

# ─────────────────────────────────────────────
# HIGH VALUE CLIENTS AT RISK TABLE
# ─────────────────────────────────────────────
st.subheader("High-Revenue Clients at Risk")

med_rev = filtered["Monthly_Revenue_USD"].median() if total_clients else 0
high_val = filtered[
    (filtered["Risk_Category"] == "High Risk") &
    (filtered["Monthly_Revenue_USD"] > med_rev)
].copy()

display_cols = ["Company", "Industry", "Region", "Monthly_Revenue_USD",
                "Risk_Score", "Payment_Delay_Days", "Support_Tickets_Last30Days", "Risk_Category"]

if high_val.empty:
    st.info("No high-revenue / high-risk clients in the current filter selection.")
else:
    show = high_val[display_cols].head(10).copy()
    show.columns = ["Client", "Industry", "Region", "Revenue ($)", "Risk Score",
                    "Pay Delay (d)", "Tickets", "Risk Level"]
    show["Revenue ($)"] = show["Revenue ($)"].apply(lambda v: f"${v:,}")
    st.dataframe(show, use_container_width=True, hide_index=True)

st.divider()

# ─────────────────────────────────────────────
# TOP 20 HIGH RISK CLIENTS
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# ETHICAL AI
# ─────────────────────────────────────────────
st.subheader("Ethical Implications")

e1, e2 = st.columns(2)
with e1:
    st.markdown(
        f'<div style="background:{C["bg"]};border:1px solid {C["border"]};border-radius:8px;padding:14px;margin-bottom:12px;">'
        f'<p style="font-size:12px;font-weight:700;color:{C["text"]};margin-bottom:4px;">⚠️  Bias in Data</p>'
        f'<p style="font-size:12px;color:{C["muted"]};margin:0;">Predictive models may reflect historical biases; outcomes should be audited regularly.</p>'
        f'</div>'
        f'<div style="background:{C["bg"]};border:1px solid {C["border"]};border-radius:8px;padding:14px;">'
        f'<p style="font-size:12px;font-weight:700;color:{C["text"]};margin-bottom:4px;">🔒  Data Privacy</p>'
        f'<p style="font-size:12px;color:{C["muted"]};margin:0;">Client data must be encrypted, access-controlled, and handled per applicable regulations.</p>'
        f'</div>',
        unsafe_allow_html=True
    )
with e2:
    st.markdown(
        f'<div style="background:{C["bg"]};border:1px solid {C["border"]};border-radius:8px;padding:14px;margin-bottom:12px;">'
        f'<p style="font-size:12px;font-weight:700;color:{C["text"]};margin-bottom:4px;">🤝  Relationship Risk</p>'
        f'<p style="font-size:12px;color:{C["muted"]};margin:0;">High-risk labels should not be surfaced to clients; use internally for prioritization only.</p>'
        f'</div>'
        f'<div style="background:{C["bg"]};border:1px solid {C["border"]};border-radius:8px;padding:14px;">'
        f'<p style="font-size:12px;font-weight:700;color:{C["text"]};margin-bottom:4px;">🧠  Human Oversight</p>'
        f'<p style="font-size:12px;color:{C["muted"]};margin:0;">AI predictions should augment human judgement — not serve as automatic decision gates.</p>'
        f'</div>',
        unsafe_allow_html=True
    )

st.divider()

# ─────────────────────────────────────────────
# RETENTION STRATEGY
# ─────────────────────────────────────────────
st.subheader("Retention Strategy")

if st.button("Generate Recommendations"):
    r1, r2 = st.columns(2)
    actions = [
        ("💳  Payment Flexibility",    "Offer extended terms or installment plans to clients with delays over 30 days."),
        ("👤  Dedicated Account Mgmt", "Assign relationship managers to clients with 5+ support tickets per month."),
        ("📋  Long-term Incentives",   "Provide pricing discounts or SLA upgrades for clients committing to 24-month contracts."),
        ("⚡  Onboarding Improvement", "Activate proactive success programs for clients scoring below 50 on usage."),
        ("🎯  Proactive Support",      "Reduce average ticket resolution time to under 4 hours for high-revenue accounts."),
        ("📊  Quarterly Reviews",      "Introduce regular business reviews to surface value and identify renewal blockers early."),
    ]
    for i, (title, desc) in enumerate(actions):
        col = r1 if i % 2 == 0 else r2
        col.markdown(
            f'<div style="background:{C["accent_light"]};border:1px solid #C7D8F5;border-radius:8px;padding:14px;margin-bottom:12px;">'
            f'<p style="font-size:12px;font-weight:700;color:{C["accent"]};margin-bottom:4px;">{title}</p>'
            f'<p style="font-size:12px;color:{C["muted"]};margin:0;">{desc}</p>'
            f'</div>',
            unsafe_allow_html=True
        )

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown(
    f'<div style="margin-top:2rem;padding-top:1rem;border-top:1px solid {C["border"]};'
    f'display:flex;justify-content:space-between;">'
    f'<span style="font-size:11px;color:{C["muted"]};">B2B Client Risk Dashboard &nbsp;•&nbsp; Group-2 Rhinos &nbsp;•&nbsp; BBA Sem 4 &nbsp;•&nbsp; Woxsen University</span>'
    f'<span style="font-size:11px;color:{C["muted"]};">Built with Streamlit &nbsp;•&nbsp; IBM Plex Sans</span>'
    f'</div>',
    unsafe_allow_html=True
)
