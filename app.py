import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="Client Risk Intelligence | Rhinos",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== GLOBAL CSS =====
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');

/* ── Root Variables ── */
:root {
    --bg:          #080C14;
    --surface:     #0E1520;
    --surface2:    #131D2E;
    --border:      rgba(99,179,255,0.12);
    --accent-blue: #3B8BFF;
    --accent-cyan: #00E5C7;
    --accent-warn: #FFB547;
    --accent-danger:#FF5C7A;
    --accent-ok:   #4ADE80;
    --text-primary:#E8F0FF;
    --text-muted:  #6B7FA8;
    --glow-blue:   0 0 30px rgba(59,139,255,0.25);
    --glow-cyan:   0 0 30px rgba(0,229,199,0.2);
}

/* ── Reset & Base ── */
html, body, [class*="css"], .stApp {
    font-family: 'Space Grotesk', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text-primary) !important;
}

/* ── Hide Streamlit Chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 3rem; max-width: 1400px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
[data-testid="stSidebar"] .stMultiSelect > div > div {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* ── Top Navigation Bar ── */
.top-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 0 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
.brand-name {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #3B8BFF, #00E5C7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.brand-sub {
    font-size: 0.72rem;
    color: var(--text-muted);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 0.15rem;
}
.badge-pill {
    display: inline-block;
    background: rgba(59,139,255,0.15);
    border: 1px solid rgba(59,139,255,0.35);
    color: var(--accent-blue);
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    padding: 0.25rem 0.7rem;
    border-radius: 99px;
    text-transform: uppercase;
    margin-left: 0.5rem;
}

/* ── Section Title ── */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.01em;
    margin-bottom: 0.25rem;
}
.section-subtitle {
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-bottom: 1rem;
}

/* ── KPI Cards ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.kpi-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--glow-blue);
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan));
}
.kpi-icon {
    font-size: 1.5rem;
    margin-bottom: 0.5rem;
}
.kpi-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.kpi-label {
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 500;
}
.kpi-blue  .kpi-value { color: var(--accent-blue); }
.kpi-cyan  .kpi-value { color: var(--accent-cyan); }
.kpi-warn  .kpi-value { color: var(--accent-warn); }
.kpi-danger .kpi-value { color: var(--accent-danger); }

/* ── Panel Cards ── */
.panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    position: relative;
    overflow: hidden;
}
.panel::after {
    content: '';
    position: absolute;
    bottom: 0; right: 0;
    width: 120px; height: 120px;
    background: radial-gradient(circle, rgba(59,139,255,0.07) 0%, transparent 70%);
    pointer-events: none;
}

/* ── Gauge Bar ── */
.gauge-track {
    background: var(--surface2);
    border-radius: 99px;
    height: 10px;
    overflow: hidden;
    margin: 0.6rem 0;
}
.gauge-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 1s ease;
}
.gauge-low    { background: linear-gradient(90deg, #4ADE80, #22D3EE); }
.gauge-medium { background: linear-gradient(90deg, #FFB547, #FF8C00); }
.gauge-high   { background: linear-gradient(90deg, #FF5C7A, #FF2D55); }

/* ── Status Chips ── */
.chip {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.35rem 0.9rem;
    border-radius: 99px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}
.chip-ok     { background: rgba(74,222,128,0.12); color: var(--accent-ok);     border: 1px solid rgba(74,222,128,0.25); }
.chip-warn   { background: rgba(255,181,71,0.12);  color: var(--accent-warn);   border: 1px solid rgba(255,181,71,0.25); }
.chip-danger { background: rgba(255,92,122,0.12);  color: var(--accent-danger); border: 1px solid rgba(255,92,122,0.25); }

/* ── Table Styling ── */
.stDataFrame, [data-testid="stDataFrame"] {
    background: var(--surface) !important;
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
    overflow: hidden !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-blue), #1a5ccc) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.04em !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(59,139,255,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(59,139,255,0.45) !important;
}

/* ── Info / Warning / Error Boxes ── */
.stAlert {
    border-radius: 10px !important;
    border: none !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* ── Divider ── */
.custom-divider {
    border: none;
    height: 1px;
    background: var(--border);
    margin: 1.5rem 0;
}

/* ── Ethics grid ── */
.ethics-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.75rem;
    margin-top: 0.8rem;
}
.ethics-item {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent-cyan);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.82rem;
    color: var(--text-primary);
    line-height: 1.5;
}

/* ── Strategy Items ── */
.strategy-item {
    display: flex;
    align-items: flex-start;
    gap: 0.7rem;
    padding: 0.75rem 1rem;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    margin-bottom: 0.5rem;
    font-size: 0.83rem;
    line-height: 1.5;
}
.strategy-icon {
    font-size: 1.1rem;
    flex-shrink: 0;
    margin-top: 0.1rem;
}

/* ── Team Member Cards ── */
.team-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.75rem;
    margin-top: 0.8rem;
}
.team-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    display: flex;
    align-items: center;
    gap: 0.7rem;
}
.team-avatar {
    width: 36px; height: 36px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan));
    display: flex; align-items: center; justify-content: center;
    font-size: 0.9rem; font-weight: 700; color: white;
    flex-shrink: 0;
}
.team-name { font-size: 0.82rem; font-weight: 600; color: var(--text-primary); }
.team-role { font-size: 0.68rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; }

/* ── Accuracy Badge ── */
.accuracy-display {
    display: flex;
    align-items: center;
    gap: 1.2rem;
    margin: 0.8rem 0;
}
.accuracy-number {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    color: var(--accent-cyan);
    letter-spacing: -0.04em;
    line-height: 1;
}
.accuracy-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── Matplotlib figure styling ── */
.stPlotlyChart, [data-testid="stPlotlyChart"] { border-radius: 12px !important; }

</style>
""", unsafe_allow_html=True)

# ===== MATPLOTLIB GLOBAL STYLE =====
plt.rcParams.update({
    'figure.facecolor':  '#0E1520',
    'axes.facecolor':    '#0E1520',
    'axes.edgecolor':    '#1E2D44',
    'axes.labelcolor':   '#6B7FA8',
    'axes.titlecolor':   '#E8F0FF',
    'xtick.color':       '#6B7FA8',
    'ytick.color':       '#6B7FA8',
    'grid.color':        '#1E2D44',
    'grid.alpha':        0.5,
    'text.color':        '#E8F0FF',
    'font.family':       'DejaVu Sans',
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

BLUE   = '#3B8BFF'
CYAN   = '#00E5C7'
WARN   = '#FFB547'
DANGER = '#FF5C7A'
OK     = '#4ADE80'

# ===== TOP BAR =====
st.markdown("""
<div class="top-bar">
  <div>
    <div class="brand-name">RiskIQ Dashboard</div>
    <div class="brand-sub">B2B Client Risk &amp; Churn Intelligence</div>
  </div>
  <div>
    <span class="badge-pill">BBA Sem 4</span>
    <span class="badge-pill">Woxsen University</span>
    <span class="badge-pill">Group-2 · Rhinos</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ===== SIDEBAR =====
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:0.5rem 0 1.2rem;">
        <div style="font-family:'Syne',sans-serif;font-size:1.15rem;font-weight:800;
                    background:linear-gradient(135deg,#3B8BFF,#00E5C7);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            ⚡ RiskIQ
        </div>
        <div style="font-size:0.68rem;color:#6B7FA8;text-transform:uppercase;letter-spacing:0.08em;margin-top:0.2rem;">
            Filter &amp; Explore
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-size:0.72rem;color:#6B7FA8;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.4rem;">Region</div>', unsafe_allow_html=True)

    # Load data early for sidebar
    try:
        data = pd.read_csv("B2B_Client_Churn_5000.csv")
        data.columns = data.columns.str.strip().str.replace(" ", "_")
        data['Renewal_Status'] = data['Renewal_Status'].map({'Yes': 1, 'No': 0})

        def calculate_risk(row):
            risk = 0
            if row['Payment_Delay_Days'] > 30:   risk += 2
            if row['Monthly_Usage_Score'] < 50:   risk += 2
            if row['Contract_Length_Months'] < 12: risk += 2
            if row['Support_Tickets_Last30Days'] > 5: risk += 2
            return risk

        data['Risk_Score'] = data.apply(calculate_risk, axis=1)
        data['Risk_Category'] = data['Risk_Score'].apply(
            lambda s: "Low Risk" if s <= 2 else ("Medium Risk" if s <= 5 else "High Risk")
        )

        region   = st.multiselect("", data['Region'].unique(),   placeholder="All regions")
        st.markdown('<div style="font-size:0.72rem;color:#6B7FA8;text-transform:uppercase;letter-spacing:0.06em;margin:0.8rem 0 0.4rem;">Industry</div>', unsafe_allow_html=True)
        industry = st.multiselect("", data['Industry'].unique(), placeholder="All industries")
        st.markdown('<div style="font-size:0.72rem;color:#6B7FA8;text-transform:uppercase;letter-spacing:0.06em;margin:0.8rem 0 0.4rem;">Risk Category</div>', unsafe_allow_html=True)
        risk_filter = st.multiselect("", data['Risk_Category'].unique(), placeholder="All categories")

        filtered = data.copy()
        if region:      filtered = filtered[filtered['Region'].isin(region)]
        if industry:    filtered = filtered[filtered['Industry'].isin(industry)]
        if risk_filter: filtered = filtered[filtered['Risk_Category'].isin(risk_filter)]

        data_loaded = True
    except Exception as e:
        st.error(f"Could not load CSV: {e}")
        data_loaded = False

    st.markdown('<hr style="border:none;height:1px;background:rgba(99,179,255,0.12);margin:1.5rem 0;">', unsafe_allow_html=True)

    # Team in sidebar
    if st.button("👥 View Team"):
        st.markdown("""
        <div class="team-grid">
          <div class="team-card">
            <div class="team-avatar">M</div>
            <div><div class="team-name">Mohnish Singh Patwal</div><div class="team-role">Member</div></div>
          </div>
          <div class="team-card">
            <div class="team-avatar">S</div>
            <div><div class="team-name">Shreyas Kandi</div><div class="team-role">Member</div></div>
          </div>
          <div class="team-card">
            <div class="team-avatar">A</div>
            <div><div class="team-name">Akash Krishna</div><div class="team-role">Member</div></div>
          </div>
          <div class="team-card">
            <div class="team-avatar">N</div>
            <div><div class="team-name">Nihal Talampally</div><div class="team-role">Member</div></div>
          </div>
        </div>
        <div style="font-size:0.7rem;color:#6B7FA8;margin-top:0.8rem;text-align:center;">Section A · BBA Semester 4</div>
        """, unsafe_allow_html=True)

# ===== MAIN CONTENT (only if data loaded) =====
if not data_loaded:
    st.error("Please ensure **B2B_Client_Churn_5000.csv** is in the same folder.")
    st.stop()

# ── KPI Metrics ──
total_clients = len(filtered)
high_risk     = (filtered['Risk_Category'] == "High Risk").sum()
churn_rate    = round((1 - filtered['Renewal_Status'].mean()) * 100, 1)
avg_revenue   = round(filtered['Monthly_Revenue_USD'].mean(), 0)

st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi-card kpi-blue">
    <div class="kpi-icon">👥</div>
    <div class="kpi-value">{total_clients:,}</div>
    <div class="kpi-label">Total Clients</div>
  </div>
  <div class="kpi-card kpi-danger">
    <div class="kpi-icon">⚠️</div>
    <div class="kpi-value">{high_risk:,}</div>
    <div class="kpi-label">High Risk Clients</div>
  </div>
  <div class="kpi-card kpi-warn">
    <div class="kpi-icon">📉</div>
    <div class="kpi-value">{churn_rate}%</div>
    <div class="kpi-label">Churn Rate</div>
  </div>
  <div class="kpi-card kpi-cyan">
    <div class="kpi-icon">💰</div>
    <div class="kpi-value">${avg_revenue:,.0f}</div>
    <div class="kpi-label">Avg Monthly Revenue</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Churn Gauge ──
gauge_class = "gauge-low" if churn_rate < 20 else ("gauge-medium" if churn_rate < 40 else "gauge-high")
chip_class  = "chip-ok"   if churn_rate < 20 else ("chip-warn"   if churn_rate < 40 else "chip-danger")
chip_text   = ("🟢 Low Risk" if churn_rate < 20 else ("🟠 Moderate Risk" if churn_rate < 40 else "🔴 High Risk"))

st.markdown(f"""
<div class="panel">
  <div class="section-title">Overall Churn Risk Gauge</div>
  <div class="section-subtitle">Percentage of clients not renewing across selected filters</div>
  <div class="gauge-track">
    <div class="gauge-fill {gauge_class}" style="width:{min(churn_rate,100)}%;"></div>
  </div>
  <div style="display:flex;align-items:center;justify-content:space-between;margin-top:0.5rem;">
    <span class="chip {chip_class}">{chip_text}</span>
    <span style="font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;color:var(--text-primary);">{churn_rate}%</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Charts Row 1 ──
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Risk Distribution</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Client count by risk category</div>', unsafe_allow_html=True)

    risk_counts = filtered['Risk_Category'].value_counts()
    colors_map  = {"Low Risk": OK, "Medium Risk": WARN, "High Risk": DANGER}
    bar_colors  = [colors_map.get(c, BLUE) for c in risk_counts.index]

    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.bar(risk_counts.index, risk_counts.values, color=bar_colors,
                  edgecolor='none', width=0.5, zorder=2)
    for bar, val in zip(bars, risk_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                f'{val:,}', ha='center', va='bottom', fontsize=9, color='#E8F0FF', fontweight='600')
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_ylabel('Clients', fontsize=9)
    fig.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Industry-wise Avg Risk Score</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Higher score = greater churn exposure</div>', unsafe_allow_html=True)

    ind_risk = filtered.groupby('Industry')['Risk_Score'].mean().sort_values()
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    h_bars = ax2.barh(ind_risk.index, ind_risk.values, color=BLUE,
                      edgecolor='none', height=0.55, zorder=2)
    for bar, val in zip(h_bars, ind_risk.values):
        ax2.text(val + 0.03, bar.get_y() + bar.get_height()/2,
                 f'{val:.1f}', va='center', fontsize=8.5, color='#E8F0FF', fontweight='600')
    ax2.set_axisbelow(True)
    ax2.xaxis.grid(True, alpha=0.3)
    ax2.set_xlabel('Avg Risk Score', fontsize=9)
    fig2.tight_layout(pad=0.5)
    st.pyplot(fig2, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Charts Row 2 ──
col3, col4 = st.columns(2)

with col3:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Revenue vs Risk Score</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Scatter — identify high-value at-risk clients</div>', unsafe_allow_html=True)

    sample = filtered.sample(min(600, len(filtered)), random_state=42)
    color_scatter = sample['Risk_Category'].map(
        {"Low Risk": OK, "Medium Risk": WARN, "High Risk": DANGER}
    )
    fig3, ax3 = plt.subplots(figsize=(5, 3))
    ax3.scatter(sample['Monthly_Revenue_USD'], sample['Risk_Score'],
                c=color_scatter, alpha=0.65, s=20, edgecolors='none', zorder=2)
    ax3.set_axisbelow(True)
    ax3.yaxis.grid(True, alpha=0.3)
    ax3.xaxis.grid(True, alpha=0.3)
    ax3.set_xlabel('Monthly Revenue (USD)', fontsize=9)
    ax3.set_ylabel('Risk Score', fontsize=9)
    patches = [mpatches.Patch(color=OK, label='Low'),
               mpatches.Patch(color=WARN, label='Medium'),
               mpatches.Patch(color=DANGER, label='High')]
    ax3.legend(handles=patches, fontsize=8, framealpha=0.1, labelcolor='#E8F0FF',
               facecolor='#0E1520', edgecolor='#1E2D44')
    fig3.tight_layout(pad=0.5)
    st.pyplot(fig3, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Churn Rate by Industry</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Proportion of non-renewals per industry</div>', unsafe_allow_html=True)

    churn_by_ind = (1 - data.groupby('Industry')['Renewal_Status'].mean()).sort_values()
    fig4, ax4 = plt.subplots(figsize=(5, 3))
    bar_cols4 = [DANGER if v > 0.4 else (WARN if v > 0.25 else OK) for v in churn_by_ind.values]
    h4 = ax4.barh(churn_by_ind.index, churn_by_ind.values * 100,
                  color=bar_cols4, edgecolor='none', height=0.55, zorder=2)
    for bar, val in zip(h4, churn_by_ind.values):
        ax4.text(val*100 + 0.3, bar.get_y() + bar.get_height()/2,
                 f'{val*100:.1f}%', va='center', fontsize=8.5, color='#E8F0FF', fontweight='600')
    ax4.set_axisbelow(True)
    ax4.xaxis.grid(True, alpha=0.3)
    ax4.set_xlabel('Churn Rate (%)', fontsize=9)
    fig4.tight_layout(pad=0.5)
    st.pyplot(fig4, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── ML Model Section ──
st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
st.markdown("""
<div class="section-title" style="font-size:1.3rem;margin-bottom:0.2rem;">🤖 Churn Prediction Model</div>
<div class="section-subtitle" style="margin-bottom:1rem;">Random Forest Classifier trained on full dataset</div>
""", unsafe_allow_html=True)

features = [
    'Monthly_Usage_Score',
    'Payment_Delay_Days',
    'Contract_Length_Months',
    'Support_Tickets_Last30Days',
    'Monthly_Revenue_USD'
]
X = data[features]
y = data['Renewal_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
pred    = model.predict(X_test)
acc     = round(accuracy_score(y_test, pred) * 100, 1)
cm      = confusion_matrix(y_test, pred)

importance = pd.DataFrame({
    'Feature':    features,
    'Importance': model.feature_importances_
}).sort_values('Importance')

col5, col6 = st.columns([1, 1.6])

with col5:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Model Accuracy</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="accuracy-display">
      <div>
        <div class="accuracy-number">{acc}%</div>
        <div class="accuracy-label">Test Set Accuracy</div>
      </div>
      <div class="chip chip-ok" style="margin-top:0.5rem;">✓ High Confidence</div>
    </div>
    """, unsafe_allow_html=True)

    # Confusion Matrix
    fig5, ax5 = plt.subplots(figsize=(3.2, 2.5))
    im = ax5.imshow(cm, cmap='Blues', aspect='auto')
    ax5.set_xticks([0,1]); ax5.set_yticks([0,1])
    ax5.set_xticklabels(['Pred: Churn','Pred: Renew'], fontsize=8)
    ax5.set_yticklabels(['Act: Churn','Act: Renew'],  fontsize=8)
    for i in range(2):
        for j in range(2):
            ax5.text(j, i, str(cm[i,j]), ha='center', va='center',
                     fontsize=12, fontweight='700',
                     color='white' if cm[i,j] > cm.max()/2 else '#E8F0FF')
    ax5.set_title('Confusion Matrix', fontsize=9, color='#6B7FA8')
    fig5.tight_layout(pad=0.3)
    st.pyplot(fig5, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col6:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Feature Importance</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Which signals drive churn prediction most</div>', unsafe_allow_html=True)

    imp_colors = [CYAN if v == importance['Importance'].max() else BLUE for v in importance['Importance']]
    fig6, ax6 = plt.subplots(figsize=(5.5, 3.2))
    h6 = ax6.barh(importance['Feature'], importance['Importance'],
                  color=imp_colors, edgecolor='none', height=0.55, zorder=2)
    for bar, val in zip(h6, importance['Importance']):
        ax6.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center', fontsize=8.5, color='#E8F0FF', fontweight='600')
    ax6.set_axisbelow(True)
    ax6.xaxis.grid(True, alpha=0.3)
    ax6.set_xlabel('Importance Score', fontsize=9)
    fig6.tight_layout(pad=0.5)
    st.pyplot(fig6, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Payment Delay Analysis ──
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Payment Delay vs Churn Probability</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">How delay in days correlates with churn rate across the full dataset</div>', unsafe_allow_html=True)

delay_churn = (1 - data.groupby('Payment_Delay_Days')['Renewal_Status'].mean()) * 100
fig7, ax7 = plt.subplots(figsize=(10, 3))
ax7.plot(delay_churn.index, delay_churn.values, color=CYAN, linewidth=2, zorder=3)
ax7.fill_between(delay_churn.index, delay_churn.values, alpha=0.12, color=CYAN, zorder=2)
ax7.axhline(y=40, color=DANGER, linestyle='--', alpha=0.5, linewidth=1, label='40% threshold')
ax7.set_axisbelow(True)
ax7.yaxis.grid(True, alpha=0.3)
ax7.set_xlabel('Payment Delay (Days)', fontsize=9)
ax7.set_ylabel('Churn Rate (%)', fontsize=9)
ax7.legend(fontsize=8, framealpha=0.1, labelcolor='#E8F0FF', facecolor='#0E1520', edgecolor='#1E2D44')
fig7.tight_layout(pad=0.5)
st.pyplot(fig7, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ── High Risk Table ──
st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
st.markdown("""
<div class="section-title" style="font-size:1.2rem;">🚨 High-Revenue Clients at Risk</div>
<div class="section-subtitle" style="margin-bottom:0.8rem;">Top 10 clients by revenue who are classified as High Risk</div>
""", unsafe_allow_html=True)

display_cols = ['Monthly_Revenue_USD', 'Risk_Score', 'Risk_Category',
                'Payment_Delay_Days', 'Support_Tickets_Last30Days']
available_cols = [c for c in display_cols if c in filtered.columns]

high_value_risk = filtered[filtered['Risk_Category'] == 'High Risk'].sort_values(
    'Monthly_Revenue_USD', ascending=False
)
if not high_value_risk.empty:
    st.dataframe(
        high_value_risk[available_cols].head(10).reset_index(drop=True),
        use_container_width=True,
        height=300
    )
else:
    st.info("No High Risk clients in current filter.")

# ── Retention Strategies ──
st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
if st.button("⚡ Generate Retention Strategy"):
    st.markdown("""
    <div class="panel">
      <div class="section-title" style="margin-bottom:1rem;">Recommended Retention Actions</div>
      <div class="strategy-item"><span class="strategy-icon">💳</span><span>Offer structured payment plans or early-pay discounts to clients with payment delays exceeding 30 days — reducing financial friction improves renewal intent.</span></div>
      <div class="strategy-item"><span class="strategy-icon">🤝</span><span>Assign dedicated account managers to clients with 5+ support tickets in the last 30 days to provide white-glove issue resolution.</span></div>
      <div class="strategy-item"><span class="strategy-icon">📋</span><span>Provide multi-year contract incentives (e.g., 10–15% discount) to clients on short-term contracts to increase stickiness and lifetime value.</span></div>
      <div class="strategy-item"><span class="strategy-icon">⚡</span><span>Implement SLA-backed support response time guarantees for Tier-1 high-revenue clients to reduce churn driven by service dissatisfaction.</span></div>
      <div class="strategy-item"><span class="strategy-icon">📈</span><span>Launch onboarding refreshers and product training for clients with usage scores below 50 to drive platform adoption and reduce disengagement.</span></div>
    </div>
    """, unsafe_allow_html=True)

# ── Ethical AI ──
st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
st.markdown("""
<div class="section-title" style="font-size:1.2rem;">⚖️ Ethical AI Considerations</div>
<div class="section-subtitle">Responsible deployment principles for this prediction system</div>
<div class="ethics-grid">
  <div class="ethics-item">📊 <strong>Data Bias</strong><br>Predictive models can inherit historical biases. Regular audits are essential to ensure fair treatment across regions and industries.</div>
  <div class="ethics-item">🔒 <strong>Client Privacy</strong><br>All client data must be handled under strict data governance frameworks and compliant with applicable privacy regulations.</div>
  <div class="ethics-item">🤖 <strong>Human Oversight</strong><br>AI predictions should augment human decision-making, not replace it. Final retention decisions require human judgment.</div>
  <div class="ethics-item">⚠️ <strong>Fairness in Labeling</strong><br>Labeling clients as high-risk can affect business relationships. Risk scores must be communicated with appropriate context.</div>
</div>
""", unsafe_allow_html=True)

# ── Footer ──
st.markdown("""
<hr class="custom-divider">
<div style="text-align:center;padding:0.5rem 0 1rem;">
  <div style="font-size:0.7rem;color:#6B7FA8;letter-spacing:0.06em;text-transform:uppercase;">
    Developed by Group-2 · Rhinos · BBA Semester 4 · Section A · Woxsen University
  </div>
  <div style="margin-top:0.4rem;font-size:0.65rem;color:#3B4A6B;">
    Powered by Random Forest ML · Streamlit · scikit-learn
  </div>
</div>
""", unsafe_allow_html=True)
