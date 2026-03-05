import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier          # Part C: Decision Tree
from sklearn.metrics import accuracy_score, confusion_matrix

# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="B2B Client Risk Intelligence | Woxsen University",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

:root {
    --bg:           #060B14;
    --surface:      #0C1422;
    --surface2:     #101C2E;
    --border:       rgba(56,139,253,0.14);
    --border-light: rgba(56,139,253,0.08);
    --blue:   #4F8EF7;
    --cyan:   #22D3EE;
    --green:  #34D399;
    --amber:  #FBBF24;
    --red:    #F87171;
    --purple: #A78BFA;
    --text-primary:   #E2EAF4;
    --text-secondary: #8FA3C0;
    --text-muted:     #4E6180;
}

html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text-primary) !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 3rem; max-width: 1380px; }
::-webkit-scrollbar { width: 6px; background: var(--bg); }
::-webkit-scrollbar-thumb { background: #1E3050; border-radius: 3px; }

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
[data-testid="stSidebar"] .stMultiSelect span {
    background: rgba(79,142,247,0.18) !important;
    color: var(--blue) !important;
    border-radius: 4px !important;
}

.header-band {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 1rem 0 1.2rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.header-left  { display: flex; align-items: center; gap: 1.2rem; }
.header-logo  { height: 44px; width: auto; }
.header-div   { width: 1px; height: 38px; background: var(--border); }
.header-title {
    font-family: 'Manrope', sans-serif;
    font-size: 1.2rem; font-weight: 800;
    color: var(--text-primary); letter-spacing: -0.02em;
}
.header-sub {
    font-size: 0.68rem; color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 0.06em; margin-top: 0.2rem;
}
.header-right { display: flex; gap: 0.5rem; flex-wrap: wrap; align-items: center; }
.tag {
    font-size: 0.66rem; font-weight: 600; letter-spacing: 0.06em;
    text-transform: uppercase; padding: 0.26rem 0.65rem;
    border-radius: 6px; border: 1px solid var(--border);
    color: var(--text-secondary); background: var(--surface2);
}

.sec-hdr {
    font-family: 'Manrope', sans-serif;
    font-size: 0.72rem; font-weight: 700;
    color: var(--text-muted); letter-spacing: 0.1em;
    text-transform: uppercase;
    margin: 1.6rem 0 0.7rem;
    display: flex; align-items: center; gap: 0.5rem;
}
.sec-hdr::after { content:''; flex:1; height:1px; background: var(--border-light); }

.panel-title {
    font-family: 'Manrope', sans-serif;
    font-size: 0.9rem; font-weight: 700;
    color: var(--text-primary); letter-spacing: -0.01em; margin-bottom: 0.12rem;
}
.panel-sub { font-size: 0.7rem; color: var(--text-muted); margin-bottom: 0.85rem; }

.kpi-row { display: grid; grid-template-columns: repeat(4,1fr); gap: 0.85rem; margin-bottom: 1rem; }
.kpi {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 1.05rem 1.25rem 1rem;
    position: relative; overflow: hidden;
}
.kpi::before { content:''; position:absolute; top:0;left:0;right:0;height:2px; }
.kpi-blue::before  { background: linear-gradient(90deg, var(--blue), var(--cyan)); }
.kpi-red::before   { background: linear-gradient(90deg, var(--red),  #FF8C94); }
.kpi-amber::before { background: linear-gradient(90deg, var(--amber), #FB923C); }
.kpi-green::before { background: linear-gradient(90deg, var(--green), var(--cyan)); }
.kpi-icon { font-size: 1.2rem; margin-bottom: 0.5rem; }
.kpi-num {
    font-family: 'Manrope', sans-serif; font-size: 1.85rem; font-weight: 800;
    letter-spacing: -0.04em; line-height: 1; margin-bottom: 0.28rem;
}
.kpi-blue  .kpi-num { color: var(--blue); }
.kpi-red   .kpi-num { color: var(--red); }
.kpi-amber .kpi-num { color: var(--amber); }
.kpi-green .kpi-num { color: var(--green); }
.kpi-lbl { font-size: 0.66rem; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.07em; }

.gauge-wrap {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 1rem 1.25rem 1.1rem; margin-bottom: 0.85rem;
}
.gauge-top { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:0.45rem; }
.gauge-lbl { font-family:'Manrope',sans-serif; font-size:0.85rem; font-weight:700; color:var(--text-primary); }
.gauge-pct { font-family:'Manrope',sans-serif; font-size:1.45rem; font-weight:800; letter-spacing:-0.03em; }
.gauge-track { background:var(--surface2); border-radius:99px; height:7px; overflow:hidden; margin-bottom:0.5rem; }
.gauge-fill  { height:100%; border-radius:99px; }
.g-low  { background:linear-gradient(90deg,var(--green),#6EE7B7); color:var(--green); }
.g-mid  { background:linear-gradient(90deg,var(--amber),#FB923C); color:var(--amber); }
.g-high { background:linear-gradient(90deg,var(--red),  #FF8C94); color:var(--red); }

.pill {
    display:inline-flex; align-items:center; gap:0.3rem;
    font-size:0.7rem; font-weight:600; letter-spacing:0.04em;
    padding:0.28rem 0.7rem; border-radius:99px;
}
.p-low  { background:rgba(52,211,153,0.1); color:var(--green); border:1px solid rgba(52,211,153,0.25); }
.p-mid  { background:rgba(251,191,36,0.1); color:var(--amber); border:1px solid rgba(251,191,36,0.25); }
.p-high { background:rgba(248,113,113,0.1);color:var(--red);   border:1px solid rgba(248,113,113,0.25); }
.p-ok   { background:rgba(52,211,153,0.1); color:var(--green); border:1px solid rgba(52,211,153,0.25); }

.chart-panel {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 1.05rem 1.25rem 0.85rem; margin-bottom: 0.85rem;
}

.stButton > button {
    background: var(--blue) !important; color: #fff !important;
    border: none !important; border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important; font-size: 0.79rem !important;
    font-weight: 600 !important; letter-spacing: 0.04em !important;
    padding: 0.52rem 1.25rem !important;
    box-shadow: 0 2px 10px rgba(79,142,247,0.28) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #3B7AE5 !important;
    box-shadow: 0 4px 16px rgba(79,142,247,0.42) !important;
    transform: translateY(-1px) !important;
}

[data-testid="stDataFrame"] {
    border-radius: 10px !important; overflow: hidden !important;
    border: 1px solid var(--border) !important;
}

.reten-item {
    display:flex; gap:0.75rem; align-items:flex-start;
    background:var(--surface2); border:1px solid var(--border);
    border-left:3px solid var(--blue);
    border-radius:8px; padding:0.72rem 1rem;
    margin-bottom:0.48rem; font-size:0.8rem;
    color:var(--text-primary); line-height:1.55;
}
.reten-ico { font-size:1rem; flex-shrink:0; margin-top:0.05rem; }

.ethics-grid { display:grid; grid-template-columns:1fr 1fr; gap:0.65rem; margin-top:0.65rem; }
.ethics-card {
    background:var(--surface2); border:1px solid var(--border);
    border-top:2px solid var(--cyan); border-radius:8px;
    padding:0.8rem 1rem; font-size:0.78rem;
    color:var(--text-primary); line-height:1.55;
}
.ethics-card strong { color:var(--cyan); font-weight:600; }

.acc-block { display:flex; align-items:baseline; gap:0.45rem; margin:0.45rem 0 0.75rem; }
.acc-num {
    font-family:'Manrope',sans-serif; font-size:2.6rem; font-weight:800;
    color:var(--cyan); letter-spacing:-0.05em; line-height:1;
}
.acc-sub { font-size:0.68rem; font-weight:600; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.06em; }

.sb-lbl { font-size:0.66rem; font-weight:700; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.3rem; }

.team-grid { display:grid; grid-template-columns:1fr 1fr; gap:0.55rem; margin-top:0.65rem; }
.team-card {
    background:var(--surface2); border:1px solid var(--border);
    border-radius:8px; padding:0.6rem 0.75rem;
    display:flex; align-items:center; gap:0.55rem;
}
.t-av {
    width:30px; height:30px; border-radius:50%; flex-shrink:0;
    background:linear-gradient(135deg,var(--blue),var(--cyan));
    display:flex; align-items:center; justify-content:center;
    font-size:0.75rem; font-weight:700; color:#fff;
}
.t-name { font-size:0.73rem; font-weight:600; color:var(--text-primary); }
.t-role { font-size:0.6rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.04em; }

.footer {
    text-align:center; padding:0.9rem 0 0.5rem;
    font-size:0.64rem; color:var(--text-muted);
    letter-spacing:0.05em; text-transform:uppercase;
    border-top:1px solid var(--border-light); margin-top:1.5rem;
}
.div { height:1px; background:var(--border-light); margin:1.2rem 0; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  MATPLOTLIB THEME
# ══════════════════════════════════════════════════════════════
BG    = '#0C1422'
GRID  = '#1A2840'
plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': BG,
    'axes.edgecolor': GRID, 'axes.labelcolor': '#8FA3C0',
    'axes.titlecolor': '#E2EAF4',
    'xtick.color': '#8FA3C0', 'ytick.color': '#8FA3C0',
    'grid.color': GRID, 'grid.alpha': 0.7,
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.spines.left': False, 'axes.spines.bottom': False,
})
CB, CC, CG, CA, CR, CP = '#4F8EF7','#22D3EE','#34D399','#FBBF24','#F87171','#A78BFA'

# ══════════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════════
try:
    data = pd.read_csv("B2B_Client_Churn_5000.csv")
    data.columns = data.columns.str.strip().str.replace(" ", "_")
    data['Renewal_Status'] = data['Renewal_Status'].map({'Yes':1,'No':0})

    def calc_risk(row):
        r = 0
        if row['Payment_Delay_Days'] > 30:        r += 2
        if row['Monthly_Usage_Score'] < 50:        r += 2
        if row['Contract_Length_Months'] < 12:     r += 2
        if row['Support_Tickets_Last30Days'] > 5:  r += 2
        return r

    data['Risk_Score']    = data.apply(calc_risk, axis=1)
    data['Risk_Category'] = data['Risk_Score'].apply(
        lambda s: "Low Risk" if s <= 2 else ("Medium Risk" if s <= 5 else "High Risk")
    )
    data_ok = True
except Exception as e:
    data_ok, data_err = False, str(e)

# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:0.8rem 0 1.2rem;">
        <img src="https://woxsen.edu.in/wp-content/uploads/2022/04/woxsen-logo-new.png"
             style="height:38px;width:auto;filter:brightness(1.1);"
             onerror="this.style.display='none'">
        <div style="font-size:0.62rem;color:#4E6180;text-transform:uppercase;
                    letter-spacing:0.08em;margin-top:0.5rem;">Dashboard Filters</div>
    </div>
    """, unsafe_allow_html=True)

    if data_ok:
        st.markdown('<div class="sb-lbl">Region</div>', unsafe_allow_html=True)
        region = st.multiselect("", data['Region'].unique(), placeholder="All regions", label_visibility="collapsed")
        st.markdown('<div class="sb-lbl" style="margin-top:0.8rem;">Industry</div>', unsafe_allow_html=True)
        industry = st.multiselect("", data['Industry'].unique(), placeholder="All industries", label_visibility="collapsed")
        st.markdown('<div class="sb-lbl" style="margin-top:0.8rem;">Risk Category</div>', unsafe_allow_html=True)
        risk_filter = st.multiselect("", data['Risk_Category'].unique(), placeholder="All categories", label_visibility="collapsed")

    st.markdown('<div style="height:1px;background:rgba(56,139,253,0.1);margin:1.2rem 0;"></div>', unsafe_allow_html=True)

    if st.button("👥 Team Members"):
        st.markdown("""
        <div class="team-grid">
          <div class="team-card"><div class="t-av">M</div><div><div class="t-name">Mohnish Singh Patwal</div><div class="t-role">Member</div></div></div>
          <div class="team-card"><div class="t-av">S</div><div><div class="t-name">Shreyas Kandi</div><div class="t-role">Member</div></div></div>
          <div class="team-card"><div class="t-av">A</div><div><div class="t-name">Akash Krishna</div><div class="t-role">Member</div></div></div>
          <div class="team-card"><div class="t-av">N</div><div><div class="t-name">Nihal Talampally</div><div class="t-role">Member</div></div></div>
        </div>
        <div style="font-size:0.62rem;color:#4E6180;text-align:center;margin-top:0.7rem;">
            Section A · BBA Semester 4 · Woxsen University
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="header-band">
  <div class="header-left">
    <img class="header-logo"
         src="https://woxsen.edu.in/wp-content/uploads/2022/04/woxsen-logo-new.png"
         onerror="this.src='';this.style.display='none';">
    <div class="header-div"></div>
    <div>
      <div class="header-title">B2B Client Risk Intelligence Dashboard</div>
      <div class="header-sub">AI-powered churn prediction &amp; retention analytics</div>
    </div>
  </div>
  <div class="header-right">
    <span class="tag">Group-2 · Rhinos</span>
    <span class="tag">BBA Sem 4 · Section A</span>
    <span class="tag">Woxsen University</span>
  </div>
</div>
""", unsafe_allow_html=True)

if not data_ok:
    st.error(f"Could not load CSV: {data_err}. Place B2B_Client_Churn_5000.csv in the same folder.")
    st.stop()

# ── Filters ─────────────────────────────────────────────────
filtered = data.copy()
if region:      filtered = filtered[filtered['Region'].isin(region)]
if industry:    filtered = filtered[filtered['Industry'].isin(industry)]
if risk_filter: filtered = filtered[filtered['Risk_Category'].isin(risk_filter)]

# ══════════════════════════════════════════════════════════════
#  KPI CARDS
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="sec-hdr">Key Performance Indicators</div>', unsafe_allow_html=True)

total   = len(filtered)
hi_risk = (filtered['Risk_Category'] == "High Risk").sum()
churn   = round((1 - filtered['Renewal_Status'].mean()) * 100, 1)
avg_rev = round(filtered['Monthly_Revenue_USD'].mean(), 0)

st.markdown(f"""
<div class="kpi-row">
  <div class="kpi kpi-blue"><div class="kpi-icon">👥</div>
    <div class="kpi-num">{total:,}</div><div class="kpi-lbl">Total Clients</div></div>
  <div class="kpi kpi-red"><div class="kpi-icon">⚠️</div>
    <div class="kpi-num">{hi_risk:,}</div><div class="kpi-lbl">High Risk Clients</div></div>
  <div class="kpi kpi-amber"><div class="kpi-icon">📉</div>
    <div class="kpi-num">{churn}%</div><div class="kpi-lbl">Predicted Churn Rate</div></div>
  <div class="kpi kpi-green"><div class="kpi-icon">💰</div>
    <div class="kpi-num">${avg_rev:,.0f}</div><div class="kpi-lbl">Avg Revenue / Client</div></div>
</div>
""", unsafe_allow_html=True)

# ── Gauge ────────────────────────────────────────────────────
g  = "g-low"  if churn < 20 else ("g-mid"  if churn < 40 else "g-high")
sp = "p-low"  if churn < 20 else ("p-mid"  if churn < 40 else "p-high")
st_txt = ("● Low Churn Risk" if churn < 20 else ("● Moderate Churn Risk" if churn < 40 else "● High Churn Risk"))

st.markdown(f"""
<div class="gauge-wrap">
  <div class="gauge-top">
    <span class="gauge-lbl">Overall Churn Risk Gauge</span>
    <span class="gauge-pct {g}">{churn}%</span>
  </div>
  <div class="gauge-track"><div class="gauge-fill {g}" style="width:{min(churn,100)}%;"></div></div>
  <span class="pill {sp}">{st_txt}</span>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  VISUALISATIONS
# ══════════════════════════════════════════════════════════════
def sfig(w=5, h=3.2):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    return fig, ax

st.markdown('<div class="sec-hdr">Risk Distribution &amp; Industry Analysis</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)

# Chart 1 — Risk Distribution
with c1:
    st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Risk Category Distribution</div><div class="panel-sub">Client count segmented by risk level</div>', unsafe_allow_html=True)
    rc = filtered['Risk_Category'].value_counts().reindex(["Low Risk","Medium Risk","High Risk"]).fillna(0)
    fig, ax = sfig()
    bars = ax.bar(rc.index, rc.values, color=[CG,CA,CR], edgecolor='none', width=0.45, zorder=2)
    for b, v in zip(bars, rc.values):
        ax.text(b.get_x()+b.get_width()/2, v+18, f'{int(v):,}', ha='center', fontsize=9, fontweight='600', color='#E2EAF4')
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0); ax.set_axisbelow(True)
    ax.set_ylabel('Clients', fontsize=8.5, color='#8FA3C0'); ax.tick_params(labelsize=8.5)
    fig.tight_layout(pad=0.4); st.pyplot(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Chart 2 — Industry Risk
with c2:
    st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Industry-wise Risk Analysis</div><div class="panel-sub">Average risk score per industry segment</div>', unsafe_allow_html=True)
    ir = filtered.groupby('Industry')['Risk_Score'].mean().sort_values()
    fig, ax = sfig()
    h = ax.barh(ir.index, ir.values, color=CB, edgecolor='none', height=0.52, zorder=2)
    for bar, v in zip(h, ir.values):
        ax.text(v+0.04, bar.get_y()+bar.get_height()/2, f'{v:.1f}', va='center', fontsize=8, fontweight='600', color='#E2EAF4')
    ax.xaxis.grid(True, color=GRID, linewidth=0.6, zorder=0); ax.set_axisbelow(True)
    ax.set_xlabel('Avg Risk Score', fontsize=8.5, color='#8FA3C0'); ax.tick_params(labelsize=8)
    fig.tight_layout(pad=0.4); st.pyplot(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

c3, c4 = st.columns(2)

# Chart 3 — Revenue vs Risk Scatter
with c3:
    st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Revenue vs Risk Score</div><div class="panel-sub">Identifying high-revenue clients at churn risk</div>', unsafe_allow_html=True)
    samp = filtered.sample(min(600, len(filtered)), random_state=42)
    sc = samp['Risk_Category'].map({"Low Risk":CG,"Medium Risk":CA,"High Risk":CR})
    fig, ax = sfig()
    ax.scatter(samp['Monthly_Revenue_USD'], samp['Risk_Score'], c=sc, alpha=0.6, s=17, edgecolors='none', zorder=2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel('Monthly Revenue (USD)', fontsize=8.5, color='#8FA3C0')
    ax.set_ylabel('Risk Score', fontsize=8.5, color='#8FA3C0')
    patches = [mpatches.Patch(color=CG,label='Low'), mpatches.Patch(color=CA,label='Medium'), mpatches.Patch(color=CR,label='High')]
    ax.legend(handles=patches, fontsize=8, framealpha=0.15, facecolor=BG, edgecolor=GRID, labelcolor='#E2EAF4')
    fig.tight_layout(pad=0.4); st.pyplot(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Chart 4 — Contract Length vs Churn  ← PART D requirement
with c4:
    st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Contract Length vs Churn Rate</div><div class="panel-sub">How contract duration relates to renewal behaviour</div>', unsafe_allow_html=True)
    cl = filtered.groupby('Contract_Length_Months')['Renewal_Status'].mean()
    cl = (1 - cl) * 100
    fig, ax = sfig()
    ax.fill_between(cl.index, cl.values, alpha=0.14, color=CP, zorder=1)
    ax.plot(cl.index, cl.values, color=CP, linewidth=2, zorder=3)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0); ax.set_axisbelow(True)
    ax.set_xlabel('Contract Length (Months)', fontsize=8.5, color='#8FA3C0')
    ax.set_ylabel('Churn Rate (%)', fontsize=8.5, color='#8FA3C0')
    fig.tight_layout(pad=0.4); st.pyplot(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Chart 5 — Churn by Industry (full width)
st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
st.markdown('<div class="panel-title">Churn Rate by Industry</div><div class="panel-sub">Percentage of non-renewals across all industry verticals</div>', unsafe_allow_html=True)
ci = (1 - data.groupby('Industry')['Renewal_Status'].mean()).sort_values() * 100
fig, ax = sfig(10, 3.2)
bc = [CR if v>40 else (CA if v>25 else CG) for v in ci.values]
h_ci = ax.barh(ci.index, ci.values, color=bc, edgecolor='none', height=0.52, zorder=2)
for bar, v in zip(h_ci, ci.values):
    ax.text(v+0.3, bar.get_y()+bar.get_height()/2, f'{v:.1f}%', va='center', fontsize=8, fontweight='600', color='#E2EAF4')
ax.xaxis.grid(True, color=GRID, linewidth=0.6, zorder=0); ax.set_axisbelow(True)
ax.set_xlabel('Churn Rate (%)', fontsize=8.5, color='#8FA3C0')
fig.tight_layout(pad=0.4); st.pyplot(fig, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# Chart 6 — Payment Delay vs Churn (full width)
st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
st.markdown('<div class="panel-title">Payment Delay vs Churn Probability</div><div class="panel-sub">How increasing payment delays elevate churn likelihood</div>', unsafe_allow_html=True)
dc = (1 - data.groupby('Payment_Delay_Days')['Renewal_Status'].mean()) * 100
fig, ax = sfig(10, 3)
ax.plot(dc.index, dc.values, color=CC, linewidth=2, zorder=3)
ax.fill_between(dc.index, dc.values, alpha=0.1, color=CC, zorder=2)
ax.axhline(40, color=CR, linestyle='--', linewidth=1, alpha=0.5, label='40% threshold')
ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0); ax.set_axisbelow(True)
ax.set_xlabel('Payment Delay (Days)', fontsize=8.5, color='#8FA3C0')
ax.set_ylabel('Churn Rate (%)', fontsize=8.5, color='#8FA3C0')
ax.legend(fontsize=8, framealpha=0.15, facecolor=BG, edgecolor=GRID, labelcolor='#E2EAF4')
fig.tight_layout(pad=0.4); st.pyplot(fig, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  PART C — DECISION TREE MODEL
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="sec-hdr">Part C — Machine Learning: Decision Tree Classifier</div>', unsafe_allow_html=True)

FEATURES = ['Monthly_Usage_Score','Payment_Delay_Days','Contract_Length_Months',
            'Support_Tickets_Last30Days','Monthly_Revenue_USD']
X = data[FEATURES]; y = data['Renewal_Status']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
dt = DecisionTreeClassifier(max_depth=6, random_state=42)
dt.fit(X_tr, y_tr)
pred = dt.predict(X_te)
acc  = round(accuracy_score(y_te, pred)*100, 1)
cm   = confusion_matrix(y_te, pred)
imp  = pd.DataFrame({'Feature':FEATURES, 'Importance':dt.feature_importances_}).sort_values('Importance')

c5, c6 = st.columns([1, 1.6])

with c5:
    st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Model Performance</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="acc-block">
      <div class="acc-num">{acc}%</div>
      <div class="acc-sub">Test Accuracy</div>
    </div>
    <span class="pill p-ok">✓ Decision Tree Classifier · depth=6</span>
    """, unsafe_allow_html=True)
    fig, ax = sfig(3.4, 2.8)
    ax.imshow(cm, cmap='Blues', aspect='auto')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Pred: Churn','Pred: Renew'], fontsize=8, color='#8FA3C0')
    ax.set_yticklabels(['Act: Churn','Act: Renew'],   fontsize=8, color='#8FA3C0')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i,j]:,}', ha='center', va='center', fontsize=12, fontweight='700',
                    color='white' if cm[i,j]>cm.max()/2 else '#E2EAF4')
    ax.set_title('Confusion Matrix', fontsize=8.5, color='#8FA3C0', pad=6)
    fig.tight_layout(pad=0.3); st.pyplot(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c6:
    st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Feature Importance</div><div class="panel-sub">Factors that most influence churn prediction — ranked by importance score</div>', unsafe_allow_html=True)
    hi = imp['Importance'].max()
    i_cols = [CC if v==hi else CB for v in imp['Importance']]
    fig, ax = sfig(5.5, 3.4)
    h = ax.barh(imp['Feature'], imp['Importance'], color=i_cols, edgecolor='none', height=0.5, zorder=2)
    for bar, v in zip(h, imp['Importance']):
        ax.text(v+0.003, bar.get_y()+bar.get_height()/2, f'{v:.3f}', va='center', fontsize=8, fontweight='600', color='#E2EAF4')
    top_f = imp.iloc[-1]['Feature']
    ax.annotate(f'Top predictor: {top_f.replace("_"," ")}',
                xy=(hi, len(imp)-1), xytext=(hi*0.5, len(imp)-1.7),
                fontsize=7.5, color=CC,
                arrowprops=dict(arrowstyle='->', color=CC, lw=1.1))
    ax.xaxis.grid(True, color=GRID, linewidth=0.6, zorder=0); ax.set_axisbelow(True)
    ax.set_xlabel('Importance Score', fontsize=8.5, color='#8FA3C0')
    fig.tight_layout(pad=0.4); st.pyplot(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Feature Interpretation ──────────────────────────────────
factor_map = {
    'Monthly_Usage_Score':        ('📊','Monthly Usage Score','Low product engagement is a strong early-warning signal of disengagement and cancellation.'),
    'Payment_Delay_Days':         ('💳','Payment Delay Days','Prolonged payment delays are a significant predictor of upcoming churn.'),
    'Contract_Length_Months':     ('📋','Contract Length','Short-term contracts correlate with higher churn — less commitment means easier exit.'),
    'Support_Tickets_Last30Days': ('🎟️','Support Tickets','High complaint volumes reflect service dissatisfaction and elevated retention risk.'),
    'Monthly_Revenue_USD':        ('💰','Monthly Revenue','Revenue tier affects perceived value; lower-spend clients show higher churn propensity.'),
}
top3 = imp.nlargest(3,'Importance')['Feature'].tolist()
st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
st.markdown('<div class="panel-title">Churn Factor Interpretation</div><div class="panel-sub">Top three factors driving churn — insight from the Decision Tree model</div>', unsafe_allow_html=True)
st.markdown('<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.65rem;margin-top:0.6rem;">', unsafe_allow_html=True)
for f in top3:
    icon, label, desc = factor_map.get(f, ('📌', f, ''))
    st.markdown(f"""
    <div style="background:var(--surface2);border:1px solid var(--border);
                border-top:2px solid var(--cyan);border-radius:8px;
                padding:0.8rem 1rem;font-size:0.78rem;color:var(--text-primary);line-height:1.55;">
        <div style="font-size:1.15rem;margin-bottom:0.3rem;">{icon}</div>
        <div style="font-weight:700;color:var(--cyan);font-size:0.8rem;margin-bottom:0.3rem;">{label}</div>
        {desc}
    </div>""", unsafe_allow_html=True)
st.markdown('</div></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  PART D — TOP 20 HIGH-RISK CLIENTS TABLE
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="sec-hdr">Part D — Top 20 High-Risk Clients</div>', unsafe_allow_html=True)

show = [c for c in ['Client_ID','Industry','Region','Monthly_Revenue_USD','Risk_Score',
                     'Risk_Category','Payment_Delay_Days','Support_Tickets_Last30Days',
                     'Contract_Length_Months'] if c in filtered.columns]
top20 = filtered[filtered['Risk_Category']=='High Risk'].sort_values('Risk_Score', ascending=False).head(20)

if not top20.empty:
    st.dataframe(top20[show].reset_index(drop=True), use_container_width=True, height=340)
else:
    st.info("No High Risk clients in the current filter selection.")

# ══════════════════════════════════════════════════════════════
#  PART E — RETENTION STRATEGY
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="sec-hdr">Part E — AI-Based Retention Suggestions</div>', unsafe_allow_html=True)

if st.button("⚡ Generate Retention Strategy"):
    st.markdown("""
    <div class="chart-panel">
      <div class="panel-title" style="margin-bottom:0.75rem;">Strategic Retention Recommendations</div>
      <div class="reten-item"><span class="reten-ico">💳</span><span><strong>Payment Relief Program:</strong> Offer structured instalment plans or early-payment discounts to clients with payment delays exceeding 30 days. Reducing financial friction directly improves renewal intent.</span></div>
      <div class="reten-item"><span class="reten-ico">🤝</span><span><strong>Dedicated Account Management:</strong> Assign senior account managers to clients with more than 5 support tickets in 30 days. Personalised outreach reduces perceived service gaps and improves satisfaction.</span></div>
      <div class="reten-item"><span class="reten-ico">📋</span><span><strong>Long-Term Contract Incentives:</strong> Provide 10–15% pricing discounts for clients committing to 24-month contracts. Extended commitment significantly lowers churn probability.</span></div>
      <div class="reten-item"><span class="reten-ico">⚡</span><span><strong>SLA-Backed Support Guarantees:</strong> Implement response-time SLAs for Tier-1 high-revenue clients. Service reliability is a critical factor in B2B renewal decisions.</span></div>
      <div class="reten-item"><span class="reten-ico">📈</span><span><strong>Engagement &amp; Onboarding Refresh:</strong> Launch targeted training sessions for clients with usage scores below 50. Higher platform adoption directly reduces disengagement-driven churn.</span></div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  PART F — ETHICAL IMPLICATIONS
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="sec-hdr">Part F — Ethical Implications of Predictive AI</div>', unsafe_allow_html=True)

st.markdown("""
<div class="chart-panel">
  <div class="panel-title">Responsible AI in B2B Churn Prediction</div>
  <div class="panel-sub">Key ethical dimensions to consider when deploying predictive decision systems</div>
  <div class="ethics-grid">
    <div class="ethics-card"><strong>📊 Algorithmic Bias</strong><br>
    Models trained on historical data can inherit and amplify systemic biases. Regular fairness audits across industry, region, and client size are essential to ensure equitable outcomes.</div>
    <div class="ethics-card"><strong>🏷️ Impact of Risk Labelling</strong><br>
    Categorising clients as "High Risk" can negatively affect business relationships. Risk scores must be treated as internal operational signals — not customer-facing judgements.</div>
    <div class="ethics-card"><strong>🔒 Data Privacy &amp; Governance</strong><br>
    All client data must be handled under strict governance frameworks and in compliance with GDPR, PDPA, and applicable regional data privacy regulations.</div>
    <div class="ethics-card"><strong>🤖 Human Oversight</strong><br>
    AI predictions should augment human decision-making, not replace it. Final retention decisions require relationship managers who can apply context, empathy, and sound judgement.</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer">
  Developed by Group-2 · Rhinos · BBA Semester 4 · Section A · Woxsen University<br>
  <span style="color:#2A3D58;display:block;margin-top:0.3rem;">
    Built with Python · Streamlit · scikit-learn Decision Tree · Matplotlib
  </span>
</div>
""", unsafe_allow_html=True)
