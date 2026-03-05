import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
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
#  DESIGN TOKENS  —  Blue + Cyan only
#  Primary:  #2563EB  (rich blue)
#  Accent:   #06B6D4  (cyan)
#  All other states use opacity/lightness variants of these two
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=Manrope:wght@700;800&display=swap');

/* ── Tokens ────────────────────────────────────────── */
:root {
    --bg:        #05090F;
    --s1:        #0A1120;   /* surface 1 */
    --s2:        #0D1628;   /* surface 2 */
    --s3:        #111E34;   /* surface 3 — hover / alt row */

    --blue:      #2563EB;
    --blue-lt:   #3B82F6;
    --blue-dim:  rgba(37,99,235,0.18);
    --blue-line: rgba(37,99,235,0.22);

    --cyan:      #06B6D4;
    --cyan-lt:   #22D3EE;
    --cyan-dim:  rgba(6,182,212,0.14);
    --cyan-line: rgba(6,182,212,0.20);

    /* semantic — still blue/cyan family */
    --hi:   var(--blue-lt);   /* highlight value */
    --lo:   var(--cyan);      /* secondary highlight */

    --border:  rgba(37,99,235,0.16);
    --border2: rgba(6,182,212,0.12);

    --t1: #DDE8F8;   /* text primary */
    --t2: #7A9BC8;   /* text secondary */
    --t3: #3D5A80;   /* text muted */
}

/* ── Base ──────────────────────────────────────────── */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--t1) !important;
    font-size: 14px;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2.2rem 3rem; max-width: 1400px; }
::-webkit-scrollbar { width: 5px; background: var(--bg); }
::-webkit-scrollbar-thumb { background: #1a2d4a; border-radius: 3px; }

/* ── Sidebar ───────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--s1) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--t1) !important; }
[data-testid="stSidebar"] .stMultiSelect > div > div {
    background: var(--s2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 7px !important;
}
[data-testid="stSidebar"] .stMultiSelect span {
    background: var(--blue-dim) !important;
    color: var(--blue-lt) !important;
    border-radius: 4px !important;
}

/* ── Header ────────────────────────────────────────── */
.hdr {
    background: var(--s1);
    border-bottom: 1px solid var(--border);
    padding: 0.95rem 0 1.1rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.hdr-left { display: flex; align-items: center; gap: 1.15rem; }
.hdr-logo { height: 42px; width: auto; }
.hdr-rule { width: 1px; height: 36px; background: var(--border); }
.hdr-title {
    font-family: 'Manrope', sans-serif;
    font-size: 1.15rem; font-weight: 800;
    color: var(--t1); letter-spacing: -0.02em; line-height: 1.2;
}
.hdr-sub {
    font-size: 0.67rem; font-weight: 500;
    color: var(--t3); text-transform: uppercase;
    letter-spacing: 0.07em; margin-top: 0.18rem;
}
.hdr-right { display: flex; gap: 0.45rem; flex-wrap: wrap; align-items: center; }
.tag {
    font-size: 0.63rem; font-weight: 600; letter-spacing: 0.06em;
    text-transform: uppercase; padding: 0.24rem 0.6rem;
    border-radius: 5px; border: 1px solid var(--border);
    color: var(--t2); background: var(--s2);
}

/* ── Section label ─────────────────────────────────── */
.slbl {
    font-family: 'Manrope', sans-serif;
    font-size: 0.68rem; font-weight: 800;
    color: var(--t3); letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 1.7rem 0 0.75rem;
    display: flex; align-items: center; gap: 0.5rem;
}
.slbl::after { content:''; flex:1; height:1px; background: var(--border); }

/* ── Panel ─────────────────────────────────────────── */
.panel {
    background: var(--s1);
    border: 1px solid var(--border);
    border-radius: 11px;
    padding: 1rem 1.2rem 0.9rem;
    margin-bottom: 0.8rem;
}
.ptitle {
    font-family: 'Manrope', sans-serif;
    font-size: 0.87rem; font-weight: 800;
    color: var(--t1); letter-spacing: -0.01em;
    margin-bottom: 0.1rem;
}
.psub { font-size: 0.68rem; color: var(--t3); margin-bottom: 0.8rem; font-weight: 400; }

/* ── KPI grid ──────────────────────────────────────── */
.kgrid { display: grid; grid-template-columns: repeat(4,1fr); gap: 0.8rem; margin-bottom: 0.9rem; }
.kcard {
    background: var(--s1);
    border: 1px solid var(--border);
    border-radius: 11px;
    padding: 1rem 1.15rem;
    position: relative; overflow: hidden;
}
.kcard::before {
    content:''; position:absolute; top:0; left:0; right:0; height:2px;
    background: linear-gradient(90deg, var(--blue), var(--cyan));
}
/* alternate — cyan-led accent for 2 cards */
.kcard.cy::before { background: linear-gradient(90deg, var(--cyan), var(--blue-lt)); }

.kico { font-size: 1.15rem; margin-bottom: 0.45rem; }
.knum {
    font-family: 'Manrope', sans-serif;
    font-size: 1.8rem; font-weight: 800;
    letter-spacing: -0.04em; line-height: 1;
    margin-bottom: 0.25rem;
}
.kcard:nth-child(odd)  .knum { color: var(--blue-lt); }
.kcard:nth-child(even) .knum { color: var(--cyan); }
.klbl { font-size: 0.63rem; font-weight: 700; color: var(--t3); text-transform: uppercase; letter-spacing: 0.08em; }

/* ── Gauge ─────────────────────────────────────────── */
.gwrap {
    background: var(--s1); border: 1px solid var(--border);
    border-radius: 11px; padding: 0.95rem 1.2rem 1.05rem; margin-bottom: 0.8rem;
}
.gtop { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:0.4rem; }
.glbl { font-family:'Manrope',sans-serif; font-size:0.88rem; font-weight:800; color:var(--t1); }
.gpct { font-family:'Manrope',sans-serif; font-size:2rem; font-weight:800; letter-spacing:-0.03em; color:#FFFFFF !important; text-shadow: 0 0 18px rgba(6,182,212,0.6); }
.gtrk { background:var(--s2); border-radius:99px; height:7px; overflow:hidden; margin-bottom:0.48rem; }
.gfil { height:100%; border-radius:99px; }
/* low/mid/high all use blue-cyan; intensity varies */
.glo  { background: linear-gradient(90deg, var(--cyan), var(--cyan-lt)); color: var(--cyan); }
.gmd  { background: linear-gradient(90deg, var(--blue-lt), var(--cyan)); color: var(--blue-lt); }
.ghi  { background: linear-gradient(90deg, var(--blue), #1D4ED8);        color: var(--blue-lt); }

/* ── Status pill ───────────────────────────────────── */
.pill {
    display:inline-flex; align-items:center; gap:0.28rem;
    font-size:0.68rem; font-weight:600; letter-spacing:0.04em;
    padding:0.25rem 0.65rem; border-radius:99px;
}
.plo { background:var(--cyan-dim);  color:var(--cyan);    border:1px solid var(--cyan-line);  }
.pmd { background:var(--blue-dim);  color:var(--blue-lt); border:1px solid var(--blue-line);  }
.phi { background:var(--blue-dim);  color:var(--blue-lt); border:1px solid var(--blue-line);  }

/* ── Button ────────────────────────────────────────── */
.stButton > button {
    background: var(--blue) !important;
    color: #fff !important; border: none !important;
    border-radius: 7px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.78rem !important; font-weight: 600 !important;
    letter-spacing: 0.04em !important; padding: 0.5rem 1.2rem !important;
    box-shadow: 0 2px 12px rgba(37,99,235,0.35) !important;
    transition: all 0.18s !important;
}
.stButton > button:hover {
    background: var(--blue-lt) !important;
    box-shadow: 0 4px 18px rgba(37,99,235,0.5) !important;
    transform: translateY(-1px) !important;
}

/* ── Dataframe ─────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 9px !important;
    overflow: hidden !important;
    border: 1px solid var(--border) !important;
}

/* ── Retention items ───────────────────────────────── */
.ritem {
    display:flex; gap:0.72rem; align-items:flex-start;
    background:var(--s2); border:1px solid var(--border);
    border-left:3px solid var(--blue);
    border-radius:7px; padding:0.7rem 0.95rem;
    margin-bottom:0.45rem;
    font-size:0.79rem; color:var(--t1); line-height:1.58;
}
.rico { font-size:0.95rem; flex-shrink:0; margin-top:0.07rem; }
.ritem strong { color: var(--cyan); font-weight:600; }

/* ── Ethics grid ───────────────────────────────────── */
.egrid { display:grid; grid-template-columns:1fr 1fr; gap:0.6rem; margin-top:0.6rem; }
.ecard {
    background:var(--s2); border:1px solid var(--border);
    border-top:2px solid var(--cyan);
    border-radius:7px; padding:0.78rem 0.95rem;
    font-size:0.77rem; color:var(--t1); line-height:1.58;
}
.ecard strong { color:var(--cyan); font-weight:600; }

/* ── Accuracy display ──────────────────────────────── */
.accb { display:flex; align-items:baseline; gap:0.4rem; margin:0.4rem 0 0.72rem; }
.accn {
    font-family:'Manrope',sans-serif; font-size:2.5rem; font-weight:800;
    color:var(--cyan); letter-spacing:-0.05em; line-height:1;
}
.accs { font-size:0.66rem; font-weight:700; color:var(--t3); text-transform:uppercase; letter-spacing:0.07em; }

/* ── Sidebar labels ────────────────────────────────── */
.sblbl { font-size:0.64rem; font-weight:700; color:var(--t3); text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.28rem; }

/* ── Team cards ────────────────────────────────────── */
.tgrid { display:grid; grid-template-columns:1fr 1fr; gap:0.5rem; margin-top:0.6rem; }
.tcard {
    background:var(--s2); border:1px solid var(--border);
    border-radius:7px; padding:0.58rem 0.75rem;
    display:flex; align-items:center; gap:0.52rem;
}
.tav {
    width:28px; height:28px; border-radius:50%; flex-shrink:0;
    background:linear-gradient(135deg, var(--blue), var(--cyan));
    display:flex; align-items:center; justify-content:center;
    font-size:0.72rem; font-weight:700; color:#fff;
}
.tnm { font-size:0.71rem; font-weight:600; color:var(--t1); }
.trl { font-size:0.58rem; color:var(--t3); text-transform:uppercase; letter-spacing:0.04em; }

/* ── Factor cards ──────────────────────────────────── */
.fcrd {
    background:var(--s2); border:1px solid var(--border);
    border-top:2px solid var(--blue-lt);
    border-radius:7px; padding:0.78rem 0.95rem;
    font-size:0.77rem; color:var(--t1); line-height:1.55;
}
.fcrd .ftitle { font-weight:700; color:var(--blue-lt); font-size:0.79rem; margin:0.25rem 0 0.28rem; }

/* ── Footer ────────────────────────────────────────── */
.footer {
    text-align:center; padding:0.85rem 0 0.4rem;
    font-size:0.62rem; color:var(--t3);
    letter-spacing:0.05em; text-transform:uppercase;
    border-top:1px solid var(--border); margin-top:1.5rem;
}

/* ── Table section label ───────────────────────────── */
.tbl-lbl {
    font-family:'Manrope',sans-serif;
    font-size:0.82rem; font-weight:800; color:var(--t1);
    margin-bottom:0.12rem;
}
.tbl-sub { font-size:0.68rem; color:var(--t3); margin-bottom:0.55rem; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  MATPLOTLIB  — Blue + Cyan palette only
# ══════════════════════════════════════════════════════════════
BG_MPL = '#0A1120'
GRID_C = '#111E34'

plt.rcParams.update({
    'figure.facecolor': BG_MPL, 'axes.facecolor': BG_MPL,
    'axes.edgecolor': GRID_C,   'axes.labelcolor': '#7A9BC8',
    'axes.titlecolor': '#DDE8F8',
    'xtick.color': '#7A9BC8',   'ytick.color': '#7A9BC8',
    'grid.color': GRID_C,       'grid.alpha': 0.8,
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.spines.top': False,   'axes.spines.right': False,
    'axes.spines.left': False,  'axes.spines.bottom': False,
})

# Two-color palette only
BLUE  = '#3B82F6'
CYAN  = '#06B6D4'
BLUE2 = '#1D4ED8'   # darker blue for contrast bars
CYAN2 = '#22D3EE'   # lighter cyan

def bar_gradient(n):
    """n shades alternating between blue and cyan family"""
    return [BLUE if i % 2 == 0 else CYAN for i in range(n)]

# ══════════════════════════════════════════════════════════════
#  LOAD & PREPARE DATA
# ══════════════════════════════════════════════════════════════
try:
    data = pd.read_csv("B2B_Client_Churn_5000.csv")
    data.columns = data.columns.str.strip().str.replace(" ", "_")
    data['Renewal_Status'] = data['Renewal_Status'].map({'Yes': 1, 'No': 0})

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
    <div style="text-align:center;padding:0.8rem 0 1.15rem;">
        <img src="https://woxsen.edu.in/wp-content/uploads/2022/04/woxsen-logo-new.png"
             style="height:36px;width:auto;filter:brightness(1.1);"
             onerror="this.style.display='none'">
        <div style="font-size:0.6rem;color:#3D5A80;text-transform:uppercase;
                    letter-spacing:0.09em;margin-top:0.45rem;">Dashboard Filters</div>
    </div>
    """, unsafe_allow_html=True)

    if data_ok:
        st.markdown('<div class="sblbl">Region</div>', unsafe_allow_html=True)
        region = st.multiselect("", data['Region'].unique(),
                                placeholder="All regions", label_visibility="collapsed")
        st.markdown('<div class="sblbl" style="margin-top:0.8rem;">Industry</div>', unsafe_allow_html=True)
        industry = st.multiselect("", data['Industry'].unique(),
                                  placeholder="All industries", label_visibility="collapsed")
        st.markdown('<div class="sblbl" style="margin-top:0.8rem;">Risk Category</div>', unsafe_allow_html=True)
        risk_filter = st.multiselect("", data['Risk_Category'].unique(),
                                     placeholder="All categories", label_visibility="collapsed")
    else:
        region = industry = risk_filter = []

    st.markdown('<div style="height:1px;background:rgba(37,99,235,0.14);margin:1.1rem 0;"></div>',
                unsafe_allow_html=True)

    if st.button("👥 Team Members"):
        st.markdown("""
        <div class="tgrid">
          <div class="tcard"><div class="tav">M</div><div><div class="tnm">Mohnish Singh Patwal</div><div class="trl">Member</div></div></div>
          <div class="tcard"><div class="tav">S</div><div><div class="tnm">Shreyas Kandi</div><div class="trl">Member</div></div></div>
          <div class="tcard"><div class="tav">A</div><div><div class="tnm">Akash Krishna</div><div class="trl">Member</div></div></div>
          <div class="tcard"><div class="tav">N</div><div><div class="tnm">Nihal Talampally</div><div class="trl">Member</div></div></div>
        </div>
        <div style="font-size:0.6rem;color:#3D5A80;text-align:center;margin-top:0.65rem;">
            Section A · BBA Semester 4 · Woxsen University
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="hdr">
  <div class="hdr-left">
    <img class="hdr-logo"
         src="https://woxsen.edu.in/wp-content/uploads/2022/04/woxsen-logo-new.png"
         onerror="this.src='';this.style.display='none';">
    <div class="hdr-rule"></div>
    <div>
      <div class="hdr-title">B2B Client Risk Intelligence Dashboard</div>
      <div class="hdr-sub">AI-powered churn prediction &amp; retention analytics</div>
    </div>
  </div>
  <div class="hdr-right">
    <span class="tag">Group-2 · Rhinos</span>
    <span class="tag">BBA Sem 4 · Section A</span>
    <span class="tag">Woxsen University</span>
  </div>
</div>
""", unsafe_allow_html=True)

if not data_ok:
    st.error(f"Could not load CSV: {data_err}. Place B2B_Client_Churn_5000.csv in the same folder.")
    st.stop()

# Apply filters
filtered = data.copy()
if region:      filtered = filtered[filtered['Region'].isin(region)]
if industry:    filtered = filtered[filtered['Industry'].isin(industry)]
if risk_filter: filtered = filtered[filtered['Risk_Category'].isin(risk_filter)]

# ══════════════════════════════════════════════════════════════
#  PART D — KPI CARDS
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="slbl">Key Performance Indicators</div>', unsafe_allow_html=True)

total   = len(filtered)
hi_cnt  = (filtered['Risk_Category'] == "High Risk").sum()
churn   = round((1 - filtered['Renewal_Status'].mean()) * 100, 1)
avg_rev = round(filtered['Monthly_Revenue_USD'].mean(), 0)

st.markdown(f"""
<div class="kgrid">
  <div class="kcard">
    <div class="kico">👥</div>
    <div class="knum">{total:,}</div>
    <div class="klbl">Total Clients</div>
  </div>
  <div class="kcard cy">
    <div class="kico">⚠️</div>
    <div class="knum">{hi_cnt:,}</div>
    <div class="klbl">High Risk Clients</div>
  </div>
  <div class="kcard">
    <div class="kico">📉</div>
    <div class="knum">{churn}%</div>
    <div class="klbl">Predicted Churn Rate</div>
  </div>
  <div class="kcard cy">
    <div class="kico">💰</div>
    <div class="knum">${avg_rev:,.0f}</div>
    <div class="klbl">Avg Revenue / Client</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Churn Gauge ──────────────────────────────────────────────
g_cls = "glo" if churn < 20 else ("gmd" if churn < 40 else "ghi")
p_cls = "plo" if churn < 20 else ("pmd" if churn < 40 else "phi")
p_txt = ("● Low Risk" if churn < 20 else ("● Moderate Risk" if churn < 40 else "● Elevated Risk"))

st.markdown(f"""
<div class="gwrap">
  <div class="gtop">
    <span class="glbl">Overall Churn Risk Gauge</span>
    <span class="gpct">{churn}%</span>
  </div>
  <div class="gtrk"><div class="gfil {g_cls}" style="width:{min(churn,100)}%;"></div></div>
  <span class="pill {p_cls}">{p_txt}</span>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  PART B + D — CHARTS
# ══════════════════════════════════════════════════════════════
def sfig(w=5, h=3.2):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG_MPL)
    ax.set_facecolor(BG_MPL)
    return fig, ax

st.markdown('<div class="slbl">Risk Distribution &amp; Industry Analysis</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)

# Chart 1 — Risk Distribution
with c1:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="ptitle">Risk Category Distribution</div>'
                '<div class="psub">Client count by risk segment</div>', unsafe_allow_html=True)
    rc = filtered['Risk_Category'].value_counts().reindex(
        ["Low Risk", "Medium Risk", "High Risk"]).fillna(0)
    fig, ax = sfig()
    # Use blue shades: light → mid → dark
    bar_cols = [CYAN, BLUE, BLUE2]
    bars = ax.bar(rc.index, rc.values, color=bar_cols, edgecolor='none', width=0.42, zorder=2)
    for b, v in zip(bars, rc.values):
        ax.text(b.get_x()+b.get_width()/2, v+15, f'{int(v):,}',
                ha='center', fontsize=9, fontweight='600', color='#DDE8F8')
    ax.yaxis.grid(True, color=GRID_C, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylabel('Clients', fontsize=8.5)
    fig.tight_layout(pad=0.4)
    st.pyplot(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Chart 2 — Industry Risk
with c2:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="ptitle">Industry-wise Risk Analysis</div>'
                '<div class="psub">Average risk score per industry</div>', unsafe_allow_html=True)
    ir = filtered.groupby('Industry')['Risk_Score'].mean().sort_values()
    fig, ax = sfig()
    # alternate blue/cyan per bar
    hb = ax.barh(ir.index, ir.values,
                 color=bar_gradient(len(ir)), edgecolor='none', height=0.52, zorder=2)
    for bar, v in zip(hb, ir.values):
        ax.text(v+0.04, bar.get_y()+bar.get_height()/2,
                f'{v:.1f}', va='center', fontsize=8, fontweight='600', color='#DDE8F8')
    ax.xaxis.grid(True, color=GRID_C, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel('Avg Risk Score', fontsize=8.5)
    fig.tight_layout(pad=0.4)
    st.pyplot(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

c3, c4 = st.columns(2)

# Chart 3 — Revenue vs Risk Scatter
with c3:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="ptitle">Revenue vs Risk Score</div>'
                '<div class="psub">Scatter — high-revenue clients at churn risk</div>', unsafe_allow_html=True)
    samp = filtered.sample(min(600, len(filtered)), random_state=42)
    # two-tone scatter: low risk = cyan, high risk = blue
    sc = samp['Risk_Category'].map(
        {"Low Risk": CYAN2, "Medium Risk": BLUE, "High Risk": BLUE2})
    fig, ax = sfig()
    ax.scatter(samp['Monthly_Revenue_USD'], samp['Risk_Score'],
               c=sc, alpha=0.65, s=16, edgecolors='none', zorder=2)
    ax.yaxis.grid(True, color=GRID_C, linewidth=0.6, zorder=0)
    ax.xaxis.grid(True, color=GRID_C, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel('Monthly Revenue (USD)', fontsize=8.5)
    ax.set_ylabel('Risk Score', fontsize=8.5)
    patches = [mpatches.Patch(color=CYAN2, label='Low'),
               mpatches.Patch(color=BLUE,  label='Medium'),
               mpatches.Patch(color=BLUE2, label='High')]
    ax.legend(handles=patches, fontsize=8, framealpha=0.15,
              facecolor=BG_MPL, edgecolor=GRID_C, labelcolor='#DDE8F8')
    fig.tight_layout(pad=0.4)
    st.pyplot(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Chart 4 — Contract Length vs Churn
with c4:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="ptitle">Contract Length vs Churn Rate</div>'
                '<div class="psub">How contract duration relates to renewal behaviour</div>', unsafe_allow_html=True)
    cl = (1 - filtered.groupby('Contract_Length_Months')['Renewal_Status'].mean()) * 100
    fig, ax = sfig()
    ax.fill_between(cl.index, cl.values, alpha=0.13, color=BLUE, zorder=1)
    ax.plot(cl.index, cl.values, color=CYAN, linewidth=2, zorder=3)
    ax.yaxis.grid(True, color=GRID_C, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel('Contract Length (Months)', fontsize=8.5)
    ax.set_ylabel('Churn Rate (%)', fontsize=8.5)
    fig.tight_layout(pad=0.4)
    st.pyplot(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Chart 5 — Churn by Industry (full width)
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown('<div class="ptitle">Churn Rate by Industry</div>'
            '<div class="psub">Percentage of non-renewals across industry verticals</div>', unsafe_allow_html=True)
ci = (1 - data.groupby('Industry')['Renewal_Status'].mean()).sort_values() * 100
fig, ax = sfig(10, 3.2)
h_ci = ax.barh(ci.index, ci.values,
               color=bar_gradient(len(ci)), edgecolor='none', height=0.52, zorder=2)
for bar, v in zip(h_ci, ci.values):
    ax.text(v+0.3, bar.get_y()+bar.get_height()/2,
            f'{v:.1f}%', va='center', fontsize=8, fontweight='600', color='#DDE8F8')
ax.xaxis.grid(True, color=GRID_C, linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
ax.set_xlabel('Churn Rate (%)', fontsize=8.5)
fig.tight_layout(pad=0.4)
st.pyplot(fig, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# Chart 6 — Payment Delay vs Churn (full width)
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown('<div class="ptitle">Payment Delay vs Churn Probability</div>'
            '<div class="psub">How increasing payment delays elevate churn likelihood</div>', unsafe_allow_html=True)
dc = (1 - data.groupby('Payment_Delay_Days')['Renewal_Status'].mean()) * 100
fig, ax = sfig(10, 3)
ax.plot(dc.index, dc.values, color=CYAN, linewidth=2, zorder=3)
ax.fill_between(dc.index, dc.values, alpha=0.1, color=CYAN, zorder=2)
ax.axhline(40, color=BLUE, linestyle='--', linewidth=1.2, alpha=0.6, label='40% threshold')
ax.yaxis.grid(True, color=GRID_C, linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
ax.set_xlabel('Payment Delay (Days)', fontsize=8.5)
ax.set_ylabel('Churn Rate (%)', fontsize=8.5)
ax.legend(fontsize=8, framealpha=0.15, facecolor=BG_MPL, edgecolor=GRID_C, labelcolor='#DDE8F8')
fig.tight_layout(pad=0.4)
st.pyplot(fig, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  PART C — DECISION TREE
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="slbl">Part C — Machine Learning: Decision Tree Classifier</div>',
            unsafe_allow_html=True)

FEATURES = ['Monthly_Usage_Score', 'Payment_Delay_Days', 'Contract_Length_Months',
            'Support_Tickets_Last30Days', 'Monthly_Revenue_USD']
X = data[FEATURES]
y = data['Renewal_Status']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
dt = DecisionTreeClassifier(max_depth=6, random_state=42)
dt.fit(X_tr, y_tr)
pred = dt.predict(X_te)
acc  = round(accuracy_score(y_te, pred) * 100, 1)
cm   = confusion_matrix(y_te, pred)
imp  = pd.DataFrame({'Feature': FEATURES,
                     'Importance': dt.feature_importances_}).sort_values('Importance')

c5, c6 = st.columns([1, 1.6])

with c5:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="ptitle">Model Performance</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="accb">
      <div class="accn">{acc}%</div>
      <div class="accs">Test Set Accuracy</div>
    </div>
    <span class="pill plo">✓ Decision Tree · max_depth=6</span>
    """, unsafe_allow_html=True)

    fig, ax = sfig(3.4, 2.8)
    # Use solid dark-blue fills so white text is always readable
    cell_colors = [
        [BLUE2, '#0D1628'],   # row 0: dark blue, near-black
        ['#0D1628', BLUE],    # row 1: near-black, medium blue
    ]
    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                       color=cell_colors[i][j], zorder=1))
            ax.text(j, i, f'{cm[i,j]:,}', ha='center', va='center',
                    fontsize=16, fontweight='800', color='#FFFFFF', zorder=2)
    ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.5, 1.5)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred: Churn', 'Pred: Renew'], fontsize=8.5,
                        color='#DDE8F8', fontweight='500')
    ax.set_yticklabels(['Act: Churn',  'Act: Renew'],  fontsize=8.5,
                        color='#DDE8F8', fontweight='500')
    ax.set_title('Confusion Matrix', fontsize=9, color='#DDE8F8',
                 fontweight='700', pad=8)
    ax.tick_params(length=0)
    fig.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c6:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="ptitle">Feature Importance</div>'
                '<div class="psub">Factors ranked by predictive influence on churn</div>',
                unsafe_allow_html=True)
    hi_val = imp['Importance'].max()
    i_cols = [CYAN if v == hi_val else BLUE for v in imp['Importance']]
    fig, ax = sfig(5.5, 3.4)
    hb = ax.barh(imp['Feature'], imp['Importance'],
                 color=i_cols, edgecolor='none', height=0.48, zorder=2)
    for bar, v in zip(hb, imp['Importance']):
        ax.text(v+0.003, bar.get_y()+bar.get_height()/2,
                f'{v:.3f}', va='center', fontsize=8, fontweight='600', color='#DDE8F8')
    top_f = imp.iloc[-1]['Feature']
    ax.annotate(f'Top predictor: {top_f.replace("_"," ")}',
                xy=(hi_val, len(imp)-1), xytext=(hi_val*0.48, len(imp)-1.65),
                fontsize=7.5, color=CYAN,
                arrowprops=dict(arrowstyle='->', color=CYAN, lw=1.1))
    ax.xaxis.grid(True, color=GRID_C, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel('Importance Score', fontsize=8.5)
    fig.tight_layout(pad=0.4)
    st.pyplot(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Factor Interpretation ────────────────────────────────────
factor_map = {
    'Monthly_Usage_Score':        ('📊', 'Monthly Usage Score',
        'Low product engagement is a strong early-warning signal of disengagement and cancellation.'),
    'Payment_Delay_Days':         ('💳', 'Payment Delay Days',
        'Prolonged payment delays are a significant predictor of upcoming churn.'),
    'Contract_Length_Months':     ('📋', 'Contract Length',
        'Short-term contracts correlate with higher churn — less commitment means easier exit.'),
    'Support_Tickets_Last30Days': ('🎟️', 'Support Tickets',
        'High complaint volumes reflect service dissatisfaction and elevated retention risk.'),
    'Monthly_Revenue_USD':        ('💰', 'Monthly Revenue',
        'Revenue tier affects perceived value; lower-spend clients show higher churn propensity.'),
}
top3 = imp.nlargest(3, 'Importance')['Feature'].tolist()

st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown('<div class="ptitle">Churn Factor Interpretation</div>'
            '<div class="psub">Top three predictors identified by the Decision Tree model</div>',
            unsafe_allow_html=True)
st.markdown('<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.6rem;margin-top:0.5rem;">',
            unsafe_allow_html=True)
for f in top3:
    ico, lbl, desc = factor_map.get(f, ('📌', f, ''))
    st.markdown(f"""
    <div class="fcrd">
        <div style="font-size:1.1rem;">{ico}</div>
        <div class="ftitle">{lbl}</div>
        <div style="color:#7A9BC8;font-size:0.75rem;">{desc}</div>
    </div>""", unsafe_allow_html=True)
st.markdown('</div></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  PART D — TWO CLIENT TABLES
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="slbl">Part D — High-Risk Client Tables</div>', unsafe_allow_html=True)

BASE_COLS = ['Client_ID', 'Industry', 'Region', 'Monthly_Revenue_USD',
             'Risk_Score', 'Risk_Category', 'Payment_Delay_Days',
             'Support_Tickets_Last30Days', 'Contract_Length_Months']
show_cols = [c for c in BASE_COLS if c in filtered.columns]
hi_risk_df = filtered[filtered['Risk_Category'] == 'High Risk']

# ── Table 1: Top 20 High-Risk Clients (sorted by Risk Score) ─
st.markdown("""
<div class="tbl-lbl">Table 1 — Top 20 High-Risk Clients</div>
<div class="tbl-sub">Clients classified as High Risk, sorted by Risk Score (highest first)</div>
""", unsafe_allow_html=True)

top20 = hi_risk_df.sort_values('Risk_Score', ascending=False).head(20)
if not top20.empty:
    st.dataframe(top20[show_cols].reset_index(drop=True),
                 use_container_width=True, height=350)
else:
    st.info("No High Risk clients in the current filter selection.")

# spacing
st.markdown('<div style="height:0.6rem;"></div>', unsafe_allow_html=True)

# ── Table 2: High-Revenue Clients at Risk (sorted by Revenue) ─
st.markdown("""
<div class="tbl-lbl">Table 2 — High-Revenue Clients at Risk</div>
<div class="tbl-sub">High Risk clients ranked by Monthly Revenue — highest commercial exposure first</div>
""", unsafe_allow_html=True)

top_rev = hi_risk_df.sort_values('Monthly_Revenue_USD', ascending=False).head(20)
if not top_rev.empty:
    st.dataframe(top_rev[show_cols].reset_index(drop=True),
                 use_container_width=True, height=350)
else:
    st.info("No High Risk clients in the current filter selection.")

# ══════════════════════════════════════════════════════════════
#  PART E — RETENTION STRATEGY
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="slbl">Part E — AI-Based Retention Suggestions</div>',
            unsafe_allow_html=True)

if st.button("⚡ Generate Retention Strategy"):
    st.markdown("""
    <div class="panel">
      <div class="ptitle" style="margin-bottom:0.7rem;">Strategic Retention Recommendations</div>
      <div class="ritem"><span class="rico">💳</span><span>
        <strong>Payment Relief Program:</strong> Offer structured instalment plans or early-payment
        discounts to clients with payment delays exceeding 30 days. Reducing financial friction
        directly improves renewal intent.
      </span></div>
      <div class="ritem"><span class="rico">🤝</span><span>
        <strong>Dedicated Account Management:</strong> Assign senior account managers to clients
        with more than 5 support tickets in the last 30 days. Personalised outreach reduces
        perceived service gaps and improves satisfaction scores.
      </span></div>
      <div class="ritem"><span class="rico">📋</span><span>
        <strong>Long-Term Contract Incentives:</strong> Provide 10–15% pricing discounts for
        clients committing to 24-month contracts. Extended commitment significantly lowers
        churn probability.
      </span></div>
      <div class="ritem"><span class="rico">⚡</span><span>
        <strong>SLA-Backed Support Guarantees:</strong> Implement response-time SLAs for
        Tier-1 high-revenue clients. Service reliability is a critical factor in B2B renewal
        decisions.
      </span></div>
      <div class="ritem"><span class="rico">📈</span><span>
        <strong>Engagement &amp; Onboarding Refresh:</strong> Launch targeted product training
        for clients with usage scores below 50. Higher platform adoption directly reduces
        disengagement-driven churn.
      </span></div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  PART F — ETHICAL IMPLICATIONS
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="slbl">Part F — Ethical Implications of Predictive AI</div>',
            unsafe_allow_html=True)

st.markdown("""
<div class="panel">
  <div class="ptitle">Responsible AI in B2B Churn Prediction</div>
  <div class="psub">Key ethical dimensions to consider when deploying predictive decision systems</div>
  <div class="egrid">
    <div class="ecard">
      <strong>📊 Algorithmic Bias</strong><br>
      Models trained on historical data can inherit and amplify systemic biases. Regular fairness
      audits across industry, region, and client size are essential to ensure equitable treatment.
    </div>
    <div class="ecard">
      <strong>🏷️ Impact of Risk Labelling</strong><br>
      Categorising clients as "High Risk" can negatively affect business relationships. Risk scores
      must be treated as internal operational signals, not customer-facing judgements.
    </div>
    <div class="ecard">
      <strong>🔒 Data Privacy &amp; Governance</strong><br>
      All client data must be handled under strict governance frameworks and in compliance with
      GDPR, PDPA, and applicable regional data privacy regulations.
    </div>
    <div class="ecard">
      <strong>🤖 Human Oversight</strong><br>
      AI predictions should augment human decision-making, not replace it. Final retention
      decisions require managers who can apply context, empathy, and sound judgement.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer">
  Developed by Group-2 · Rhinos · BBA Semester 4 · Section A · Woxsen University<br>
  <span style="color:#1E3050;display:block;margin-top:0.25rem;">
    Python · Streamlit · scikit-learn Decision Tree · Matplotlib
  </span>
</div>
""", unsafe_allow_html=True)
