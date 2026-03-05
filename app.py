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
    page_title="B2B Client Risk Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════
#  COLOR PALETTE  —  Neon White · Bright Red · Neon Blue
# ══════════════════════════════════════════════════════════════
# Neon White : #F0F8FF / #FFFFFF
# Bright Red : #FF1A1A / #FF4444
# Neon Blue  : #00BFFF / #33CCFF

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Manrope:wght@700;800&display=swap');

:root {
    --bg:       #000000;
    --s1:       #0A0A0A;
    --s2:       #111111;
    --s3:       #181818;

    --nblue:    #00BFFF;
    --nblue-lt: #33CCFF;
    --nblue-glow: rgba(0,191,255,0.25);
    --nblue-dim:  rgba(0,191,255,0.12);
    --nblue-bdr:  rgba(0,191,255,0.22);

    --red:      #FF1A1A;
    --red-lt:   #FF4444;
    --red-glow: rgba(255,26,26,0.22);
    --red-dim:  rgba(255,26,26,0.10);
    --red-bdr:  rgba(255,26,26,0.22);

    --white:    #F0F8FF;
    --white2:   #C8DCF0;
    --white3:   #6A8AAA;
    --white4:   #304560;

    --border:   rgba(0,191,255,0.18);
}

html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--white) !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 3rem; max-width: 1400px; }
::-webkit-scrollbar { width: 5px; background: var(--bg); }
::-webkit-scrollbar-thumb { background: #1a2d45; border-radius: 3px; }

/* ── Sidebar — comprehensive visibility fix ─────────── */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div,
[data-testid="stSidebar"] > div:first-child {
    background-color: #0A0A0A !important;
    border-right: 1px solid rgba(0,191,255,0.22) !important;
}
/* Force ALL text in sidebar to be visible white */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] strong,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] .stMarkdown * {
    color: #F0F8FF !important;
}
/* Multiselect input box */
[data-testid="stSidebar"] [data-baseweb="select"] > div:first-child {
    background-color: #111111 !important;
    border: 1px solid rgba(0,191,255,0.3) !important;
    border-radius: 7px !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] * {
    color: #F0F8FF !important;
    background-color: transparent !important;
}
/* Selected tags/chips */
[data-testid="stSidebar"] [data-baseweb="tag"] {
    background-color: rgba(0,191,255,0.18) !important;
    color: #33CCFF !important;
    border: 1px solid rgba(0,191,255,0.35) !important;
}
/* Dropdown menu options */
[data-testid="stSidebar"] ul[role="listbox"],
[data-testid="stSidebar"] ul[role="listbox"] li {
    background-color: #111111 !important;
    color: #F0F8FF !important;
}
/* Divider line */
[data-testid="stSidebar"] hr {
    border-color: rgba(0,191,255,0.18) !important;
}

/* ── Section label ─────────────────────────────────── */
.slbl {
    font-family: 'Manrope', sans-serif;
    font-size: 0.67rem; font-weight: 800;
    color: var(--white3); letter-spacing: 0.13em; text-transform: uppercase;
    margin: 1.8rem 0 0.8rem;
    display: flex; align-items: center; gap: 0.5rem;
}
.slbl::after { content:''; flex:1; height:1px; background: var(--nblue-bdr); }

/* ── Panel card ────────────────────────────────────── */
.panel {
    background: var(--s1);
    border: 1px solid var(--nblue-bdr);
    border-radius: 12px;
    padding: 1.05rem 1.3rem 1rem;
    margin-bottom: 0.85rem;
}
.ptitle {
    font-family: 'Manrope', sans-serif;
    font-size: 0.88rem; font-weight: 800;
    color: var(--white); letter-spacing: -0.01em; margin-bottom: 0.1rem;
}
.psub { font-size: 0.68rem; color: var(--white3); margin-bottom: 0.85rem; }

/* ── KPI grid ──────────────────────────────────────── */
.kgrid { display:grid; grid-template-columns:repeat(4,1fr); gap:0.85rem; margin-bottom:0.9rem; }
.kcard {
    background: var(--s1);
    border: 1px solid var(--nblue-bdr);
    border-radius: 12px;
    padding: 1.05rem 1.2rem;
    position: relative; overflow: hidden;
}
/* top accent line — alternates blue / red */
.kcard.kb::before { content:''; position:absolute; top:0;left:0;right:0;height:2px;
    background: linear-gradient(90deg, var(--nblue), var(--nblue-lt)); }
.kcard.kr::before { content:''; position:absolute; top:0;left:0;right:0;height:2px;
    background: linear-gradient(90deg, var(--red), var(--red-lt)); }
.kico { font-size:1.15rem; margin-bottom:0.45rem; }
.knum {
    font-family:'Manrope',sans-serif; font-size:1.85rem; font-weight:800;
    letter-spacing:-0.04em; line-height:1; margin-bottom:0.28rem;
}
.kb .knum { color: var(--nblue-lt); text-shadow: 0 0 12px var(--nblue-glow); }
.kr .knum { color: var(--red-lt);   text-shadow: 0 0 12px var(--red-glow); }
.klbl { font-size:0.63rem; font-weight:700; color:var(--white3);
        text-transform:uppercase; letter-spacing:0.08em; }

/* ── Gauge ─────────────────────────────────────────── */
.gwrap {
    background: var(--s1); border: 1px solid var(--nblue-bdr);
    border-radius: 12px; padding:1rem 1.3rem 1.05rem; margin-bottom:0.85rem;
}
.gtop { display:flex; justify-content:space-between; align-items:center; margin-bottom:0.45rem; }
.glbl { font-family:'Manrope',sans-serif; font-size:0.88rem; font-weight:800; color:var(--white); }
.gpct {
    font-family:'Manrope',sans-serif; font-size:2.1rem; font-weight:800;
    letter-spacing:-0.04em; color:#FFFFFF !important;
    text-shadow: 0 0 20px var(--nblue-glow), 0 0 8px rgba(0,191,255,0.5);
}
.gtrk { background:var(--s2); border-radius:99px; height:8px; overflow:hidden; margin-bottom:0.5rem; }
.gfil { height:100%; border-radius:99px; }
.glo { background: linear-gradient(90deg, var(--nblue), var(--nblue-lt)); }
.gmd { background: linear-gradient(90deg, var(--nblue), var(--red-lt)); }
.ghi { background: linear-gradient(90deg, var(--red), var(--red-lt)); }

/* ── Pill badge ────────────────────────────────────── */
.pill {
    display:inline-flex; align-items:center; gap:0.28rem;
    font-size:0.69rem; font-weight:600; letter-spacing:0.04em;
    padding:0.28rem 0.7rem; border-radius:99px;
}
.pb { background:var(--nblue-dim); color:var(--nblue-lt); border:1px solid var(--nblue-bdr); }
.pr { background:var(--red-dim);   color:var(--red-lt);   border:1px solid var(--red-bdr); }

/* ── Button ────────────────────────────────────────── */
.stButton > button {
    background: var(--nblue) !important;
    color: #04070D !important; border: none !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.79rem !important; font-weight: 700 !important;
    letter-spacing: 0.05em !important; padding: 0.52rem 1.3rem !important;
    box-shadow: 0 2px 14px var(--nblue-glow) !important;
    transition: all 0.18s !important;
}
.stButton > button:hover {
    background: var(--nblue-lt) !important;
    box-shadow: 0 4px 20px rgba(0,191,255,0.5) !important;
    transform: translateY(-1px) !important;
}

/* ── Dataframe ─────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 10px !important;
    border: 1px solid var(--nblue-bdr) !important;
    overflow: hidden !important;
}

/* ── Retention items ───────────────────────────────── */
.ritem {
    display:flex; gap:0.75rem; align-items:flex-start;
    background:var(--s2); border:1px solid var(--nblue-bdr);
    border-left:3px solid var(--nblue);
    border-radius:8px; padding:0.72rem 1rem;
    margin-bottom:0.48rem;
    font-size:0.8rem; color:var(--white); line-height:1.58;
}
.rico { font-size:0.95rem; flex-shrink:0; margin-top:0.07rem; }
.ritem strong { color: var(--nblue-lt); font-weight:600; }

/* ── Ethics grid ───────────────────────────────────── */
.egrid { display:grid; grid-template-columns:1fr 1fr; gap:0.65rem; margin-top:0.6rem; }
.ecard {
    background:var(--s2); border:1px solid var(--nblue-bdr);
    border-top:2px solid var(--red);
    border-radius:8px; padding:0.8rem 1rem;
    font-size:0.78rem; color:var(--white); line-height:1.58;
}
.ecard strong { color:var(--red-lt); font-weight:600; }

/* ── Accuracy ──────────────────────────────────────── */
.accb { display:flex; align-items:baseline; gap:0.4rem; margin:0.4rem 0 0.75rem; }
.accn {
    font-family:'Manrope',sans-serif; font-size:2.6rem; font-weight:800;
    color:var(--nblue-lt); letter-spacing:-0.05em; line-height:1;
    text-shadow: 0 0 18px var(--nblue-glow);
}
.accs { font-size:0.66rem; font-weight:700; color:var(--white3);
        text-transform:uppercase; letter-spacing:0.07em; }

/* ── Factor cards ──────────────────────────────────── */
.fcrd {
    background:var(--s2); border:1px solid var(--nblue-bdr);
    border-top:2px solid var(--nblue);
    border-radius:8px; padding:0.8rem 1rem;
    font-size:0.77rem; color:var(--white2); line-height:1.55;
}
.ftitle { font-weight:700; color:var(--nblue-lt); font-size:0.79rem; margin:0.25rem 0 0.28rem; }

/* ── Table labels ──────────────────────────────────── */
.tbl-lbl {
    font-family:'Manrope',sans-serif; font-size:0.84rem; font-weight:800;
    color:var(--white); margin-bottom:0.1rem;
}
.tbl-sub { font-size:0.68rem; color:var(--white3); margin-bottom:0.55rem; }

/* ── Footer ────────────────────────────────────────── */
.footer {
    text-align:center; padding:0.9rem 0 0.5rem;
    font-size:0.62rem; color:var(--white3);
    letter-spacing:0.05em; text-transform:uppercase;
    border-top:1px solid var(--nblue-bdr); margin-top:1.5rem;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  MATPLOTLIB THEME  — Neon White · Bright Red · Neon Blue
# ══════════════════════════════════════════════════════════════
BG_M   = '#0A0A0A'
GRID_M = '#181818'

plt.rcParams.update({
    'figure.facecolor': BG_M,  'axes.facecolor': BG_M,
    'axes.edgecolor':   GRID_M,'axes.labelcolor': '#6A8AAA',
    'axes.titlecolor':  '#F0F8FF',
    'xtick.color': '#6A8AAA',  'ytick.color': '#6A8AAA',
    'grid.color':  GRID_M,     'grid.alpha': 0.7,
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.spines.top':   False, 'axes.spines.right': False,
    'axes.spines.left':  False, 'axes.spines.bottom': False,
})

NBLUE  = '#00BFFF'
NBLUE2 = '#33CCFF'
RED    = '#FF1A1A'
RED2   = '#FF4444'
WHITE  = '#F0F8FF'

def alt_colors(n):
    """Alternate neon blue / red for bar charts"""
    return [NBLUE if i % 2 == 0 else RED2 for i in range(n)]

# ══════════════════════════════════════════════════════════════
#  LOAD DATA  (before sidebar so filters can reference it)
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
#  HEADER
# ══════════════════════════════════════════════════════════════
hc1, hc2 = st.columns([1, 7])
with hc1:
    try:
        st.image("logo.png", use_container_width=True)
    except Exception:
        st.markdown(
            "<p style='font-family:Manrope,sans-serif;font-size:1.1rem;font-weight:800;"
            "color:#00BFFF;line-height:1.3;padding:0.5rem 0;'>"
            "Woxsen<br><span style='color:#FF4444;'>University</span></p>",
            unsafe_allow_html=True
        )
with hc2:
    st.markdown(
        "<p style='font-family:Manrope,sans-serif;font-size:1.25rem;font-weight:800;"
        "color:#F0F8FF;letter-spacing:-0.02em;margin:0.3rem 0 0.12rem;'>"
        "B2B Client Risk Intelligence Dashboard</p>"
        "<p style='font-size:0.67rem;color:#4A6A8A;text-transform:uppercase;"
        "letter-spacing:0.07em;margin:0 0 0.45rem;'>"
        "AI-powered churn prediction &amp; retention analytics</p>"
        "<span style='font-size:0.63rem;font-weight:600;letter-spacing:0.06em;"
        "text-transform:uppercase;padding:0.22rem 0.6rem;border-radius:5px;"
        "border:1px solid rgba(0,191,255,0.25);color:#C8DCF0;"
        "background:#111111;margin-right:0.4rem;'>Group-2 · Rhinos</span>"
        "<span style='font-size:0.63rem;font-weight:600;letter-spacing:0.06em;"
        "text-transform:uppercase;padding:0.22rem 0.6rem;border-radius:5px;"
        "border:1px solid rgba(0,191,255,0.25);color:#C8DCF0;"
        "background:#111111;margin-right:0.4rem;'>BBA Semester 4</span>"
        "<span style='font-size:0.63rem;font-weight:600;letter-spacing:0.06em;"
        "text-transform:uppercase;padding:0.22rem 0.6rem;border-radius:5px;"
        "border:1px solid rgba(0,191,255,0.25);color:#C8DCF0;"
        "background:#111111;'>Woxsen University</span>",
        unsafe_allow_html=True
    )

st.markdown("<div style='height:1px;background:rgba(0,191,255,0.18);margin:0.5rem 0 0.8rem;'></div>",
            unsafe_allow_html=True)

if not data_ok:
    st.error(f"Could not load CSV: {data_err}. Place B2B_Client_Churn_5000.csv in the same folder.")
    st.stop()

# ══════════════════════════════════════════════════════════════
#  FILTERS — on main page inside expander (always accessible)
# ══════════════════════════════════════════════════════════════
with st.expander("🔽  Filters & Team", expanded=False):
    fa, fb, fc, fd = st.columns([2, 2, 2, 3])
    with fa:
        region      = st.multiselect("Region",        data['Region'].unique())
    with fb:
        industry    = st.multiselect("Industry",      data['Industry'].unique())
    with fc:
        risk_filter = st.multiselect("Risk Category", data['Risk_Category'].unique())
    with fd:
        st.markdown(
            "<p style='font-size:0.78rem;font-weight:700;color:#00BFFF;"
            "margin:0 0 0.3rem;'>Group-2 · Rhinos</p>"
            "<p style='font-size:0.78rem;color:#F0F8FF;line-height:1.7;margin:0;'>"
            "Mohnish Singh Patwal &nbsp;·&nbsp; Shreyas Kandi<br>"
            "Akash Krishna &nbsp;·&nbsp; Nihal Talampally</p>"
            "<p style='font-size:0.67rem;color:#4A6A8A;margin:0.3rem 0 0;'>BBA Semester 4 · Woxsen University</p>",
            unsafe_allow_html=True
        )

st.markdown("<div style='height:1px;background:rgba(0,191,255,0.18);margin:0.4rem 0 0.5rem;'></div>",
            unsafe_allow_html=True)

# ── Apply filters ────────────────────────────────────────────
filtered = data.copy()
if region:      filtered = filtered[filtered['Region'].isin(region)]
if industry:    filtered = filtered[filtered['Industry'].isin(industry)]
if risk_filter: filtered = filtered[filtered['Risk_Category'].isin(risk_filter)]

# ══════════════════════════════════════════════════════════════
#  KPI CARDS
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="slbl">Key Performance Indicators</div>', unsafe_allow_html=True)

total   = len(filtered)
hi_cnt  = (filtered['Risk_Category'] == "High Risk").sum()
churn   = round((1 - filtered['Renewal_Status'].mean()) * 100, 1)
avg_rev = round(filtered['Monthly_Revenue_USD'].mean(), 0)

st.markdown(f"""
<div class="kgrid">
  <div class="kcard kb">
    <div class="kico">👥</div>
    <div class="knum">{total:,}</div>
    <div class="klbl">Total Clients</div>
  </div>
  <div class="kcard kr">
    <div class="kico">⚠️</div>
    <div class="knum">{hi_cnt:,}</div>
    <div class="klbl">High Risk Clients</div>
  </div>
  <div class="kcard kb">
    <div class="kico">📉</div>
    <div class="knum">{churn}%</div>
    <div class="klbl">Predicted Churn Rate</div>
  </div>
  <div class="kcard kr">
    <div class="kico">💰</div>
    <div class="knum">${avg_rev:,.0f}</div>
    <div class="klbl">Avg Revenue / Client</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Gauge ────────────────────────────────────────────────────
g_cls = "glo" if churn < 20 else ("gmd" if churn < 40 else "ghi")
p_cls = "pb"  if churn < 20 else "pr"
p_txt = "● Low Risk" if churn < 20 else ("● Moderate Risk" if churn < 40 else "● Elevated Risk")

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
#  CHARTS
# ══════════════════════════════════════════════════════════════
def sfig(w=5, h=3.2):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG_M)
    ax.set_facecolor(BG_M)
    return fig, ax

st.markdown('<div class="slbl">Risk Distribution &amp; Industry Analysis</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)

# Chart 1 — Risk Distribution
with c1:
    st.markdown('<div class="panel"><div class="ptitle">Risk Category Distribution</div>'
                '<div class="psub">Client count by risk segment</div>', unsafe_allow_html=True)
    rc = filtered['Risk_Category'].value_counts().reindex(
        ["Low Risk", "Medium Risk", "High Risk"]).fillna(0)
    bar_c = [NBLUE, NBLUE2, RED]
    fig, ax = sfig()
    bars = ax.bar(rc.index, rc.values, color=bar_c, edgecolor='none', width=0.42, zorder=2)
    for b, v in zip(bars, rc.values):
        ax.text(b.get_x()+b.get_width()/2, v+15, f'{int(v):,}',
                ha='center', fontsize=9, fontweight='700', color=WHITE)
    ax.yaxis.grid(True, color=GRID_M, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True); ax.set_ylabel('Clients', fontsize=8.5)
    fig.tight_layout(pad=0.4)
    st.pyplot(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Chart 2 — Industry Risk
with c2:
    st.markdown('<div class="panel"><div class="ptitle">Industry-wise Risk Analysis</div>'
                '<div class="psub">Average risk score per industry</div>', unsafe_allow_html=True)
    ir = filtered.groupby('Industry')['Risk_Score'].mean().sort_values()
    fig, ax = sfig()
    hb = ax.barh(ir.index, ir.values,
                 color=alt_colors(len(ir)), edgecolor='none', height=0.52, zorder=2)
    for bar, v in zip(hb, ir.values):
        ax.text(v+0.04, bar.get_y()+bar.get_height()/2,
                f'{v:.1f}', va='center', fontsize=8, fontweight='600', color=WHITE)
    ax.xaxis.grid(True, color=GRID_M, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True); ax.set_xlabel('Avg Risk Score', fontsize=8.5)
    fig.tight_layout(pad=0.4)
    st.pyplot(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

c3, c4 = st.columns(2)

# Chart 3 — Revenue vs Risk Scatter
with c3:
    st.markdown('<div class="panel"><div class="ptitle">Revenue vs Risk Score</div>'
                '<div class="psub">High-revenue clients at churn risk</div>', unsafe_allow_html=True)
    samp = filtered.sample(min(600, len(filtered)), random_state=42)
    sc = samp['Risk_Category'].map(
        {"Low Risk": NBLUE2, "Medium Risk": NBLUE, "High Risk": RED})
    fig, ax = sfig()
    ax.scatter(samp['Monthly_Revenue_USD'], samp['Risk_Score'],
               c=sc, alpha=0.65, s=17, edgecolors='none', zorder=2)
    ax.yaxis.grid(True, color=GRID_M, linewidth=0.6, zorder=0)
    ax.xaxis.grid(True, color=GRID_M, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel('Monthly Revenue (USD)', fontsize=8.5)
    ax.set_ylabel('Risk Score', fontsize=8.5)
    patches = [mpatches.Patch(color=NBLUE2, label='Low'),
               mpatches.Patch(color=NBLUE,  label='Medium'),
               mpatches.Patch(color=RED,    label='High')]
    ax.legend(handles=patches, fontsize=8, framealpha=0.15,
              facecolor=BG_M, edgecolor=GRID_M, labelcolor=WHITE)
    fig.tight_layout(pad=0.4)
    st.pyplot(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Chart 4 — Contract Length vs Churn
with c4:
    st.markdown('<div class="panel"><div class="ptitle">Contract Length vs Churn Rate</div>'
                '<div class="psub">How contract duration relates to renewal behaviour</div>', unsafe_allow_html=True)
    cl = (1 - filtered.groupby('Contract_Length_Months')['Renewal_Status'].mean()) * 100
    fig, ax = sfig()
    ax.fill_between(cl.index, cl.values, alpha=0.12, color=NBLUE, zorder=1)
    ax.plot(cl.index, cl.values, color=NBLUE, linewidth=2.2, zorder=3)
    ax.yaxis.grid(True, color=GRID_M, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel('Contract Length (Months)', fontsize=8.5)
    ax.set_ylabel('Churn Rate (%)', fontsize=8.5)
    fig.tight_layout(pad=0.4)
    st.pyplot(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Chart 5 — Churn by Industry (full width)
st.markdown('<div class="panel"><div class="ptitle">Churn Rate by Industry</div>'
            '<div class="psub">Percentage of non-renewals across industry verticals</div>',
            unsafe_allow_html=True)
ci = (1 - data.groupby('Industry')['Renewal_Status'].mean()).sort_values() * 100
fig, ax = sfig(10, 3.2)
h_ci = ax.barh(ci.index, ci.values,
               color=alt_colors(len(ci)), edgecolor='none', height=0.52, zorder=2)
for bar, v in zip(h_ci, ci.values):
    ax.text(v+0.3, bar.get_y()+bar.get_height()/2,
            f'{v:.1f}%', va='center', fontsize=8, fontweight='600', color=WHITE)
ax.xaxis.grid(True, color=GRID_M, linewidth=0.6, zorder=0)
ax.set_axisbelow(True); ax.set_xlabel('Churn Rate (%)', fontsize=8.5)
fig.tight_layout(pad=0.4)
st.pyplot(fig, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# Chart 6 — Payment Delay vs Churn (full width)
st.markdown('<div class="panel"><div class="ptitle">Payment Delay vs Churn Probability</div>'
            '<div class="psub">How increasing payment delays elevate churn likelihood</div>',
            unsafe_allow_html=True)
dc = (1 - data.groupby('Payment_Delay_Days')['Renewal_Status'].mean()) * 100
fig, ax = sfig(10, 3)
ax.plot(dc.index, dc.values, color=NBLUE, linewidth=2.2, zorder=3)
ax.fill_between(dc.index, dc.values, alpha=0.1, color=NBLUE, zorder=2)
ax.axhline(40, color=RED, linestyle='--', linewidth=1.3, alpha=0.7, label='40% threshold')
ax.yaxis.grid(True, color=GRID_M, linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
ax.set_xlabel('Payment Delay (Days)', fontsize=8.5)
ax.set_ylabel('Churn Rate (%)', fontsize=8.5)
ax.legend(fontsize=8, framealpha=0.15, facecolor=BG_M, edgecolor=GRID_M, labelcolor=WHITE)
fig.tight_layout(pad=0.4)
st.pyplot(fig, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  DECISION TREE MODEL
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="slbl">Part C — Machine Learning: Decision Tree Classifier</div>',
            unsafe_allow_html=True)

FEATURES = ['Monthly_Usage_Score', 'Payment_Delay_Days', 'Contract_Length_Months',
            'Support_Tickets_Last30Days', 'Monthly_Revenue_USD']
X = data[FEATURES]; y = data['Renewal_Status']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
dt = DecisionTreeClassifier(max_depth=6, random_state=42)
dt.fit(X_tr, y_tr)
pred = dt.predict(X_te)
acc  = round(accuracy_score(y_te, pred) * 100, 1)
cm   = confusion_matrix(y_te, pred)
imp  = pd.DataFrame({'Feature': FEATURES,
                     'Importance': dt.feature_importances_}).sort_values('Importance')

c5, c6 = st.columns([1, 1.6])

# Confusion matrix
with c5:
    st.markdown('<div class="panel"><div class="ptitle">Model Performance</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div class="accb">
      <div class="accn">{acc}%</div>
      <div class="accs">Test Set Accuracy</div>
    </div>
    <span class="pill pb">✓ Decision Tree · max_depth=6</span>
    """, unsafe_allow_html=True)

    cell_bg = [['#0A1224', '#1A0A0A'], ['#0A1224', '#0A1224']]
    cell_cl = [[NBLUE, RED], [NBLUE2, NBLUE]]
    fig, ax = sfig(3.4, 2.8)
    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle(
                (j-0.5, i-0.5), 1, 1,
                color=cell_cl[i][j], alpha=0.25, zorder=1))
            ax.add_patch(plt.Rectangle(
                (j-0.5, i-0.5), 1, 1,
                fill=False, edgecolor=cell_cl[i][j], linewidth=1.5, zorder=2))
            ax.text(j, i, f'{cm[i,j]:,}', ha='center', va='center',
                    fontsize=16, fontweight='800', color='#FFFFFF', zorder=3)
    ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.5, 1.5)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred: Churn', 'Pred: Renew'], fontsize=8.5, color=WHITE, fontweight='500')
    ax.set_yticklabels(['Act: Churn',  'Act: Renew'],  fontsize=8.5, color=WHITE, fontweight='500')
    ax.set_title('Confusion Matrix', fontsize=9, color=WHITE, fontweight='700', pad=8)
    ax.tick_params(length=0)
    fig.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Feature importance
with c6:
    st.markdown('<div class="panel"><div class="ptitle">Feature Importance</div>'
                '<div class="psub">Factors ranked by predictive influence on churn</div>',
                unsafe_allow_html=True)
    hi_val = imp['Importance'].max()
    i_cols = [RED if v == hi_val else NBLUE for v in imp['Importance']]
    fig, ax = sfig(5.5, 3.4)
    hb = ax.barh(imp['Feature'], imp['Importance'],
                 color=i_cols, edgecolor='none', height=0.48, zorder=2)
    for bar, v in zip(hb, imp['Importance']):
        ax.text(v+0.003, bar.get_y()+bar.get_height()/2,
                f'{v:.3f}', va='center', fontsize=8, fontweight='600', color=WHITE)
    top_f = imp.iloc[-1]['Feature']
    ax.annotate(f'Top predictor: {top_f.replace("_"," ")}',
                xy=(hi_val, len(imp)-1), xytext=(hi_val*0.48, len(imp)-1.65),
                fontsize=7.5, color=RED,
                arrowprops=dict(arrowstyle='->', color=RED, lw=1.1))
    ax.xaxis.grid(True, color=GRID_M, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True); ax.set_xlabel('Importance Score', fontsize=8.5)
    fig.tight_layout(pad=0.4)
    st.pyplot(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Factor cards
factor_map = {
    'Monthly_Usage_Score':        ('📊', 'Monthly Usage Score',
        'Low product engagement is a strong early-warning signal of disengagement.'),
    'Payment_Delay_Days':         ('💳', 'Payment Delay Days',
        'Prolonged payment delays are a significant predictor of upcoming churn.'),
    'Contract_Length_Months':     ('📋', 'Contract Length',
        'Short-term contracts correlate with higher churn — less commitment, easier exit.'),
    'Support_Tickets_Last30Days': ('🎟️', 'Support Tickets',
        'High complaint volumes reflect service dissatisfaction and elevated risk.'),
    'Monthly_Revenue_USD':        ('💰', 'Monthly Revenue',
        'Revenue tier affects perceived value; lower-spend clients churn more.'),
}
top3 = imp.nlargest(3, 'Importance')['Feature'].tolist()
st.markdown('<div class="panel"><div class="ptitle">Churn Factor Interpretation</div>'
            '<div class="psub">Top three predictors from the Decision Tree model</div>'
            '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.6rem;margin-top:0.5rem;">',
            unsafe_allow_html=True)
for f in top3:
    ico, lbl, desc = factor_map.get(f, ('📌', f, ''))
    st.markdown(f"""
    <div class="fcrd">
      <div style="font-size:1.1rem;">{ico}</div>
      <div class="ftitle">{lbl}</div>
      <div style="font-size:0.75rem;color:#6A8AAA;">{desc}</div>
    </div>""", unsafe_allow_html=True)
st.markdown('</div></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  TWO CLIENT TABLES
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="slbl">Part D — High-Risk Client Tables</div>', unsafe_allow_html=True)

BASE_COLS = ['Client_ID', 'Industry', 'Region', 'Monthly_Revenue_USD',
             'Risk_Score', 'Risk_Category', 'Payment_Delay_Days',
             'Support_Tickets_Last30Days', 'Contract_Length_Months']
show_cols  = [c for c in BASE_COLS if c in filtered.columns]
hi_risk_df = filtered[filtered['Risk_Category'] == 'High Risk']

st.markdown(
    '<div class="tbl-lbl">Table 1 — Top 20 High-Risk Clients</div>'
    '<div class="tbl-sub">Sorted by Risk Score — highest risk first</div>',
    unsafe_allow_html=True
)
top20 = hi_risk_df.sort_values('Risk_Score', ascending=False).head(20)
if not top20.empty:
    st.dataframe(top20[show_cols].reset_index(drop=True), use_container_width=True, height=350)
else:
    st.info("No High Risk clients in the current filter selection.")

st.markdown('<div style="height:0.7rem;"></div>', unsafe_allow_html=True)

st.markdown(
    '<div class="tbl-lbl">Table 2 — High-Revenue Clients at Risk</div>'
    '<div class="tbl-sub">High Risk clients ranked by Monthly Revenue — highest commercial exposure first</div>',
    unsafe_allow_html=True
)
top_rev = hi_risk_df.sort_values('Monthly_Revenue_USD', ascending=False).head(20)
if not top_rev.empty:
    st.dataframe(top_rev[show_cols].reset_index(drop=True), use_container_width=True, height=350)
else:
    st.info("No High Risk clients in the current filter selection.")

# ══════════════════════════════════════════════════════════════
#  RETENTION STRATEGY
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="slbl">Part E — AI-Based Retention Suggestions</div>',
            unsafe_allow_html=True)

if st.button("⚡ Generate Retention Strategy"):
    st.markdown("""
    <div class="panel">
      <div class="ptitle" style="margin-bottom:0.75rem;">Strategic Retention Recommendations</div>
      <div class="ritem"><span class="rico">💳</span><span>
        <strong>Payment Relief Program:</strong> Offer structured instalment plans or
        early-payment discounts to clients with payment delays exceeding 30 days.
      </span></div>
      <div class="ritem"><span class="rico">🤝</span><span>
        <strong>Dedicated Account Management:</strong> Assign senior account managers to
        clients with more than 5 support tickets in the last 30 days.
      </span></div>
      <div class="ritem"><span class="rico">📋</span><span>
        <strong>Long-Term Contract Incentives:</strong> Provide 10–15% pricing discounts
        for clients committing to 24-month contracts.
      </span></div>
      <div class="ritem"><span class="rico">⚡</span><span>
        <strong>SLA-Backed Support Guarantees:</strong> Implement response-time SLAs for
        Tier-1 high-revenue clients to improve service reliability.
      </span></div>
      <div class="ritem"><span class="rico">📈</span><span>
        <strong>Engagement &amp; Onboarding Refresh:</strong> Launch targeted product
        training for clients with usage scores below 50.
      </span></div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  ETHICAL IMPLICATIONS
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="slbl">Part F — Ethical Implications of Predictive AI</div>',
            unsafe_allow_html=True)
st.markdown("""
<div class="panel">
  <div class="ptitle">Responsible AI in B2B Churn Prediction</div>
  <div class="psub">Key ethical dimensions when deploying predictive decision systems</div>
  <div class="egrid">
    <div class="ecard"><strong>📊 Algorithmic Bias</strong><br>
      Models trained on historical data can inherit systemic biases. Regular fairness audits
      are essential to ensure equitable treatment across all client segments.</div>
    <div class="ecard"><strong>🏷️ Impact of Risk Labelling</strong><br>
      Categorising clients as "High Risk" can affect relationships. Scores must be treated as
      internal signals, not customer-facing judgements.</div>
    <div class="ecard"><strong>🔒 Data Privacy &amp; Governance</strong><br>
      All client data must comply with GDPR, PDPA, and applicable regional data privacy
      regulations under strict governance frameworks.</div>
    <div class="ecard"><strong>🤖 Human Oversight</strong><br>
      AI predictions should augment — not replace — human decisions. Final retention actions
      require human context, empathy, and sound judgement.</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer">
  Developed by Group-2 · Rhinos · BBA Semester 4 · Woxsen University<br>
  <span style="color:#1E3050;display:block;margin-top:0.25rem;">
    Python · Streamlit · scikit-learn Decision Tree · Matplotlib
  </span>
</div>
""", unsafe_allow_html=True)
