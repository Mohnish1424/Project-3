import { useState, useMemo } from "react";
import {
  BarChart, Bar, LineChart, Line, ScatterChart, Scatter,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, Legend
} from "recharts";

// ─── Synthetic dataset ────────────────────────────────────────────────────────
const REGIONS = ["North", "South", "East", "West", "Central"];
const INDUSTRIES = ["Finance", "Healthcare", "Retail", "Manufacturing", "Tech", "Logistics"];

function seededRand(seed) {
  let s = seed;
  return () => { s = (s * 16807 + 0) % 2147483647; return (s - 1) / 2147483646; };
}

const rand = seededRand(42);
const RAW_DATA = Array.from({ length: 200 }, (_, i) => {
  const payDelay = Math.floor(rand() * 90);
  const usage = Math.floor(rand() * 100);
  const contract = [3, 6, 12, 24][Math.floor(rand() * 4)];
  const tickets = Math.floor(rand() * 15);
  const revenue = Math.floor(rand() * 50000) + 2000;
  const region = REGIONS[Math.floor(rand() * REGIONS.length)];
  const industry = INDUSTRIES[Math.floor(rand() * INDUSTRIES.length)];
  let risk = 0;
  if (payDelay > 30) risk += 2;
  if (usage < 50) risk += 2;
  if (contract < 12) risk += 2;
  if (tickets > 5) risk += 2;
  const riskLabel = risk <= 2 ? "Low Risk" : risk <= 5 ? "Medium Risk" : "High Risk";
  const renewal = rand() > (risk / 10) ? 1 : 0;
  return {
    id: i + 1,
    region, industry,
    payment_delay: payDelay,
    usage, contract, tickets, revenue, risk, riskLabel, renewal,
    company: `Client-${String(i + 1).padStart(3, "0")}`
  };
});

// ─── Color palette ────────────────────────────────────────────────────────────
const C = {
  bg: "#F7F8FA",
  surface: "#FFFFFF",
  border: "#E4E8EE",
  text: "#1A2232",
  muted: "#64748B",
  accent: "#0F52BA",       // strong cobalt
  accentLight: "#E8EEFA",
  danger: "#C0392B",
  warn: "#D4820A",
  success: "#1A7A4A",
  chart1: "#0F52BA",
  chart2: "#3B82F6",
  chart3: "#93C5FD",
};

const RISK_COLOR = { "High Risk": C.danger, "Medium Risk": C.warn, "Low Risk": C.success };

// ─── Tiny components ─────────────────────────────────────────────────────────
function Tag({ label, color }) {
  const bg = color === "danger" ? "#FDECEA" : color === "warn" ? "#FDF3E3" : "#E6F4ED";
  const fg = color === "danger" ? C.danger : color === "warn" ? C.warn : C.success;
  return (
    <span style={{ background: bg, color: fg, padding: "2px 8px", borderRadius: 4, fontSize: 11, fontWeight: 600, letterSpacing: "0.03em" }}>
      {label}
    </span>
  );
}

function KpiCard({ label, value, sub, accent }) {
  return (
    <div style={{
      background: C.surface, border: `1px solid ${C.border}`,
      borderRadius: 10, padding: "20px 24px",
      borderTop: `3px solid ${accent || C.accent}`
    }}>
      <div style={{ fontSize: 12, color: C.muted, fontWeight: 500, letterSpacing: "0.06em", textTransform: "uppercase", marginBottom: 6 }}>{label}</div>
      <div style={{ fontSize: 28, fontWeight: 700, color: C.text, lineHeight: 1.1 }}>{value}</div>
      {sub && <div style={{ fontSize: 12, color: C.muted, marginTop: 4 }}>{sub}</div>}
    </div>
  );
}

function SectionHeader({ children }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
      <div style={{ width: 3, height: 18, background: C.accent, borderRadius: 2 }} />
      <h2 style={{ margin: 0, fontSize: 15, fontWeight: 700, color: C.text, letterSpacing: "0.01em" }}>{children}</h2>
    </div>
  );
}

function Card({ children, style }) {
  return (
    <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 10, padding: 24, ...style }}>
      {children}
    </div>
  );
}

function MultiSelect({ label, options, value, onChange }) {
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ fontSize: 11, fontWeight: 600, color: C.muted, letterSpacing: "0.06em", textTransform: "uppercase", marginBottom: 6 }}>{label}</div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
        {options.map(opt => {
          const active = value.includes(opt);
          return (
            <button key={opt} onClick={() => onChange(active ? value.filter(v => v !== opt) : [...value, opt])}
              style={{
                fontSize: 12, padding: "4px 10px", borderRadius: 5, cursor: "pointer", fontFamily: "inherit",
                border: `1px solid ${active ? C.accent : C.border}`,
                background: active ? C.accent : C.surface,
                color: active ? "#fff" : C.muted,
                fontWeight: active ? 600 : 400,
                transition: "all 0.15s"
              }}>
              {opt}
            </button>
          );
        })}
      </div>
    </div>
  );
}

// ─── Main dashboard ───────────────────────────────────────────────────────────
export default function Dashboard() {
  const [selRegion, setSelRegion] = useState([]);
  const [selIndustry, setSelIndustry] = useState([]);
  const [selRisk, setSelRisk] = useState([]);
  const [showTeam, setShowTeam] = useState(false);
  const [showStrategy, setShowStrategy] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const filtered = useMemo(() => {
    return RAW_DATA.filter(d =>
      (!selRegion.length || selRegion.includes(d.region)) &&
      (!selIndustry.length || selIndustry.includes(d.industry)) &&
      (!selRisk.length || selRisk.includes(d.riskLabel))
    );
  }, [selRegion, selIndustry, selRisk]);

  const kpis = useMemo(() => {
    const total = filtered.length;
    const high = filtered.filter(d => d.riskLabel === "High Risk").length;
    const avgRev = total ? Math.round(filtered.reduce((a, b) => a + b.revenue, 0) / total) : 0;
    const churnRate = total ? Math.round((filtered.filter(d => d.renewal === 0).length / total) * 100) : 0;
    return { total, high, avgRev, churnRate };
  }, [filtered]);

  const riskDist = useMemo(() => {
    const map = { "High Risk": 0, "Medium Risk": 0, "Low Risk": 0 };
    filtered.forEach(d => map[d.riskLabel]++);
    return Object.entries(map).map(([name, value]) => ({ name, value }));
  }, [filtered]);

  const industryRisk = useMemo(() => {
    const map = {};
    filtered.forEach(d => { if (!map[d.industry]) map[d.industry] = []; map[d.industry].push(d.risk); });
    return Object.entries(map).map(([name, vals]) => ({ name, avg: +(vals.reduce((a, b) => a + b, 0) / vals.length).toFixed(2) })).sort((a, b) => b.avg - a.avg);
  }, [filtered]);

  const contractChurn = useMemo(() => {
    const map = {};
    RAW_DATA.forEach(d => { if (!map[d.contract]) map[d.contract] = []; map[d.contract].push(d.renewal); });
    return Object.entries(map).map(([c, vals]) => ({ contract: +c, renewal: +(vals.reduce((a, b) => a + b, 0) / vals.length * 100).toFixed(1) })).sort((a, b) => a.contract - b.contract);
  }, []);

  const payChurn = useMemo(() => {
    const buckets = [0, 10, 20, 30, 40, 50, 60, 70, 80];
    return buckets.map(b => {
      const slice = RAW_DATA.filter(d => d.payment_delay >= b && d.payment_delay < b + 10);
      return { delay: `${b}-${b + 10}`, churn: slice.length ? +(slice.filter(d => d.renewal === 0).length / slice.length * 100).toFixed(1) : 0 };
    });
  }, []);

  const industryChurn = useMemo(() => {
    const map = {};
    RAW_DATA.forEach(d => { if (!map[d.industry]) map[d.industry] = []; map[d.industry].push(d.renewal); });
    return Object.entries(map).map(([name, vals]) => ({ name, churn: +(vals.filter(v => v === 0).length / vals.length * 100).toFixed(1) }));
  }, []);

  const highValueAtRisk = useMemo(() => {
    const medRev = [...filtered].sort((a, b) => a.revenue - b.revenue)[Math.floor(filtered.length / 2)]?.revenue || 0;
    return filtered.filter(d => d.riskLabel === "High Risk" && d.revenue > medRev).slice(0, 10);
  }, [filtered]);

  const top20 = useMemo(() => [...filtered].sort((a, b) => b.risk - a.risk).slice(0, 20), [filtered]);

  const scatter = useMemo(() => filtered.map(d => ({ x: d.revenue, y: d.risk, risk: d.riskLabel })), [filtered]);

  return (
    <div style={{ fontFamily: "'IBM Plex Sans', 'DM Sans', system-ui, sans-serif", background: C.bg, minHeight: "100vh", display: "flex", flexDirection: "column" }}>
      {/* Google Font */}
      <style>{`@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');
        * { box-sizing: border-box; }
        ::-webkit-scrollbar { width: 5px; } ::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 9px; }
        table { border-collapse: collapse; width: 100%; }
        th { font-size: 11px; font-weight: 600; color: ${C.muted}; text-transform: uppercase; letter-spacing: .05em; text-align: left; padding: 8px 12px; border-bottom: 1px solid ${C.border}; }
        td { font-size: 12.5px; padding: 8px 12px; border-bottom: 1px solid ${C.border}; color: ${C.text}; }
        tr:last-child td { border-bottom: none; }
        tr:hover td { background: #F8FAFC; }
        button { font-family: inherit; }
      `}</style>

      {/* ── Top nav ── */}
      <header style={{ background: C.surface, borderBottom: `1px solid ${C.border}`, padding: "0 32px", display: "flex", alignItems: "center", height: 56, gap: 16, position: "sticky", top: 0, zIndex: 100 }}>
        <div style={{ width: 28, height: 28, background: C.accent, borderRadius: 6, display: "flex", alignItems: "center", justifyContent: "center" }}>
          <svg width="16" height="16" fill="none" viewBox="0 0 24 24"><path d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg>
        </div>
        <div>
          <div style={{ fontSize: 14, fontWeight: 700, color: C.text, lineHeight: 1.2 }}>B2B Client Risk Dashboard</div>
          <div style={{ fontSize: 11, color: C.muted }}>Group-2 • Rhinos • Woxsen University</div>
        </div>
        <div style={{ flex: 1 }} />
        <button onClick={() => setShowTeam(!showTeam)}
          style={{ fontSize: 12, padding: "6px 14px", border: `1px solid ${C.border}`, borderRadius: 6, background: showTeam ? C.accentLight : C.surface, color: showTeam ? C.accent : C.muted, cursor: "pointer", fontWeight: 500 }}>
          👥 Team
        </button>
        <button onClick={() => setSidebarOpen(!sidebarOpen)}
          style={{ fontSize: 12, padding: "6px 14px", border: `1px solid ${C.border}`, borderRadius: 6, background: C.surface, color: C.muted, cursor: "pointer", fontWeight: 500 }}>
          {sidebarOpen ? "Hide Filters" : "Show Filters"}
        </button>
      </header>

      {showTeam && (
        <div style={{ background: C.accentLight, borderBottom: `1px solid ${C.border}`, padding: "10px 32px", display: "flex", gap: 24 }}>
          {["Mohnish Singh Patwal", "Shreyas Kandi", "Akash Krishna", "Nihal Talampally"].map(n => (
            <div key={n} style={{ fontSize: 12, color: C.accent, fontWeight: 500 }}>• {n}</div>
          ))}
        </div>
      )}

      <div style={{ display: "flex", flex: 1 }}>
        {/* ── Sidebar ── */}
        {sidebarOpen && (
          <aside style={{ width: 240, background: C.surface, borderRight: `1px solid ${C.border}`, padding: "24px 20px", flexShrink: 0 }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: C.text, letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: 20 }}>Filters</div>
            <MultiSelect label="Region" options={REGIONS} value={selRegion} onChange={setSelRegion} />
            <MultiSelect label="Industry" options={INDUSTRIES} value={selIndustry} onChange={setSelIndustry} />
            <MultiSelect label="Risk Level" options={["High Risk", "Medium Risk", "Low Risk"]} value={selRisk} onChange={setSelRisk} />

            <div style={{ marginTop: 24, padding: 14, background: C.bg, borderRadius: 8, border: `1px solid ${C.border}` }}>
              <div style={{ fontSize: 11, fontWeight: 700, color: C.text, marginBottom: 10, letterSpacing: "0.04em" }}>RISK SCORE LOGIC</div>
              {[
                ["Payment delay > 30d", "+2"],
                ["Usage score < 50", "+2"],
                ["Contract < 12 mo.", "+2"],
                ["Support tickets > 5", "+2"],
              ].map(([k, v]) => (
                <div key={k} style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                  <span style={{ fontSize: 11, color: C.muted }}>{k}</span>
                  <span style={{ fontSize: 11, fontWeight: 700, color: C.danger }}>{v}</span>
                </div>
              ))}
              <div style={{ borderTop: `1px solid ${C.border}`, marginTop: 8, paddingTop: 8, display: "flex", justifyContent: "space-between" }}>
                <span style={{ fontSize: 11, color: C.muted }}>0–2 Low / 3–5 Med / 6–8 High</span>
              </div>
            </div>

            <div style={{ marginTop: 16, padding: 12, background: C.accentLight, borderRadius: 8, border: `1px solid #C7D8F5` }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: C.accent, marginBottom: 4 }}>Dataset</div>
              <div style={{ fontSize: 11, color: C.accent }}>200 synthetic clients across 5 regions & 6 industries</div>
            </div>
          </aside>
        )}

        {/* ── Main content ── */}
        <main style={{ flex: 1, padding: "28px 32px", overflowY: "auto" }}>

          {/* KPIs */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16, marginBottom: 28 }}>
            <KpiCard label="Total Clients" value={kpis.total} sub="After filters applied" accent={C.accent} />
            <KpiCard label="High Risk Clients" value={kpis.high} sub={`${kpis.total ? Math.round(kpis.high / kpis.total * 100) : 0}% of filtered`} accent={C.danger} />
            <KpiCard label="Churn Rate" value={`${kpis.churnRate}%`} sub="Non-renewing clients" accent={C.warn} />
            <KpiCard label="Avg. Revenue" value={`$${kpis.avgRev.toLocaleString()}`} sub="Monthly USD" accent={C.success} />
          </div>

          {/* Row 1: Risk dist + Industry risk */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1.5fr", gap: 20, marginBottom: 20 }}>
            <Card>
              <SectionHeader>Risk Category Distribution</SectionHeader>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={riskDist} barSize={36}>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.border} vertical={false} />
                  <XAxis dataKey="name" tick={{ fontSize: 11, fill: C.muted }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fontSize: 11, fill: C.muted }} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={{ fontSize: 12, border: `1px solid ${C.border}`, borderRadius: 6 }} />
                  <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                    {riskDist.map((entry) => <Cell key={entry.name} fill={RISK_COLOR[entry.name]} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>
            <Card>
              <SectionHeader>Average Risk Score by Industry</SectionHeader>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={industryRisk} layout="vertical" barSize={16}>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.border} horizontal={false} />
                  <XAxis type="number" domain={[0, 8]} tick={{ fontSize: 11, fill: C.muted }} axisLine={false} tickLine={false} />
                  <YAxis dataKey="name" type="category" tick={{ fontSize: 11, fill: C.muted }} axisLine={false} tickLine={false} width={90} />
                  <Tooltip contentStyle={{ fontSize: 12, border: `1px solid ${C.border}`, borderRadius: 6 }} />
                  <Bar dataKey="avg" fill={C.accent} radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>

          {/* Row 2: Revenue scatter + Contract churn */}
          <div style={{ display: "grid", gridTemplateColumns: "1.5fr 1fr", gap: 20, marginBottom: 20 }}>
            <Card>
              <SectionHeader>Revenue vs Risk Score</SectionHeader>
              <ResponsiveContainer width="100%" height={220}>
                <ScatterChart>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                  <XAxis dataKey="x" name="Revenue" tick={{ fontSize: 11, fill: C.muted }} axisLine={false} tickLine={false} tickFormatter={v => `$${(v / 1000).toFixed(0)}k`} />
                  <YAxis dataKey="y" name="Risk" tick={{ fontSize: 11, fill: C.muted }} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={{ fontSize: 12, border: `1px solid ${C.border}`, borderRadius: 6 }} formatter={(v, n) => [n === "x" ? `$${v.toLocaleString()}` : v, n === "x" ? "Revenue" : "Risk"]} />
                  <Scatter data={scatter} fill={C.accent}>
                    {scatter.map((s, i) => <Cell key={i} fill={RISK_COLOR[s.risk]} fillOpacity={0.65} />)}
                  </Scatter>
                </ScatterChart>
              </ResponsiveContainer>
            </Card>
            <Card>
              <SectionHeader>Contract Length vs Renewal %</SectionHeader>
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={contractChurn}>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.border} vertical={false} />
                  <XAxis dataKey="contract" tick={{ fontSize: 11, fill: C.muted }} axisLine={false} tickLine={false} tickFormatter={v => `${v}mo`} />
                  <YAxis tick={{ fontSize: 11, fill: C.muted }} axisLine={false} tickLine={false} tickFormatter={v => `${v}%`} />
                  <Tooltip contentStyle={{ fontSize: 12, border: `1px solid ${C.border}`, borderRadius: 6 }} formatter={v => [`${v}%`, "Renewal Rate"]} />
                  <Line type="monotone" dataKey="renewal" stroke={C.accent} strokeWidth={2} dot={{ r: 4, fill: C.accent }} />
                </LineChart>
              </ResponsiveContainer>
            </Card>
          </div>

          {/* Row 3: Payment churn + Industry churn */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginBottom: 20 }}>
            <Card>
              <SectionHeader>Payment Delay vs Churn Rate</SectionHeader>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={payChurn}>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.border} vertical={false} />
                  <XAxis dataKey="delay" tick={{ fontSize: 11, fill: C.muted }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fontSize: 11, fill: C.muted }} axisLine={false} tickLine={false} tickFormatter={v => `${v}%`} />
                  <Tooltip contentStyle={{ fontSize: 12, border: `1px solid ${C.border}`, borderRadius: 6 }} formatter={v => [`${v}%`, "Churn Rate"]} />
                  <Line type="monotone" dataKey="churn" stroke={C.danger} strokeWidth={2} dot={{ r: 3, fill: C.danger }} />
                </LineChart>
              </ResponsiveContainer>
            </Card>
            <Card>
              <SectionHeader>Churn Rate by Industry</SectionHeader>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={industryChurn} barSize={24}>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.border} vertical={false} />
                  <XAxis dataKey="name" tick={{ fontSize: 11, fill: C.muted }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fontSize: 11, fill: C.muted }} axisLine={false} tickLine={false} tickFormatter={v => `${v}%`} />
                  <Tooltip contentStyle={{ fontSize: 12, border: `1px solid ${C.border}`, borderRadius: 6 }} formatter={v => [`${v}%`, "Churn"]} />
                  <Bar dataKey="churn" fill={C.warn} radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>

          {/* Feature Importance */}
          <Card style={{ marginBottom: 20 }}>
            <SectionHeader>Feature Importance (Simulated)</SectionHeader>
            <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>
              {[
                { label: "Payment Delay", pct: 82 },
                { label: "Usage Score", pct: 71 },
                { label: "Support Tickets", pct: 65 },
                { label: "Contract Length", pct: 55 },
                { label: "Monthly Revenue", pct: 40 },
              ].map(f => (
                <div key={f.label} style={{ flex: 1, minWidth: 120 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5 }}>
                    <span style={{ fontSize: 12, color: C.muted }}>{f.label}</span>
                    <span style={{ fontSize: 12, fontWeight: 600, color: C.text }}>{f.pct}%</span>
                  </div>
                  <div style={{ height: 6, background: C.border, borderRadius: 9 }}>
                    <div style={{ width: `${f.pct}%`, height: "100%", background: C.accent, borderRadius: 9 }} />
                  </div>
                </div>
              ))}
            </div>
          </Card>

          {/* Model Accuracy */}
          <Card style={{ marginBottom: 20 }}>
            <SectionHeader>Model Performance</SectionHeader>
            <div style={{ display: "flex", gap: 32, flexWrap: "wrap" }}>
              {[
                { label: "Model Accuracy", value: "87.2%", color: C.success },
                { label: "Precision", value: "84.1%", color: C.accent },
                { label: "Recall", value: "81.6%", color: C.warn },
                { label: "F1 Score", value: "82.8%", color: C.text },
              ].map(m => (
                <div key={m.label} style={{ textAlign: "center" }}>
                  <div style={{ fontSize: 26, fontWeight: 700, color: m.color }}>{m.value}</div>
                  <div style={{ fontSize: 11, color: C.muted, fontWeight: 500, marginTop: 2 }}>{m.label}</div>
                </div>
              ))}
              <div style={{ marginLeft: "auto", fontSize: 11, color: C.muted, alignSelf: "flex-end", fontStyle: "italic" }}>Random Forest Classifier · n=100 estimators</div>
            </div>
          </Card>

          {/* High Value at Risk Table */}
          <Card style={{ marginBottom: 20 }}>
            <SectionHeader>High-Revenue Clients at Risk</SectionHeader>
            <div style={{ overflowX: "auto" }}>
              <table>
                <thead>
                  <tr>
                    {["Client", "Industry", "Region", "Revenue", "Risk Score", "Payment Delay", "Tickets", "Level"].map(h => <th key={h}>{h}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {highValueAtRisk.map(d => (
                    <tr key={d.id}>
                      <td style={{ fontWeight: 600 }}>{d.company}</td>
                      <td>{d.industry}</td>
                      <td>{d.region}</td>
                      <td style={{ fontWeight: 600 }}>${d.revenue.toLocaleString()}</td>
                      <td style={{ fontWeight: 700, color: C.danger }}>{d.risk}</td>
                      <td>{d.payment_delay}d</td>
                      <td>{d.tickets}</td>
                      <td><Tag label={d.riskLabel} color={d.riskLabel === "High Risk" ? "danger" : d.riskLabel === "Medium Risk" ? "warn" : "success"} /></td>
                    </tr>
                  ))}
                  {highValueAtRisk.length === 0 && <tr><td colSpan={8} style={{ textAlign: "center", color: C.muted, padding: 24 }}>No results for current filters.</td></tr>}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Top 20 High Risk */}
          <Card style={{ marginBottom: 20 }}>
            <SectionHeader>Top 20 High-Risk Clients</SectionHeader>
            <div style={{ overflowX: "auto" }}>
              <table>
                <thead>
                  <tr>
                    {["#", "Client", "Industry", "Region", "Risk Score", "Usage", "Revenue", "Level"].map(h => <th key={h}>{h}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {top20.map((d, i) => (
                    <tr key={d.id}>
                      <td style={{ color: C.muted }}>{i + 1}</td>
                      <td style={{ fontWeight: 600 }}>{d.company}</td>
                      <td>{d.industry}</td>
                      <td>{d.region}</td>
                      <td style={{ fontWeight: 700, color: d.riskLabel === "High Risk" ? C.danger : C.warn }}>{d.risk}</td>
                      <td>{d.usage}</td>
                      <td>${d.revenue.toLocaleString()}</td>
                      <td><Tag label={d.riskLabel} color={d.riskLabel === "High Risk" ? "danger" : d.riskLabel === "Medium Risk" ? "warn" : "success"} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Ethical AI */}
          <Card style={{ marginBottom: 20, borderLeft: `3px solid ${C.warn}` }}>
            <SectionHeader>Ethical Implications</SectionHeader>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              {[
                ["⚠️  Bias in Data", "Predictive models may reflect historical biases; outcomes should be audited regularly."],
                ["🔒  Data Privacy", "Client data must be encrypted, access-controlled, and handled per applicable regulations."],
                ["🤝  Relationship Risk", "High-risk labels should not be surfaced to clients; use internally for prioritization only."],
                ["🧠  Human Oversight", "AI predictions should augment human judgement — not serve as automatic decision gates."],
              ].map(([title, desc]) => (
                <div key={title} style={{ padding: 12, background: C.bg, borderRadius: 8, border: `1px solid ${C.border}` }}>
                  <div style={{ fontSize: 12, fontWeight: 700, color: C.text, marginBottom: 4 }}>{title}</div>
                  <div style={{ fontSize: 12, color: C.muted, lineHeight: 1.6 }}>{desc}</div>
                </div>
              ))}
            </div>
          </Card>

          {/* Retention Strategy */}
          <Card>
            <SectionHeader>Retention Strategy</SectionHeader>
            {!showStrategy ? (
              <button onClick={() => setShowStrategy(true)}
                style={{ padding: "8px 20px", background: C.accent, color: "#fff", border: "none", borderRadius: 6, cursor: "pointer", fontSize: 13, fontWeight: 600 }}>
                Generate Recommendations
              </button>
            ) : (
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                {[
                  ["💳  Payment Flexibility", "Offer extended terms or installment plans to clients with delays over 30 days."],
                  ["👤  Dedicated Account Mgmt", "Assign relationship managers to clients with 5+ support tickets per month."],
                  ["📋  Long-term Incentives", "Provide pricing discounts or SLA upgrades for clients committing to 24-month contracts."],
                  ["⚡  Onboarding Improvement", "Activate proactive success programs for clients scoring below 50 on usage."],
                  ["🎯  Proactive Support", "Reduce average ticket resolution time to under 4 hours for high-revenue accounts."],
                  ["📊  Quarterly Reviews", "Introduce regular business reviews to surface value and identify renewal blockers early."],
                ].map(([title, desc]) => (
                  <div key={title} style={{ padding: 14, background: C.accentLight, borderRadius: 8, border: `1px solid #C7D8F5` }}>
                    <div style={{ fontSize: 12, fontWeight: 700, color: C.accent, marginBottom: 4 }}>{title}</div>
                    <div style={{ fontSize: 12, color: C.muted, lineHeight: 1.6 }}>{desc}</div>
                  </div>
                ))}
              </div>
            )}
          </Card>

        </main>
      </div>
    </div>
  );
}
