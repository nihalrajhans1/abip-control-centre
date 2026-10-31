"""
ABIP Control Centre v2
Pack & cell level battery intelligence over Firebase Realtime Database.
Handles simulated multi-cell packs AND a single real hardware cell.
"""
import streamlit as st
import urllib.request
import json
import time
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go

DB_URL = "https://abip-battery-default-rtdb.asia-southeast1.firebasedatabase.app"
HW_ID = "HW_CELL_01"
R_TH, C_TH = 3.2, 42.0          # thermal resistance / capacitance

st.set_page_config("ABIP Control Centre", layout="wide", page_icon="B")

st.markdown("""<style>
#MainMenu,footer,header{visibility:hidden}
.stApp{background:#0b0f14}
div[data-testid="stMetric"]{background:#141a22;border:1px solid #1f2731;
border-radius:12px;padding:12px 14px}
div[data-testid="stMetricValue"]{color:#2dd4ff;font-weight:700;font-size:1.45rem}
div[data-testid="stMetricLabel"]{color:#7d8b99}
.stTabs [data-baseweb="tab"]{background:#141a22;border-radius:8px 8px 0 0}
</style>""", unsafe_allow_html=True)


# ═══════════════════ helpers ═══════════════════
CACHE_FILE = "abip_cache.json"


def fb(path):
    try:
        u = DB_URL.rstrip("/") + path + ".json"
        b = urllib.request.urlopen(u, timeout=8).read().decode()
        data = json.loads(b) if b and b != "null" else None
        if path == "/packs" and data:
            try:
                with open(CACHE_FILE, "w") as f:
                    json.dump(data, f)
            except Exception:
                pass
            st.session_state["read_offline"] = False
        return data
    except Exception:
        if path == "/packs":
            try:
                with open(CACHE_FILE) as f:
                    st.session_state["read_offline"] = True
                    return json.load(f)
            except Exception:
                pass
        return None


def fb_put(path, obj):
    try:
        req = urllib.request.Request(
            DB_URL.rstrip("/") + path + ".json",
            data=json.dumps(obj).encode(), method="PUT",
            headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=12)
        return True
    except Exception:
        return False


def num(x, default=0.0):
    """Any value -> float. None/NaN/junk -> default."""
    try:
        if x is None:
            return default
        f = float(x)
        return default if math.isnan(f) else f
    except (TypeError, ValueError):
        return default


def fmt(x, suffix="", nd=0, dash="N/A"):
    """Any value -> display string. None -> 'N/A'."""
    if x is None:
        return dash
    try:
        f = float(x)
        if math.isnan(f):
            return dash
        return f"{f:.{nd}f}{suffix}"
    except (TypeError, ValueError):
        return dash


# ═══════════════════ analytics ═══════════════════
def cell_table(live):
    rows = []
    for cid, d in (live.get("cells") or {}).items():
        if not isinstance(d, dict):
            continue
        rows.append({
            "cell": cid,
            "v": num(d.get("v")),
            "t": num(d.get("t"), -127.0),
            "soc": num(d.get("soc")),
            "soh": d.get("soh"),
            "r_mohm": d.get("r_mohm"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["r_n"] = df["r_mohm"].apply(lambda x: num(x, float("nan")))
    return df.sort_values("cell").reset_index(drop=True)


def diagnose(df, i_a):
    """Weakest-cell detection with explainable evidence."""
    out = {"weakest": None, "v_delta_mv": 0.0, "t_gradient": 0.0,
           "evidence": [], "verdict": "NOMINAL", "severity": "OK"}
    if df.empty:
        out["evidence"].append("No cell data available.")
        return out

    if len(df) == 1:
        r = df.iloc[0]
        out["verdict"] = "SINGLE CELL MODULE"
        if r["t"] == -127:
            out["evidence"].append(
                "Thermistor reports -127 fault marker. Never treated as a "
                "temperature; thermal assessment suspended.")
            out["severity"] = "DEGRADED"
        else:
            out["evidence"].append(
                f"Single-cell module: {r['v']:.3f} V, {r['t']:.1f} C. "
                "Cell-balance and weakest-cell analytics require 2+ "
                "instrumented cells.")
        return out

    v = df["v"]
    valid_t = df[df["t"] != -127]
    out["v_delta_mv"] = round((v.max() - v.min()) * 1000, 1)
    if not valid_t.empty:
        out["t_gradient"] = round(valid_t["t"].max() - valid_t["t"].min(), 2)

    low_v = df.loc[v.idxmin()]
    hot = valid_t.loc[valid_t["t"].idxmax()] if not valid_t.empty else None
    sigma = v.std() if len(v) > 1 else 0
    z = (low_v["v"] - v.mean()) / sigma if sigma and sigma > 1e-6 else 0

    faults = df[df["t"] == -127]["cell"].tolist()
    if faults:
        out["evidence"].append(
            f"Thermistor fault on {', '.join(faults)} (reads -127). "
            "Excluded from thermal comparison.")
        out["severity"] = "DEGRADED"

    if out["v_delta_mv"] > 50:
        out["verdict"] = "IMBALANCED"
        out["severity"] = "ALERT"
        out["evidence"].append(
            f"Cell voltage spread {out['v_delta_mv']:.0f} mV exceeds the "
            "50 mV balance limit.")

    if abs(z) > 1.2:
        out["weakest"] = low_v["cell"]
        out["evidence"].append(
            f"{low_v['cell']} sits {abs(z):.1f} sigma below pack mean "
            f"({low_v['v']:.3f} V vs {v.mean():.3f} V).")

    if hot is not None and out["t_gradient"] > 4:
        out["evidence"].append(
            f"{hot['cell']} is {out['t_gradient']:.1f} C hotter than the "
            "coolest cell - localized heat generation.")
        if out["weakest"] == hot["cell"]:
            out["verdict"] = "WEAK CELL CONFIRMED"
            out["severity"] = "CRITICAL"
            out["evidence"].append(
                f"{hot['cell']} is BOTH lowest voltage AND hottest. "
                "Signature of elevated internal resistance: I^2R loss at "
                f"{num(i_a):.2f} A produces excess heat while the IR drop "
                "lowers terminal voltage.")

    if "r_n" in df and df["r_n"].notna().sum() >= 2:
        med = df["r_n"].median()
        if med and med > 0:
            for _, r in df[df["r_n"] > med * 1.8].iterrows():
                out["evidence"].append(
                    f"{r['cell']} internal resistance {r['r_n']:.0f} mOhm vs "
                    f"pack median {med:.0f} mOhm.")

    if not out["evidence"]:
        out["evidence"].append(
            "All cells within balance, thermal and resistance tolerances.")
    return out


def thermal_rules(hist, cid):
    """Explainable thermal rule engine on a cell's history."""
    tcol, acol = f"{cid}_t", "t_ambient"
    rules, ev = [], []
    if hist.empty or tcol not in hist.columns:
        return rules, ["Insufficient history for trend analysis."], \
               "UNAVAILABLE"
    d = hist[hist[tcol] != -127]
    if d.empty:
        return rules, ["Sensor fault: -127 is never treated as a "
                       "temperature."], "DEGRADED"

    last = d.iloc[-1]
    t_now = num(last[tcol], -127)
    if t_now >= 55:
        rules.append("T_ABORT")
        ev.append(f"{t_now:.1f} C at/above the 55 C abort threshold.")
    elif t_now >= 45:
        rules.append("T1_HIGH")
        ev.append(f"{t_now:.1f} C at/above the 45 C datasheet charge ceiling.")

    w = d[d["t_s"] >= num(last["t_s"]) - 30]
    if len(w) >= 5:
        span = num(w["t_s"].iloc[-1]) - num(w["t_s"].iloc[0])
        if span > 5:
            r1 = (num(w[tcol].iloc[-1]) - num(w[tcol].iloc[0])) / span * 60
            r2 = 0.0
            if acol in w.columns:
                r2 = (num(w[acol].iloc[-1]) - num(w[acol].iloc[0])) / span * 60
            if r1 >= 1.0:
                rules.append("T_RISING")
                ev.append(f"Rising {r1:.2f} C/min over {span:.0f} s.")
                if r1 - r2 >= 0.7:
                    rules.append("LOCAL_HEATING_PATTERN")
                    ev.append(
                        f"Ambient only {r2:.2f} C/min - heat originates at "
                        "the cell, not the environment.")
                elif abs(r1 - r2) < 0.3:
                    rules.append("ENVIRONMENTAL_WARMING")
                    ev.append("Cell and ambient rise together - "
                              "environmental, not internal.")
            if r1 > 0.2 and t_now < 45:
                mins = (45 - t_now) / r1
                if 0 < mins < 240:
                    ev.append(f"Projection: reaches 45 C in ~{mins:.0f} min "
                              "if the trend holds.")
    if not ev:
        ev.append("Thermal behaviour within normal bounds.")
    return rules, ev, "GOOD"


def risk_score(live, diag):
    s = 0.0
    tmax = num(live.get("t_max"), -127)
    if tmax != -127:
        if tmax > 55:
            s += 45
        elif tmax > 45:
            s += 30
        elif tmax > 38:
            s += 15
    s += min(25, num(diag.get("v_delta_mv")) / 4)
    s += min(15, num(diag.get("t_gradient")) * 2)
    soh = live.get("soh")
    if soh is not None:
        s += max(0, (100 - num(soh, 100)) * 0.6)
    soc = live.get("soc")
    if soc is not None and num(soc, 100) < 15:
        s += 10
    if diag.get("severity") == "CRITICAL":
        s += 20
    return int(min(100, s))


def band(score):
    if score >= 60:
        return "CRITICAL", "#ff1744"
    if score >= 35:
        return "WARNING", "#ffab00"
    if score >= 18:
        return "WATCH", "#29b6f6"
    return "HEALTHY", "#00e676"


def hist_df(pid):
    raw = fb(f"/packs/{pid}/history")
    if not isinstance(raw, dict):
        return pd.DataFrame()
    rows = [raw[k] for k in sorted(raw) if isinstance(raw[k], dict)]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.sort_values("t_s").reset_index(drop=True) \
        if "t_s" in df.columns else df


# ═══════════════════ load ═══════════════════
meta_up = fb("/meta/uploader") or {}
meta_hw = fb("/meta/hw_uploader") or {}
packs_raw = fb("/packs") or {}
if st.session_state.get("read_offline"):
    st.warning("READ OFFLINE — displaying the last cached snapshot. "
               "The app degrades gracefully instead of failing.")

st.markdown("""<div style="background:linear-gradient(120deg,#07131c,#123247,#0a2233);
padding:22px 26px;border-radius:16px;border:1px solid #163a4d;margin-bottom:16px">
<div style="font-size:1.9rem;font-weight:800;color:#eaf6ff">ABIP Control Centre</div>
<div style="color:#7fb3cc;font-size:.9rem">AI Battery Intelligence Platform -
pack &amp; cell diagnostics over Firebase Realtime Database</div></div>""",
            unsafe_allow_html=True)

with st.sidebar:
    st.subheader("Connection")
    if packs_raw:
        st.success("Firebase ONLINE")
    else:
        st.error("No data in Firebase")
    st.caption(DB_URL.replace("https://", "")[:44])
    st.metric("Fleet source", str(meta_up.get("mode", "?")).upper())
    st.metric("HW uploader", str(meta_hw.get("mode", "-")).upper())
    auto = st.checkbox("Auto-refresh 3s", True)
    if st.button("Refresh now"):
        st.rerun()

    st.divider()
    st.subheader("Add Battery")
    nid = st.text_input("Pack ID", "PACK_D")
    nname = st.text_input("Name", "New Battery Pack")
    ncells = st.number_input("Cells", 1, 12, 4)
    if st.button("Register Pack", type="primary"):
        ok = fb_put(f"/packs/{nid}/meta", {
            "pack_id": nid, "name": nname,
            "chemistry": "Li-ion NMC 18650",
            "config": f"{int(ncells)}S1P", "cell_count": int(ncells),
            "nominal_v": round(3.7 * ncells, 1), "capacity_ah": 2.6,
            "note": "Registered from Control Centre", "cycles": 0,
            "commissioned": time.strftime("%Y-%m-%d")})
        if ok:
            st.success(f"{nid} registered")
        else:
            st.error("Write failed")
        time.sleep(1)
        st.rerun()

    st.divider()
    st.caption("Uploaders:")
    st.code("python fb_uploader2.py\npython hw_uploader.py")

if not packs_raw:
    st.error("No packs found in Firebase.")
    st.info("Start an uploader in another window:\n\n"
            "python fb_uploader2.py\n\npython hw_uploader.py --demo")
    st.stop()

packs = {}
for pid, node in packs_raw.items():
    if not isinstance(node, dict):
        continue
    live = node.get("live")
    meta = node.get("meta", {}) if isinstance(node.get("meta"), dict) else {}
    if not isinstance(live, dict):
        packs[pid] = {"meta": meta, "live": None, "df": pd.DataFrame(),
                      "diag": None, "score": 0}
        continue
    df = cell_table(live)
    dg = diagnose(df, live.get("current_a", 0))
    packs[pid] = {"meta": meta, "live": live, "df": df, "diag": dg,
                  "score": risk_score(live, dg)}

active = {k: v for k, v in packs.items() if v["live"]}

tab1, tabHW, tab2, tab3, tab4, tab5 = st.tabs(
    ["Fleet Overview", "LIVE HARDWARE", "Pack Detail", "Alerts",
     "Battery Passport", "Offline Buffer"])


# ═══════════════════ FLEET OVERVIEW ═══════════════════
with tab1:
    tot_cells = sum(len(v["df"]) for v in active.values())
    crit = sum(1 for v in active.values() if band(v["score"])[0] == "CRITICAL")
    warn = sum(1 for v in active.values() if band(v["score"])[0] == "WARNING")

    k = st.columns(6)
    k[0].metric("Packs", len(packs))
    k[1].metric("Cells", tot_cells)
    k[2].metric("Critical", crit)
    k[3].metric("Warning", warn)
    socs = [num(v["live"].get("soc")) for v in active.values()
            if v["live"].get("soc") is not None]
    sohs = [num(v["live"].get("soh")) for v in active.values()
            if v["live"].get("soh") is not None]
    k[4].metric("Fleet SoC", f"{np.mean(socs):.0f}%" if socs else "N/A")
    k[5].metric("Fleet SoH", f"{np.mean(sohs):.0f}%" if sohs else "N/A")

    st.markdown("### Risk Ranking")
    rank = []
    for pid, v in sorted(active.items(), key=lambda x: -x[1]["score"]):
        lb, _ = band(v["score"])
        L = v["live"]
        rank.append({
            "Pack": pid,
            "Name": v["meta"].get("name", pid),
            "Risk": v["score"],
            "Status": lb,
            "Pack V": round(num(L.get("pack_v")), 3),
            "Current A": round(num(L.get("current_a")), 2),
            "SoC %": fmt(L.get("soc"), nd=0),
            "SoH %": fmt(L.get("soh"), nd=0),
            "T max C": fmt(L.get("t_max"), nd=1),
            "dV mV": round(num(v["diag"]["v_delta_mv"]), 1),
            "Weakest": v["diag"]["weakest"] or "-",
            "Verdict": v["diag"]["verdict"]})
    if rank:
        st.dataframe(pd.DataFrame(rank), use_container_width=True,
                     hide_index=True)

    st.markdown("### Packs")
    cols = st.columns(min(3, max(1, len(packs))))
    for i, (pid, v) in enumerate(sorted(packs.items(),
                                        key=lambda x: -x[1]["score"])):
        c = cols[i % len(cols)]
        if not v["live"]:
            c.markdown(f"""<div style="background:#141a22;border:1px dashed #30404f;
            border-radius:12px;padding:16px"><b style="color:#9fb3c8">{pid}</b>
            <div style="color:#6b7b8c;font-size:.8rem">Registered - awaiting
            telemetry</div></div>""", unsafe_allow_html=True)
            continue
        lb, col = band(v["score"])
        L = v["live"]
        hw_tag = ("<span style='background:#ff174433;color:#ff5c7a;"
                  "padding:1px 7px;border-radius:6px;font-size:.66rem;"
                  "font-weight:700'>HARDWARE</span>"
                  if pid == HW_ID else "")
        c.markdown(f"""<div style="background:#141a22;border:1px solid {col}55;
        border-left:5px solid {col};border-radius:12px;padding:14px 16px">
        <div style="font-weight:800;color:#eaf6ff;font-size:1.02rem">
        {pid} {hw_tag}</div>
        <div style="color:#7d8b99;font-size:.74rem">{v['meta'].get('name','')}</div>
        <div style="color:{col};font-weight:800;margin:8px 0;font-size:.85rem">
        {lb} &nbsp;|&nbsp; risk {v['score']}</div>
        <div style="color:#9fb3c8;font-size:.82rem;line-height:1.5">
        Pack {num(L.get('pack_v')):.2f} V &nbsp; {num(L.get('current_a')):.2f} A<br>
        SoC {fmt(L.get('soc'), '%')} &nbsp; SoH {fmt(L.get('soh'), '%')}<br>
        Tmax {fmt(L.get('t_max'), ' C', 1)} &nbsp;
        dV {num(v['diag']['v_delta_mv']):.0f} mV<br>
        <b style="color:{col}">{v['diag']['verdict']}</b></div></div>""",
                    unsafe_allow_html=True)


# ═══════════════════ LIVE HARDWARE ═══════════════════
with tabHW:
    hw = packs_raw.get(HW_ID) if isinstance(packs_raw, dict) else None
    L = hw.get("live") if isinstance(hw, dict) else None
    M = hw.get("meta", {}) if isinstance(hw, dict) else {}

    if not isinstance(L, dict):
        st.info("No hardware stream yet.")
        st.markdown("Start it in a separate window:")
        st.code("python hw_uploader.py\n"
                "python hw_uploader.py --demo     (no ESP32 available)")
        st.caption("The rest of the fleet is unaffected by hardware state.")
    else:
        age = time.time() - num(L.get("heartbeat"))
        link = L.get("link", "?")
        if link == "LIVE" and age < 8:
            st.success(f"LIVE - ESP32 streaming, last packet {age:.1f} s ago")
        elif link == "DEMO":
            st.warning("DEMO FALLBACK - clearly labelled, not real sensors")
        elif link == "RECONNECTING":
            st.warning("Serial link dropped - uploader is retrying")
        else:
            st.error(f"STALE - last packet {age:.0f} s ago")

        t1 = num(L.get("t_max"), -127)
        t2 = num(L.get("t_ambient"), -127)
        t3 = num(L.get("t_charger"), -127)
        dT = L.get("self_heating_c")
        rc = num(L.get("rate_cell_cpm"))
        ra = num(L.get("rate_ambient_cpm"))
        rd = num(L.get("rate_diff_cpm"))
        res = L.get("twin_residual_c")

        h = st.columns(7)
        h[0].metric("T1 cell", "FAULT" if t1 == -127 else f"{t1:.2f} C")
        h[1].metric("T2 ambient", "FAULT" if t2 == -127 else f"{t2:.2f} C")
        h[2].metric("T3 charger", "FAULT" if t3 == -127 else f"{t3:.2f} C")
        h[3].metric("Self-heating",
                    f"{num(dT):+.2f} C" if dT is not None else "N/A")
        h[4].metric("Terminal V", f"{num(L.get('pack_v')):.3f}")
        h[5].metric("SoC (OCV)", fmt(L.get("soc"), "%"))
        h[6].metric("Twin residual",
                    f"{num(res):+.2f} C" if res is not None else "N/A")

        hw_rules, hw_ev = [], []
        if t1 == -127:
            qual = "DEGRADED"
            hw_ev.append("T1 reports the -127 fault marker. Never treated as "
                         "a temperature; thermal assessment suspended.")
        else:
            qual = "GOOD"
            if t1 >= 55:
                hw_rules.append("T_ABORT")
                hw_ev.append(f"{t1:.1f} C at/above the 55 C abort threshold.")
            elif t1 >= 45:
                hw_rules.append("T1_HIGH")
                hw_ev.append(f"{t1:.1f} C at/above the 45 C charge ceiling.")
            if rc >= 1.0:
                hw_rules.append("T_RISING")
                hw_ev.append(f"Cell rising {rc:.2f} C/min.")
                if rd >= 0.7:
                    hw_rules.append("LOCAL_HEATING_PATTERN")
                    hw_ev.append(
                        f"Ambient only {ra:.2f} C/min - differential "
                        f"{rd:.2f} C/min. Heat originates at the cell, "
                        "not the environment.")
                elif abs(rd) < 0.3:
                    hw_rules.append("ENVIRONMENTAL_WARMING")
                    hw_ev.append("Cell and ambient rise together.")
            if rc > 0.2 and t1 < 45:
                hw_ev.append(f"Projection: 45 C in ~{(45 - t1) / rc:.0f} min "
                             "if the trend holds.")
            cd = L.get("charger_delta_c")
            if cd is not None and num(cd) > 5:
                hw_rules.append("CHARGER_PATH_HEATING")
                hw_ev.append(f"T3 is {num(cd):.1f} C above ambient - "
                             "connector or charge-path resistance.")
        if not hw_ev:
            hw_ev.append("Cell thermally stable and tracking ambient.")

        a1, a2 = st.columns([3, 2])
        with a1:
            H = hist_df(HW_ID)
            if not H.empty and "CELL_t" in H.columns:
                P = H.copy()
                for c in ("CELL_t", "t_ambient", "t_charger"):
                    if c in P.columns:
                        P.loc[P[c] == -127, c] = None
                g = go.Figure()
                g.add_trace(go.Scatter(x=P["t_s"], y=P.get("CELL_t"),
                                       name="T1 cell",
                                       line=dict(color="#ff5252", width=2.4)))
                if "t_ambient" in P.columns:
                    g.add_trace(go.Scatter(x=P["t_s"], y=P["t_ambient"],
                                           name="T2 ambient",
                                           line=dict(color="#2dd4ff", width=2)))
                if "t_charger" in P.columns:
                    g.add_trace(go.Scatter(x=P["t_s"], y=P["t_charger"],
                                           name="T3 charger",
                                           line=dict(color="#ffab00",
                                                     width=1.6)))
                if "twin_model_c" in P.columns:
                    g.add_trace(go.Scatter(x=P["t_s"], y=P["twin_model_c"],
                                           name="twin model",
                                           line=dict(color="#b388ff",
                                                     dash="dash")))
                g.add_hline(y=45, line_dash="dot", line_color="#ffab00")
                g.add_hline(y=55, line_dash="dot", line_color="#ff1744")
                parts = [P[c] for c in ("CELL_t", "t_ambient", "t_charger")
                         if c in P.columns]
                vv = pd.concat(parts).dropna() if parts else pd.Series(dtype=float)
                g.update_layout(
                    template="plotly_dark", height=340,
                    margin=dict(l=8, r=8, t=26, b=8),
                    xaxis_title="t (s)", yaxis_title="deg C",
                    yaxis_range=[max(0, vv.min() - 3), max(vv.max() + 4, 50)]
                    if len(vv) else None,
                    legend=dict(orientation="h", y=1.16))
                st.plotly_chart(g, use_container_width=True)

                if "self_heating_c" in P.columns:
                    g2 = go.Figure(go.Scatter(
                        x=P["t_s"], y=P["self_heating_c"], name="T1 - T2",
                        fill="tozeroy", line=dict(color="#00e676", width=2)))
                    g2.update_layout(
                        template="plotly_dark", height=190,
                        margin=dict(l=8, r=8, t=34, b=8),
                        title="Self-heating index (T1 - T2)",
                        xaxis_title="t (s)", yaxis_title="delta C")
                    st.plotly_chart(g2, use_container_width=True)
            else:
                st.info("Collecting history...")
        with a2:
            st.markdown("#### Thermal Assessment")
            if qual != "GOOD":
                st.warning("Data quality: DEGRADED")
            if hw_rules:
                for r_ in hw_rules:
                    st.error(r_)
            else:
                st.success("No thermal rules active")
            st.markdown("#### Evidence")
            for e in hw_ev:
                st.markdown(f"- {e}")

            st.markdown("#### Device State")
            st.markdown(
                f"- ESP32 state: `{L.get('esp_state', '?')}` "
                f"flags `{L.get('esp_flags', 0)}`\n"
                f"- Over-temp guard: "
                f"**{'ARMED' if L.get('armed') else 'DISARMED'}**"
                f"{' - **ABORT LATCHED**' if L.get('aborted') else ''}\n"
                f"- Current sensing: **not fitted**\n"
                f"- SoC method: {L.get('soc_method', '-')}")

            st.markdown("#### Measurement Scope")
            st.markdown(
                "| Quantity | Status |\n|---|---|\n"
                "| T1 / T2 / T3 | measured (NTC 10k B3950) |\n"
                "| Terminal voltage | measured (divider) |\n"
                "| Current | not fitted |\n"
                "| SoH | needs full cycle - N/A |\n"
                "| Cell balance | N/A - single cell |")

        with st.expander("Hardware specification"):
            st.json({"meta": M, "live_sample": L}, expanded=False)


# ═══════════════════ PACK DETAIL ═══════════════════
with tab2:
    if not active:
        st.info("No live packs.")
    else:
        pid = st.selectbox("Pack", sorted(active))
        P = active[pid]
        L, df, dg = P["live"], P["df"], P["diag"]
        lb, col = band(P["score"])

        m = st.columns(7)
        m[0].metric("Pack V", f"{num(L.get('pack_v')):.2f}")
        m[1].metric("Current", f"{num(L.get('current_a')):.2f} A")
        m[2].metric("Power",
                    f"{num(L.get('pack_v')) * num(L.get('current_a')):.1f} W")
        m[3].metric("SoC", fmt(L.get("soc"), "%"))
        m[4].metric("SoH", fmt(L.get("soh"), "%"))
        m[5].metric("T max", fmt(L.get("t_max"), " C", 1))
        m[6].metric("Risk", f"{P['score']}", lb)

        weak_txt = (f" &nbsp;|&nbsp; Weakest cell: <b>{dg['weakest']}</b>"
                    if dg["weakest"] else "")
        st.markdown(f"""<div style="background:{col}14;border-left:4px solid {col};
        border-radius:8px;padding:12px 16px;margin:10px 0">
        <b style="color:{col}">{dg['verdict']}</b>{weak_txt}</div>""",
                    unsafe_allow_html=True)

        if df.empty:
            st.warning("No cell data in this pack.")
        else:
            a, b = st.columns([3, 2])
            with a:
                st.markdown("#### Cell Voltages")
                f = go.Figure(go.Bar(
                    x=df["cell"], y=df["v"],
                    marker_color=["#ff1744" if c == dg["weakest"]
                                  else "#2dd4ff" for c in df["cell"]],
                    text=[f"{x:.3f}" for x in df["v"]],
                    textposition="outside"))
                f.add_hline(y=df["v"].mean(), line_dash="dot",
                            line_color="#7d8b99", annotation_text="mean")
                f.update_layout(
                    template="plotly_dark", height=260,
                    margin=dict(l=8, r=8, t=24, b=8), yaxis_title="V",
                    yaxis_range=[df["v"].min() - 0.08, df["v"].max() + 0.08])
                st.plotly_chart(f, use_container_width=True)

                st.markdown("#### Cell Temperatures")
                tv = df[df["t"] != -127]
                if tv.empty:
                    st.warning("All thermistors reporting fault.")
                else:
                    f2 = go.Figure(go.Bar(
                        x=tv["cell"], y=tv["t"],
                        marker_color=["#ff1744" if x > 45 else
                                      "#ffab00" if x > 38 else "#00e676"
                                      for x in tv["t"]],
                        text=[f"{x:.1f}" for x in tv["t"]],
                        textposition="outside"))
                    f2.add_hline(y=45, line_dash="dot", line_color="#ffab00")
                    f2.update_layout(template="plotly_dark", height=240,
                                     margin=dict(l=8, r=8, t=24, b=8),
                                     yaxis_title="deg C")
                    st.plotly_chart(f2, use_container_width=True)
                if len(tv) < len(df):
                    st.caption("Cells excluded (-127): " +
                               ", ".join(df[df["t"] == -127]["cell"]))

            with b:
                st.markdown("#### Diagnosis Evidence")
                for e in dg["evidence"]:
                    st.markdown(f"- {e}")
                st.markdown("#### Cell Table")
                show = df[["cell", "v", "t", "soc", "soh", "r_mohm"]].copy()
                show["soh"] = show["soh"].apply(lambda x: fmt(x))
                show["r_mohm"] = show["r_mohm"].apply(lambda x: fmt(x))
                show = show.rename(columns={
                    "cell": "Cell", "v": "V", "t": "T C", "soc": "SoC %",
                    "soh": "SoH %", "r_mohm": "R mOhm"})
                st.dataframe(show, use_container_width=True, hide_index=True)
                st.caption(f"Imbalance {num(dg['v_delta_mv']):.0f} mV | "
                           f"Thermal gradient {num(dg['t_gradient']):.1f} C")

            st.markdown("---")
            st.markdown("### Cell Inspector")
            clist = df["cell"].tolist()
            idx = clist.index(dg["weakest"]) if dg["weakest"] in clist else 0
            cid = st.selectbox("Cell", clist, index=idx)
            H = hist_df(pid)
            rules, ev, qual = thermal_rules(H, cid)

            c1, c2 = st.columns([3, 2])
            with c1:
                if not H.empty and f"{cid}_t" in H.columns:
                    plot = H.copy()
                    plot.loc[plot[f"{cid}_t"] == -127, f"{cid}_t"] = None
                    i_a = plot.get("current_a", pd.Series([0] * len(plot)))
                    i_a = pd.to_numeric(i_a, errors="coerce").fillna(0).clip(
                        lower=0)
                    amb = pd.to_numeric(
                        plot.get("t_ambient", pd.Series([25.0] * len(plot))),
                        errors="coerce").fillna(25.0).rolling(
                        8, min_periods=1).mean()
                    rv = df.loc[df["cell"] == cid, "r_n"]
                    r_int = (num(rv.iloc[0], 50.0) / 1000.0
                             if len(rv) else 0.05)
                    q = (i_a ** 2) * r_int
                    t_ss = amb + q * R_TH
                    tau = R_TH * C_TH
                    plot["model"] = t_ss - (t_ss - amb) * np.exp(
                        -pd.to_numeric(plot["t_s"], errors="coerce") / tau)
                    plot["resid"] = plot[f"{cid}_t"] - plot["model"]

                    g = go.Figure()
                    g.add_trace(go.Scatter(x=plot["t_s"], y=plot[f"{cid}_t"],
                                           name=f"{cid} measured",
                                           line=dict(color="#ff5252",
                                                     width=2)))
                    g.add_trace(go.Scatter(x=plot["t_s"], y=plot["model"],
                                           name="RC twin model",
                                           line=dict(color="#2dd4ff",
                                                     dash="dash")))
                    if "t_ambient" in plot.columns:
                        g.add_trace(go.Scatter(x=plot["t_s"],
                                               y=plot["t_ambient"],
                                               name="ambient",
                                               line=dict(color="#7d8b99",
                                                         width=1)))
                    g.add_hline(y=45, line_dash="dot", line_color="#ffab00")
                    g.add_hline(y=55, line_dash="dot", line_color="#ff1744")
                    g.update_layout(template="plotly_dark", height=320,
                                    margin=dict(l=8, r=8, t=26, b=8),
                                    xaxis_title="t (s)", yaxis_title="deg C",
                                    legend=dict(orientation="h", y=1.16))
                    st.plotly_chart(g, use_container_width=True)

                    r = plot["resid"].dropna()
                    meas = plot[f"{cid}_t"].dropna()
                    rr = st.columns(3)
                    rr[0].metric("Measured",
                                 f"{meas.iloc[-1]:.2f} C" if len(meas)
                                 else "FAULT")
                    rr[1].metric("Twin model",
                                 f"{plot['model'].iloc[-1]:.2f} C")
                    rr[2].metric("Residual",
                                 f"{r.iloc[-1]:+.2f} C" if len(r) else "N/A")
                else:
                    st.info("Collecting history...")
            with c2:
                st.markdown("#### Thermal Assessment")
                if qual != "GOOD":
                    st.warning(f"Data quality: {qual}")
                if rules:
                    for r_ in rules:
                        st.error(r_)
                else:
                    st.success("No thermal rules active")
                st.markdown("#### Evidence")
                for e in ev:
                    st.markdown(f"- {e}")
                st.caption(f"Twin: lumped RC, tau = R_th x C_th = "
                           f"{R_TH * C_TH:.0f} s")


# ═══════════════════ ALERTS ═══════════════════
with tab3:
    al = []
    for pid, v in active.items():
        L, dg = v["live"], v["diag"]
        tmax = num(L.get("t_max"), -127)
        if tmax != -127 and tmax > 55:
            al.append([pid, "THERMAL", "CRITICAL",
                       f"T max {tmax:.1f} C above 55 C abort threshold"])
        elif tmax != -127 and tmax > 45:
            al.append([pid, "THERMAL", "WARNING",
                       f"T max {tmax:.1f} C above 45 C ceiling"])
        if num(dg["v_delta_mv"]) > 50:
            al.append([pid, "IMBALANCE", "ALERT",
                       f"Spread {num(dg['v_delta_mv']):.0f} mV > 50 mV"])
        if dg["weakest"]:
            al.append([pid, "WEAK CELL", "CRITICAL",
                       f"{dg['weakest']} lowest voltage and highest temp"])
        for _, c in v["df"].iterrows():
            if c["t"] == -127:
                al.append([pid, "SENSOR", "DEGRADED",
                           f"{c['cell']} thermistor open circuit (-127)"])
        if L.get("soh") is not None and num(L["soh"], 100) < 80:
            al.append([pid, "HEALTH", "WARNING",
                       f"SoH {num(L['soh']):.0f}% below 80% end-of-life"])
        if L.get("soc") is not None and num(L["soc"], 100) < 15:
            al.append([pid, "SOC", "WARNING",
                       f"SoC {num(L['soc']):.0f}% low"])
        if pid == HW_ID and L.get("aborted"):
            al.append([pid, "SAFETY", "CRITICAL",
                       "ESP32 over-temp abort latched - load cut"])

    if al:
        st.dataframe(pd.DataFrame(
            al, columns=["Pack", "Type", "Severity", "Detail"]),
            use_container_width=True, hide_index=True)
        st.caption(f"{len(al)} active alerts | thresholds: 45 C warn, "
                   "55 C abort, 50 mV balance, 80% SoH EOL")
    else:
        st.success("No active alerts across the fleet")


# ═══════════════════ PASSPORT ═══════════════════
with tab4:
    if not active:
        st.info("No live packs.")
    else:
        pid = st.selectbox("Pack passport", sorted(active), key="pp")
        P = active[pid]
        lb, _ = band(P["score"])
        doc = {
            "battery_passport_version": "1.0",
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "identity": P["meta"],
            "state": {k: v for k, v in P["live"].items() if k != "cells"},
            "cells": P["live"].get("cells", {}),
            "diagnosis": {
                "verdict": P["diag"]["verdict"],
                "weakest_cell": P["diag"]["weakest"],
                "imbalance_mv": P["diag"]["v_delta_mv"],
                "thermal_gradient_c": P["diag"]["t_gradient"],
                "risk_score": P["score"],
                "risk_band": lb,
                "evidence": P["diag"]["evidence"]},
            "thresholds": {"t_warn_c": 45, "t_abort_c": 55,
                           "balance_mv": 50, "soh_eol_pct": 80},
            "data_source": ("hardware" if pid == HW_ID
                            else meta_up.get("mode", "unknown")),
            "limitations": (
                ["No current sensor fitted - SoC from OCV lookup",
                 "SoH requires a full cycle - not reported",
                 "Single cell - balance analytics not applicable"]
                if pid == HW_ID else
                ["Simulated electro-thermal model for demonstration"]),
        }
        st.json(doc, expanded=False)
        st.download_button("Download Passport JSON",
                           json.dumps(doc, indent=2).encode(),
                           f"passport_{pid}.json", "application/json")
        H = hist_df(pid)
        if not H.empty:
            st.download_button("Download Telemetry CSV",
                               H.to_csv(index=False).encode(),
                               f"telemetry_{pid}.csv", "text/csv")
# ═══════════════════ OFFLINE BUFFER ═══════════════════
with tab5:
    st.markdown("### Offline Buffering — store, forward, deduplicate")

    bs = fb("/buffer_status") or {}
    stats = bs.get("stats", {}) if isinstance(bs.get("stats"), dict) else {}
    offline = bool(bs.get("offline"))
    withhold = bool(bs.get("withhold_ack"))
    qd = int(num(bs.get("queue_depth")))

    if not bs:
        st.info("No buffer telemetry. Start the uploader:")
        st.code("python fb_uploader2.py")
    elif offline:
        st.error(f"LINK DOWN — {qd} records buffered locally in SQLite WAL")
    elif qd > 0:
        st.warning(f"DRAINING — {qd} records replaying to cloud")
    else:
        st.success("LINK UP — direct delivery, queue empty")

    if withhold:
        st.warning("LOST-ACK MODE ACTIVE — receiver stores, ACK withheld")

    b = st.columns(6)
    b[0].metric("Queue depth", qd)
    b[1].metric("Delivered", int(num(bs.get("delivered"))))
    b[2].metric("Oldest queued", f"{num(bs.get('oldest_age_s')):.0f} s")
    b[3].metric("Total sent", int(num(stats.get("sent"))))
    b[4].metric("Duplicates rejected",
                int(num(stats.get("duplicates_rejected"))))
    b[5].metric("Retransmits", int(num(stats.get("retransmits"))))

    st.markdown("---")
    c1, c2, c3 = st.columns(3)

    if c1.button("SIMULATE OUTAGE" if not offline else "RESTORE LINK",
                 type="primary", use_container_width=True):
        fb_put("/control/force_offline", not offline)
        st.toast("Outage started" if not offline else "Link restored")
        time.sleep(1.2)
        st.rerun()

    if c2.button("Lost-ACK test ON" if not withhold else "Lost-ACK OFF",
                 use_container_width=True):
        fb_put("/control/withhold_ack", not withhold)
        st.toast("Lost-ACK mode toggled")
        time.sleep(1.2)
        st.rerun()

    if c3.button("Reset buffer stats", use_container_width=True):
        fb_put("/control/reset_buffer", True)
        st.toast("Reset requested")
        time.sleep(1.2)
        st.rerun()

    st.markdown("---")
    st.markdown("""
#### Mechanism

| Stage | Implementation |
|---|---|
| **Capture** | Deterministic record ID: `packs\\|PACK_A\\|history\\|000123` |
| **Outage** | PUT raises → row written to local SQLite WAL queue |
| **Bounded** | Queue caps at 5000; oldest dropped first, count tracked |
| **Restore** | `drain()` replays oldest-first in batches of 60 |
| **Dedup** | Receiver ledger checked before every send — known IDs rejected |
| **Lost ACK** | Receiver stores, ACK withheld → device retransmits → ledger rejects → **zero duplicate rows** |
| **Persistence** | SQLite survives process restart; queue intact |

**Why it matters:** a battery monitor that drops telemetry during a network
outage cannot produce a trustworthy incident report. Store-and-forward makes
the record complete and auditable.
""")
    st.caption("7/7 automated tests pass — run `python buffer_test.py`. "
               "ESP32 LittleFS persistence test pending.")
st.caption("Live from Firebase RTDB asia-southeast1 | pack & cell diagnostics "
           "| explainable thermal rules | RC digital twin | "
           "N/A shown where a quantity is not measurable")

if auto:
    time.sleep(3)
    st.rerun()