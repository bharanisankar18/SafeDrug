"""
SafeDose - Drug Safety Signal Detection Dashboard
=================================================
Streamlit + Plotly pharmacovigilance dashboard over the pre-computed
Parquet tables in data/processed/ (produced by src/03_clean.py ... 07).

Run from the project root:
    streamlit run src/app.py

Pages (sidebar navigation):
    Overview          - portfolio KPIs, reporting volume, top drugs/reactions, demographics
    Drug Explorer     - single-drug safety profile, detected signals, co-prescriptions
    Safety Signals    - Evans-criteria disproportionality (PRR / ROR / chi-square)
    Compare Drugs     - side-by-side reaction and demographic profile
    Trends & Spikes   - monthly volume with Z >= 3 surge detection
    Interaction Network - FP-Growth co-prescription graph
    Architecture      - pipeline status, Hadoop/Spark stack, MapReduce output
"""

from __future__ import annotations

import html
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SafeDose | Drug Safety Intelligence",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
NAVY = "#0F4C81"
INK = "#0B2545"
TEAL = "#0E9F9A"
AMBER = "#D97706"
RED = "#B91C1C"
SLATE = "#64748B"
GRID = "#E5E7EB"
TEXT = "#1F2937"
FONT = "Inter, -apple-system, Segoe UI, Roboto, sans-serif"

DRUG_A_COLOR = NAVY
DRUG_B_COLOR = AMBER

PLOT_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d", "toggleSpikelines"],
}

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, .stApp, .stMarkdown, button, input, textarea, select {{
        font-family: {FONT};
    }}
    .block-container {{ padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1440px; }}
    [data-testid="stHeader"] {{ background: transparent; }}
    #MainMenu, footer {{ visibility: hidden; }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{ background: #FFFFFF; border-right: 1px solid {GRID}; }}
    .brand {{ display:flex; align-items:center; gap:.6rem; margin:.2rem 0 1.1rem; }}
    .brand-mark {{ width:36px; height:36px; border-radius:9px; background:{NAVY}; color:#fff;
                   display:flex; align-items:center; justify-content:center; font-weight:700; font-size:1rem; }}
    .brand-name {{ font-weight:700; font-size:1.15rem; color:{INK}; line-height:1.1; }}
    .brand-tag {{ font-size:.74rem; color:{SLATE}; }}
    .side-label {{ font-size:.7rem; text-transform:uppercase; letter-spacing:.08em; color:{SLATE};
                   font-weight:600; margin:1rem 0 .4rem; }}
    .side-row {{ display:flex; justify-content:space-between; font-size:.82rem; padding:.28rem 0;
                 border-bottom:1px dashed {GRID}; color:{TEXT}; }}
    .side-row span:last-child {{ font-weight:600; }}
    .dot {{ display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:.45rem; }}

    /* Header */
    .app-header {{ display:flex; justify-content:space-between; align-items:flex-end; gap:1rem;
                   flex-wrap:wrap; padding-bottom:.9rem; border-bottom:1px solid {GRID}; margin-bottom:1rem; }}
    .app-eyebrow {{ font-size:.72rem; text-transform:uppercase; letter-spacing:.1em; color:{TEAL}; font-weight:700; }}
    .app-title {{ font-size:1.65rem; font-weight:700; color:{INK}; letter-spacing:-.01em; margin:.1rem 0 0; }}
    .app-sub {{ color:{SLATE}; font-size:.9rem; margin-top:.15rem; }}
    .badges {{ display:flex; gap:.4rem; flex-wrap:wrap; }}
    .badge {{ background:#E8F0F8; color:{NAVY}; padding:.32rem .75rem; border-radius:999px;
              font-size:.76rem; font-weight:600; white-space:nowrap; }}
    .badge-teal {{ background:#E6F6F5; color:#0B7A76; }}

    .notice {{ background:#FFFBEB; border:1px solid #FDE68A; border-left:4px solid {AMBER};
               border-radius:10px; padding:.65rem 1rem; color:#78350F; font-size:.84rem; margin-bottom:1.1rem; }}

    /* KPI cards */
    .kpi {{ background:#fff; border:1px solid {GRID}; border-top:3px solid var(--tone); border-radius:12px;
            padding:.9rem 1.05rem; height:108px; display:flex; flex-direction:column; justify-content:space-between;
            box-shadow:0 1px 2px rgba(15,23,42,.04); }}
    .kpi-label {{ font-size:.7rem; text-transform:uppercase; letter-spacing:.07em; color:{SLATE}; font-weight:600;
                  white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .kpi-value {{ font-size:1.6rem; font-weight:700; color:{INK}; line-height:1.1; white-space:nowrap;
                  overflow:hidden; text-overflow:ellipsis; }}
    .kpi-note {{ font-size:.76rem; color:{SLATE}; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .kpi-split {{ display:flex; gap:.9rem; align-items:baseline; }}
    .kpi-split .a {{ color:{DRUG_A_COLOR}; }} .kpi-split .b {{ color:{DRUG_B_COLOR}; }}
    .kpi-split .vs {{ font-size:.8rem; color:{SLATE}; font-weight:500; }}

    /* Cards (bordered containers created with key="card_...") */
    [class*="st-key-card"] {{ background:#FFFFFF; border-radius:12px !important; border-color:{GRID} !important;
                             box-shadow:0 1px 2px rgba(15,23,42,.04); }}
    .card-title {{ font-size:.95rem; font-weight:600; color:#111827; margin:0; }}
    .card-cap {{ font-size:.8rem; color:{SLATE}; margin:.1rem 0 .35rem; }}

    .page-title {{ font-size:1.15rem; font-weight:700; color:{INK}; margin:.2rem 0 .1rem; }}
    .page-cap {{ font-size:.86rem; color:{SLATE}; margin-bottom:.9rem; max-width:980px; }}

    .step {{ display:flex; gap:.8rem; align-items:flex-start; padding:.6rem 0; border-bottom:1px solid #F1F5F9; }}
    .step:last-child {{ border-bottom:none; }}
    .step-num {{ flex:0 0 30px; height:30px; border-radius:8px; background:#E8F0F8; color:{NAVY};
                 font-weight:700; display:flex; align-items:center; justify-content:center; font-size:.85rem; }}
    .step-body {{ flex:1; }}
    .step-name {{ font-weight:600; font-size:.88rem; color:#111827; }}
    .step-desc {{ font-size:.8rem; color:{SLATE}; }}
    .step-status {{ font-size:.75rem; font-weight:600; white-space:nowrap; }}

    div[data-testid="stDataFrame"] {{ border-radius:8px; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def esc(value) -> str:
    return html.escape(str(value))


def kpi(label: str, value: str, note: str = "", tone: str = NAVY) -> None:
    st.markdown(
        f'<div class="kpi" style="--tone:{tone}">'
        f'<div class="kpi-label">{esc(label)}</div>'
        f'<div class="kpi-value">{esc(value)}</div>'
        f'<div class="kpi-note">{esc(note) if note else "&nbsp;"}</div></div>',
        unsafe_allow_html=True,
    )


def kpi_row(items: list[tuple]) -> None:
    cols = st.columns(len(items), gap="small")
    for col, item in zip(cols, items):
        with col:
            kpi(*item)


def kpi_compare(label: str, a: str, b: str, note: str = "") -> None:
    st.markdown(
        f'<div class="kpi" style="--tone:{SLATE}">'
        f'<div class="kpi-label">{esc(label)}</div>'
        f'<div class="kpi-value kpi-split"><span class="a">{esc(a)}</span>'
        f'<span class="vs">vs</span><span class="b">{esc(b)}</span></div>'
        f'<div class="kpi-note">{esc(note) if note else "&nbsp;"}</div></div>',
        unsafe_allow_html=True,
    )


def page_header(title: str, caption: str) -> None:
    st.markdown(f'<div class="page-title">{esc(title)}</div><div class="page-cap">{caption}</div>',
                unsafe_allow_html=True)


def card_header(title: str, caption: str = "") -> None:
    cap = f'<div class="card-cap">{caption}</div>' if caption else '<div class="card-cap">&nbsp;</div>'
    st.markdown(f'<div class="card-title">{esc(title)}</div>{cap}', unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int = 340, legend: bool = False) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=8, r=16, t=10, b=8),
        font=dict(family=FONT, size=12, color=TEXT),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=legend,
        hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(family=FONT, size=12, color=TEXT)),
        coloraxis_showscale=False,
        bargap=0.28,
    )
    if legend:
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title_text=""))
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, title_font=dict(size=11, color=SLATE))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, title_font=dict(size=11, color=SLATE))
    return fig


def show_chart(fig: go.Figure, key: str | None = None) -> None:
    st.plotly_chart(fig, width="stretch", theme=None, config=PLOT_CONFIG, key=key)


def hbar(df: pd.DataFrame, x: str, y: str, color: str, height: int = 340,
         text: str | None = None, x_title: str = "Reports") -> go.Figure:
    fig = px.bar(df, x=x, y=y, orientation="h", text=text)
    fig.update_traces(marker_color=color, marker_line_width=0,
                      hovertemplate="<b>%{y}</b><br>" + x_title + ": %{x:,}<extra></extra>")
    if text:
        fig.update_traces(texttemplate="%{text}%", textposition="outside", cliponaxis=False,
                          textfont=dict(size=11, color=SLATE))
    fig.update_layout(yaxis=dict(categoryorder="total ascending", title=None), xaxis_title=x_title)
    return style_fig(fig, height)


def donut(labels, values, colors, height: int = 280) -> go.Figure:
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.62, sort=False,
                           marker=dict(colors=colors, line=dict(color="#fff", width=2)),
                           textinfo="percent", textfont=dict(size=12),
                           hovertemplate="<b>%{label}</b><br>%{value:,} reports (%{percent})<extra></extra>"))
    fig = style_fig(fig, height, legend=True)
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.02, xanchor="center", x=0.5))
    return fig


def age_hist(ages: pd.Series, color: str, height: int = 280, nbins: int = 30) -> go.Figure:
    fig = px.histogram(x=ages, nbins=nbins)
    fig.update_traces(marker_color=color, marker_line_width=0,
                      hovertemplate="Age %{x}<br>%{y:,} reports<extra></extra>")
    fig.update_layout(xaxis_title="Age (years)", yaxis_title="Reports")
    return style_fig(fig, height)


def fmt_int(n) -> str:
    return f"{int(n):,}"


def pct(series_bool: pd.Series) -> float:
    return float(series_bool.mean() * 100.0) if len(series_bool) else 0.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TABLES = {
    "clean": PROCESSED_DIR / "clean_reports.parquet",
    "signals": PROCESSED_DIR / "signals.parquet",
    "interactions": PROCESSED_DIR / "interactions.parquet",
    "trends": PROCESSED_DIR / "trends.parquet",
    "mapreduce": PROCESSED_DIR / "hadoop_mr_counts",
}

SERIOUS_MAP = {"1": "Serious", "2": "Non-serious"}
SEX_MAP = {"1": "Male", "2": "Female", "0": "Unknown"}


@st.cache_data(show_spinner="Loading safety tables...")
def load_tables() -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for name, path in TABLES.items():
        out[name] = pd.read_parquet(path) if path.exists() else pd.DataFrame()

    clean = out["clean"]
    if not clean.empty:
        clean["seriousness"] = clean["seriousness"].astype("string")
        clean["sex"] = clean["sex"].astype("string")
        clean["age"] = pd.to_numeric(clean["age"], errors="coerce")
        clean["receive_dt"] = pd.to_datetime(clean["receivedate"], format="%Y%m%d", errors="coerce")

    sig = out["signals"]
    if not sig.empty:
        # Older pipeline versions used `pair_count`; current one writes `report_count`.
        if "report_count" not in sig.columns and "pair_count" in sig.columns:
            sig = sig.rename(columns={"pair_count": "report_count"})
        if "is_signal" not in sig.columns:
            sig["is_signal"] = (sig["report_count"] >= 3) & (sig["prr"] >= 2.0) & (sig["chi_square"] >= 4.0)
        out["signals"] = sig

    tr = out["trends"]
    if not tr.empty:
        tr["month_dt"] = pd.to_datetime(tr["year_month"], format="%Y-%m", errors="coerce")
    return out


@st.cache_data(show_spinner=False)
def derived(clean: pd.DataFrame) -> dict:
    reports = clean.drop_duplicates("safetyreportid")
    drug_counts = clean.groupby("drug_name")["safetyreportid"].nunique().sort_values(ascending=False)
    rx_counts = clean.groupby("reaction_name")["safetyreportid"].nunique().sort_values(ascending=False)
    monthly = (
        reports.dropna(subset=["receive_dt"])
        .assign(month=lambda d: d["receive_dt"].dt.to_period("M").dt.to_timestamp())
        .groupby("month")
        .agg(reports=("safetyreportid", "size"), serious=("seriousness", lambda s: (s == "1").sum()))
        .reset_index()
    )
    return {
        "reports": reports,
        "drug_counts": drug_counts,
        "rx_counts": rx_counts,
        "monthly": monthly,
        "date_min": reports["receive_dt"].min(),
        "date_max": reports["receive_dt"].max(),
    }


data = load_tables()
clean_df = data["clean"]
signals_df = data["signals"]
interactions_df = data["interactions"]
trends_df = data["trends"]
mr_df = data["mapreduce"]

if clean_df.empty or signals_df.empty:
    st.error(
        "Processed tables not found in `data/processed/`. Run the pipeline first:\n\n"
        "```\npython src/03_clean.py\npython src/04_signals.py\npython src/05_interactions.py\npython src/06_trends.py\n```"
    )
    st.stop()

D = derived(clean_df)
reports_df = D["reports"]
drug_counts = D["drug_counts"]
ALL_DRUGS = drug_counts.index.tolist()  # ordered by report volume, most reported first


def default_index(options: list, preferred: str, fallback: int = 0) -> int:
    return options.index(preferred) if preferred in options else min(fallback, len(options) - 1)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
PAGES = [
    "Overview",
    "Drug Explorer",
    "Safety Signals",
    "Compare Drugs",
    "Trends & Spikes",
    "Interaction Network",
    "Architecture",
]
PAGE_ICONS = {
    "Overview": ":material/dashboard:",
    "Drug Explorer": ":material/medication:",
    "Safety Signals": ":material/crisis_alert:",
    "Compare Drugs": ":material/compare_arrows:",
    "Trends & Spikes": ":material/trending_up:",
    "Interaction Network": ":material/hub:",
    "Architecture": ":material/account_tree:",
}

with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-mark">SD</div>'
        '<div><div class="brand-name">SafeDose</div>'
        '<div class="brand-tag">Pharmacovigilance Intelligence</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="side-label">Navigation</div>', unsafe_allow_html=True)
    page = st.radio(
        "Navigation",
        PAGES,
        format_func=lambda p: f"{PAGE_ICONS[p]}  {p}",
        label_visibility="collapsed",
    )

    st.markdown('<div class="side-label">Dataset</div>', unsafe_allow_html=True)
    window = (
        f"{D['date_min']:%b %Y} – {D['date_max']:%b %Y}"
        if pd.notna(D["date_min"]) and pd.notna(D["date_max"]) else "n/a"
    )
    rows = [
        ("Source", "FDA FAERS"),
        ("Window", window),
        ("Case reports", fmt_int(len(reports_df))),
        ("Drug–reaction rows", fmt_int(len(clean_df))),
        ("Pairs scored", fmt_int(len(signals_df))),
    ]
    st.markdown("".join(f'<div class="side-row"><span>{esc(k)}</span><span>{esc(v)}</span></div>' for k, v in rows),
                unsafe_allow_html=True)

    st.markdown('<div class="side-label">Pipeline tables</div>', unsafe_allow_html=True)
    status_html = ""
    for name, label in [("clean", "Clean reports"), ("signals", "Signals"), ("interactions", "Interactions"),
                        ("trends", "Trends"), ("mapreduce", "MapReduce counts")]:
        ok = not data[name].empty
        color = TEAL if ok else "#CBD5E1"
        status_html += (f'<div class="side-row"><span><span class="dot" style="background:{color}"></span>'
                        f'{label}</span><span>{"Ready" if ok else "Missing"}</span></div>')
    st.markdown(status_html, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header + disclaimer (shown on every page)
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="app-header">
      <div>
        <div class="app-eyebrow">Post-market drug safety surveillance</div>
        <div class="app-title">SafeDose Signal Detection</div>
        <div class="app-sub">Disproportionality mining over FDA adverse event reports · Apache Spark pipeline</div>
      </div>
      <div class="badges">
        <span class="badge">{esc(window)}</span>
        <span class="badge badge-teal">{fmt_int(len(reports_df))} case reports</span>
      </div>
    </div>
    <div class="notice"><strong>Association, not causation.</strong>
      Signals are statistical disproportionality findings from spontaneous FAERS reports. Each one is a hypothesis
      that needs clinical and epidemiological review before any conclusion about a drug is drawn.</div>
    """,
    unsafe_allow_html=True,
)


# ===========================================================================
# PAGE: OVERVIEW
# ===========================================================================
def page_overview() -> None:
    n_signals = int(signals_df["is_signal"].sum())
    n_inter = int(interactions_df["is_high_risk_interaction"].sum()) if not interactions_df.empty else 0
    n_spikes = int(trends_df["is_spike"].sum()) if not trends_df.empty else 0
    serious_rate = pct(reports_df["seriousness"] == "1")

    kpi_row([
        ("Case reports", fmt_int(len(reports_df)), f"{serious_rate:.1f}% serious outcomes", NAVY),
        ("Suspect drugs", fmt_int(len(drug_counts)), f"{fmt_int(clean_df['reaction_name'].nunique())} distinct reactions", TEAL),
        ("Safety signals", fmt_int(n_signals), "Evans criteria met", RED),
        ("High-risk interactions", fmt_int(n_inter), "FP-Growth drug pairs", AMBER),
        ("Volume spikes", fmt_int(n_spikes), "Monthly Z ≥ 3.0", SLATE),
    ])
    st.write("")

    with st.container(border=True, key="card_monthly"):
        card_header("Monthly reporting volume", "Unique case reports received per month, with the serious subset (zoomed to the active reporting period; double-click to see all)")
        m = D["monthly"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=m["month"], y=m["reports"], name="All reports", mode="lines",
                                 line=dict(color=NAVY, width=2.4), fill="tozeroy", fillcolor="rgba(15,76,129,0.08)",
                                 hovertemplate="%{x|%b %Y}<br>%{y:,} reports<extra></extra>"))
        fig.add_trace(go.Scatter(x=m["month"], y=m["serious"], name="Serious", mode="lines",
                                 line=dict(color=RED, width=1.8, dash="dot"),
                                 hovertemplate="%{x|%b %Y}<br>%{y:,} serious<extra></extra>"))
        active = m[m["reports"] >= max(5, 0.02 * m["reports"].max())]
        if not active.empty:
            fig.update_xaxes(range=[active["month"].min() - pd.DateOffset(months=1), m["month"].max() + pd.DateOffset(days=15)])
        fig.update_layout(yaxis_title="Reports", xaxis_title=None, hovermode="x unified")
        show_chart(style_fig(fig, 280, legend=True))

    c1, c2 = st.columns(2, gap="medium")
    with c1, st.container(border=True, key="card_topdrugs"):
        card_header("Most reported suspect drugs", "Top 10 by unique case reports")
        top = drug_counts.head(10).rename_axis("drug").reset_index(name="reports")
        show_chart(hbar(top, "reports", "drug", NAVY, 360))
    with c2, st.container(border=True, key="card_toprx"):
        card_header("Most reported adverse reactions", "Top 10 MedDRA preferred terms by unique case reports")
        top = D["rx_counts"].head(10).rename_axis("reaction").reset_index(name="reports")
        show_chart(hbar(top, "reports", "reaction", TEAL, 360))

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1, st.container(border=True, key="card_serious"):
        card_header("Outcome seriousness", "Share of case reports")
        s = reports_df["seriousness"].map(SERIOUS_MAP).fillna("Not specified").value_counts()
        s = s.reindex([k for k in ["Serious", "Non-serious", "Not specified"] if k in s.index])
        show_chart(donut(s.index, s.values, [RED, NAVY, "#CBD5E1"]))
    with c2, st.container(border=True, key="card_sex"):
        card_header("Patient sex", "Share of case reports")
        s = reports_df["sex"].map(SEX_MAP).fillna("Unknown").value_counts()
        s = s.reindex([k for k in ["Female", "Male", "Unknown"] if k in s.index])
        show_chart(donut(s.index, s.values, [TEAL, NAVY, "#CBD5E1"]))
    with c3, st.container(border=True, key="card_age"):
        ages = reports_df["age"].dropna()
        ages = ages[(ages >= 0) & (ages <= 105)]
        card_header("Patient age", f"{fmt_int(len(ages))} reports with age · median {ages.median():.0f} yrs"
                    if len(ages) else "No age data")
        show_chart(age_hist(ages, NAVY))

    with st.container(border=True, key="card_topsignals"):
        card_header("Strongest safety signals", "Evans-criteria signals ranked by chi-square (top 10)")
        top_sig = signals_df[signals_df["is_signal"]].sort_values("chi_square", ascending=False).head(10)
        st.dataframe(
            top_sig[["drug_name", "reaction_name", "report_count", "prr", "ror", "chi_square"]],
            hide_index=True, width="stretch",
            column_config={
                "drug_name": "Drug", "reaction_name": "Reaction",
                "report_count": st.column_config.NumberColumn("Reports (a)", format="%d"),
                "prr": st.column_config.NumberColumn("PRR", format="%.2f"),
                "ror": st.column_config.NumberColumn("ROR", format="%.2f"),
                "chi_square": st.column_config.NumberColumn("χ²", format="%.1f"),
            },
        )


# ===========================================================================
# PAGE: DRUG EXPLORER
# ===========================================================================
def page_explorer() -> None:
    page_header("Drug Explorer", "Safety profile of a single suspect drug: reactions, demographics, "
                                 "detected signals and frequent co-prescriptions.")

    with st.container(border=True, key="card_explorer_pick"):
        c1, c2 = st.columns([2, 3], gap="medium")
        with c1:
            drug = st.selectbox("Drug", ALL_DRUGS, index=default_index(ALL_DRUGS, "adalimumab"),
                                help="Sorted by report volume (most reported first). Type to search.")
        with c2:
            rank = ALL_DRUGS.index(drug) + 1
            st.markdown(f'<div class="card-cap" style="margin-top:2rem">Rank #{rank:,} of {len(ALL_DRUGS):,} '
                        f'suspect drugs by report volume.</div>', unsafe_allow_html=True)

    recs = clean_df[clean_df["drug_name"] == drug]
    reps = recs.drop_duplicates("safetyreportid")
    n = len(reps)
    ages = reps["age"].dropna()
    ages = ages[(ages >= 0) & (ages <= 105)]
    drug_signals = signals_df[(signals_df["drug_name"] == drug) & signals_df["is_signal"]]

    st.write("")
    kpi_row([
        ("Case reports", fmt_int(n), f"{fmt_int(recs['reaction_name'].nunique())} distinct reactions", NAVY),
        ("Serious outcome rate", f"{pct(reps['seriousness'] == '1'):.1f}%", "of this drug's reports", RED),
        ("Female patients", f"{pct(reps['sex'] == '2'):.1f}%", f"Male {pct(reps['sex'] == '1'):.1f}%", TEAL),
        ("Mean patient age", f"{ages.mean():.1f} yrs" if len(ages) else "n/a",
         f"{fmt_int(len(ages))} reports with age", SLATE),
        ("Safety signals", fmt_int(len(drug_signals)), "Evans criteria met", AMBER),
    ])
    st.write("")

    c1, c2 = st.columns([3, 2], gap="medium")
    with c1, st.container(border=True, key="card_drug_rx"):
        card_header("Top 15 adverse reactions", "Label shows % of this drug's case reports")
        rx = (recs.groupby("reaction_name")["safetyreportid"].nunique()
              .sort_values(ascending=False).head(15).rename_axis("reaction").reset_index(name="reports"))
        rx["share"] = (rx["reports"] / max(n, 1) * 100).round(1)
        show_chart(hbar(rx, "reports", "reaction", NAVY, 545, text="share"))
    with c2:
        with st.container(border=True, key="card_drug_age"):
            card_header("Patient age", f"Median {ages.median():.0f} yrs" if len(ages) else "No age data reported")
            if len(ages):
                show_chart(age_hist(ages, TEAL, 190, nbins=20))
            else:
                st.info("No patient age data for this drug.")
        with st.container(border=True, key="card_drug_co"):
            card_header("Frequent co-prescriptions", "From FP-Growth interaction mining")
            co = pd.DataFrame()
            if not interactions_df.empty:
                co = interactions_df[(interactions_df["drug_1"] == drug) | (interactions_df["drug_2"] == drug)].copy()
            if co.empty:
                st.caption("No co-prescribed pairs above the minimum support threshold.")
            else:
                co["partner"] = np.where(co["drug_1"] == drug, co["drug_2"], co["drug_1"])
                co = co.sort_values("pair_report_count", ascending=False).head(6)
                st.dataframe(
                    co[["partner", "pair_report_count", "reaction_rate_ratio", "top_reaction"]],
                    hide_index=True, width="stretch", height=250,
                    column_config={
                        "partner": "Co-drug",
                        "pair_report_count": st.column_config.NumberColumn("Co-reports", format="%d"),
                        "reaction_rate_ratio": st.column_config.NumberColumn("Rate ratio", format="%.2f"),
                        "top_reaction": "Top reaction",
                    },
                )

    with st.container(border=True, key="card_drug_signals"):
        card_header("Detected safety signals for this drug", "Drug–reaction pairs meeting Evans criteria, by PRR")
        if drug_signals.empty:
            st.caption("No drug–reaction pair for this drug meets the Evans criteria.")
        else:
            st.dataframe(
                drug_signals.sort_values("prr", ascending=False)
                [["reaction_name", "report_count", "prr", "ror", "ror_lower_ci", "ror_upper_ci", "chi_square"]],
                hide_index=True, width="stretch", height=min(38 + 35 * len(drug_signals), 380),
                column_config={
                    "reaction_name": "Reaction",
                    "report_count": st.column_config.NumberColumn("Reports (a)", format="%d"),
                    "prr": st.column_config.NumberColumn("PRR", format="%.2f"),
                    "ror": st.column_config.NumberColumn("ROR", format="%.2f"),
                    "ror_lower_ci": st.column_config.NumberColumn("ROR 95% low", format="%.2f"),
                    "ror_upper_ci": st.column_config.NumberColumn("ROR 95% high", format="%.2f"),
                    "chi_square": st.column_config.NumberColumn("χ²", format="%.1f"),
                },
            )


# ===========================================================================
# PAGE: SAFETY SIGNALS
# ===========================================================================
def page_signals() -> None:
    page_header(
        "Safety Signals",
        "Disproportionality analysis compares how often a reaction is reported for a drug against all other drugs. "
        "<b>Evans criteria:</b> PRR ≥ 2, Yates χ² ≥ 4 (p &lt; 0.05) and at least 3 reports. "
        "Adjust the thresholds below to explore weaker or stronger associations.",
    )

    with st.container(border=True, key="card_sig_filters"):
        f1, f2, f3, f4 = st.columns(4, gap="medium")
        drug_q = f1.text_input("Drug contains", placeholder="e.g. adalimumab")
        rx_q = f2.text_input("Reaction contains", placeholder="e.g. infection")
        min_prr = f3.slider("Minimum PRR", 1.0, 20.0, 2.0, 0.5)
        min_a = f4.slider("Minimum reports (a)", 1, 50, 3, 1)

    sig = signals_df
    if drug_q.strip():
        sig = sig[sig["drug_name"].str.contains(drug_q.strip().lower(), na=False, regex=False)]
    if rx_q.strip():
        sig = sig[sig["reaction_name"].str.contains(rx_q.strip().lower(), na=False, regex=False)]
    sig = sig[(sig["prr"] >= min_prr) & (sig["chi_square"] >= 4.0) & (sig["report_count"] >= min_a)]
    sig = sig.sort_values("chi_square", ascending=False)

    st.write("")
    kpi_row([
        ("Matching signals", fmt_int(len(sig)), f"of {fmt_int(len(signals_df))} scored pairs", RED),
        ("Drugs involved", fmt_int(sig["drug_name"].nunique()), "distinct suspect drugs", NAVY),
        ("Reactions involved", fmt_int(sig["reaction_name"].nunique()), "distinct MedDRA terms", TEAL),
        ("Median PRR", f"{sig['prr'].median():.2f}" if len(sig) else "n/a", "effect size", AMBER),
    ])
    st.write("")

    if sig.empty:
        st.info("No pairs match these filters. Lower the minimum PRR or report count.")
        return

    with st.container(border=True, key="card_volcano"):
        card_header("Signal volcano plot",
                    f"Effect size vs statistical strength for all {len(sig):,} matching pairs. "
                    "Hover a point for details; drag to zoom.")
        p = sig.copy()
        p["log2_prr"] = np.log2(p["prr"].clip(0.1, 1e5))
        p["log10_chi"] = np.log10(p["chi_square"].clip(1, 1e6))
        tiers = ["PRR 2–5", "PRR 5–10", "PRR 10–100", "PRR ≥ 100"]
        p["strength"] = pd.cut(p["prr"], [0, 5, 10, 100, np.inf], labels=tiers, right=False)
        fig = px.scatter(
            p, x="log2_prr", y="log10_chi", color="strength", render_mode="webgl",
            color_discrete_map=dict(zip(tiers, [TEAL, NAVY, AMBER, RED])),
            category_orders={"strength": tiers},
            custom_data=["drug_name", "reaction_name", "report_count", "prr", "ror", "chi_square"],
        )
        fig.update_traces(
            marker=dict(size=6, opacity=0.55, line=dict(width=0)),
            hovertemplate="<b>%{customdata[0]}</b> → %{customdata[1]}<br>Reports: %{customdata[2]:,}"
                          "<br>PRR: %{customdata[3]:.2f} · ROR: %{customdata[4]:.2f}<br>χ²: %{customdata[5]:.1f}<extra></extra>",
        )
        fig.add_vline(x=1.0, line_dash="dash", line_color=SLATE, line_width=1,
                      annotation_text="PRR = 2", annotation_font_color=SLATE, annotation_font_size=11)
        fig.add_hline(y=np.log10(4.0), line_dash="dash", line_color=SLATE, line_width=1,
                      annotation_text="χ² = 4", annotation_font_color=SLATE, annotation_font_size=11,
                      annotation_position="bottom right")
        fig.update_layout(xaxis_title="log₂(PRR)  ·  effect size", yaxis_title="log₁₀(χ²)  ·  statistical strength")
        show_chart(style_fig(fig, 440, legend=True))

    with st.container(border=True, key="card_sig_table"):
        h1, h2 = st.columns([4, 1], vertical_alignment="bottom")
        with h1:
            card_header("Signal table", f"Top 200 of {len(sig):,} matching pairs, ranked by χ²")
        with h2:
            st.download_button("Download CSV", sig.drop(columns=["is_signal"], errors="ignore").to_csv(index=False),
                               file_name="safedose_signals.csv", mime="text/csv", width="stretch",
                               icon=":material/download:")
        st.dataframe(
            sig.head(200)[["drug_name", "reaction_name", "report_count", "prr", "ror",
                           "ror_lower_ci", "ror_upper_ci", "chi_square"]],
            hide_index=True, width="stretch", height=420,
            column_config={
                "drug_name": "Drug",
                "reaction_name": "Reaction",
                "report_count": st.column_config.NumberColumn("Reports (a)", format="%d"),
                "total_drug_reports": st.column_config.NumberColumn("Drug total", format="%d"),
                "prr": st.column_config.NumberColumn("PRR", format="%.2f"),
                "ror": st.column_config.NumberColumn("ROR", format="%.2f"),
                "ror_lower_ci": st.column_config.NumberColumn("ROR 95% low", format="%.2f"),
                "ror_upper_ci": st.column_config.NumberColumn("ROR 95% high", format="%.2f"),
                "chi_square": st.column_config.NumberColumn("χ²", format="%.1f"),
            },
        )


# ===========================================================================
# PAGE: COMPARE DRUGS
# ===========================================================================
def page_compare() -> None:
    page_header("Compare Drugs", "Side-by-side reaction profile, seriousness and demographics of two suspect drugs. "
                                 "Reaction rates are normalised to each drug's own report count.")

    with st.container(border=True, key="card_cmp_pick"):
        c1, c2 = st.columns(2, gap="medium")
        drug_a = c1.selectbox("Drug A", ALL_DRUGS, index=default_index(ALL_DRUGS, "adalimumab"), key="cmp_a")
        drug_b = c2.selectbox("Drug B", ALL_DRUGS, index=default_index(ALL_DRUGS, "methotrexate", 1), key="cmp_b")

    if drug_a == drug_b:
        st.warning("Choose two different drugs to compare.")
        return

    rec_a, rec_b = clean_df[clean_df["drug_name"] == drug_a], clean_df[clean_df["drug_name"] == drug_b]
    rep_a, rep_b = rec_a.drop_duplicates("safetyreportid"), rec_b.drop_duplicates("safetyreportid")

    def mean_age(r):
        a = r["age"].dropna()
        a = a[(a >= 0) & (a <= 105)]
        return f"{a.mean():.0f}" if len(a) else "–"

    st.write("")
    st.markdown(
        f'<div style="font-size:.82rem;color:{SLATE};margin-bottom:.4rem">'
        f'<span class="dot" style="background:{DRUG_A_COLOR}"></span><b style="color:{TEXT}">{esc(drug_a)}</b>'
        f'&nbsp;&nbsp;&nbsp;<span class="dot" style="background:{DRUG_B_COLOR}"></span><b style="color:{TEXT}">{esc(drug_b)}</b></div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(4, gap="small")
    with cols[0]:
        kpi_compare("Case reports", fmt_int(len(rep_a)), fmt_int(len(rep_b)))
    with cols[1]:
        kpi_compare("Serious rate", f"{pct(rep_a['seriousness'] == '1'):.0f}%", f"{pct(rep_b['seriousness'] == '1'):.0f}%")
    with cols[2]:
        kpi_compare("Female share", f"{pct(rep_a['sex'] == '2'):.0f}%", f"{pct(rep_b['sex'] == '2'):.0f}%")
    with cols[3]:
        kpi_compare("Mean age (yrs)", mean_age(rep_a), mean_age(rep_b))
    st.write("")

    def top_rx(rec, rep, k=10):
        s = rec.groupby("reaction_name")["safetyreportid"].nunique().sort_values(ascending=False).head(k)
        return (s / max(len(rep), 1) * 100).round(1)

    ta, tb = top_rx(rec_a, rep_a), top_rx(rec_b, rep_b)
    union = list(dict.fromkeys(list(ta.index) + list(tb.index)))
    full_a = (rec_a[rec_a["reaction_name"].isin(union)].groupby("reaction_name")["safetyreportid"].nunique()
              / max(len(rep_a), 1) * 100).round(1)
    full_b = (rec_b[rec_b["reaction_name"].isin(union)].groupby("reaction_name")["safetyreportid"].nunique()
              / max(len(rep_b), 1) * 100).round(1)
    comp = pd.DataFrame({"a": full_a.reindex(union).fillna(0), "b": full_b.reindex(union).fillna(0)})
    comp = comp.assign(mx=comp.max(axis=1)).sort_values("mx").tail(15)

    with st.container(border=True, key="card_cmp_chart"):
        card_header("Reaction rates", "% of each drug's case reports mentioning the reaction (union of each drug's top 10)")
        fig = go.Figure()
        fig.add_trace(go.Bar(y=comp.index, x=comp["a"], name=drug_a, orientation="h", marker_color=DRUG_A_COLOR,
                             hovertemplate="%{y}<br>" + esc(drug_a) + ": %{x:.1f}%<extra></extra>"))
        fig.add_trace(go.Bar(y=comp.index, x=comp["b"], name=drug_b, orientation="h", marker_color=DRUG_B_COLOR,
                             hovertemplate="%{y}<br>" + esc(drug_b) + ": %{x:.1f}%<extra></extra>"))
        fig.update_layout(barmode="group", xaxis_title="% of case reports", yaxis_title=None, bargroupgap=0.08)
        show_chart(style_fig(fig, max(380, 30 * len(comp) + 80), legend=True))

    sa = signals_df[(signals_df["drug_name"] == drug_a) & signals_df["is_signal"]]
    sb = signals_df[(signals_df["drug_name"] == drug_b) & signals_df["is_signal"]]
    c1, c2 = st.columns(2, gap="medium")
    for col, d, s, color, key in [(c1, drug_a, sa, DRUG_A_COLOR, "card_cmp_sa"), (c2, drug_b, sb, DRUG_B_COLOR, "card_cmp_sb")]:
        with col, st.container(border=True, key=key):
            card_header(f"Top signals · {d}", f"{len(s):,} Evans-criteria signals")
            if s.empty:
                st.caption("No signals detected.")
            else:
                st.dataframe(s.sort_values("prr", ascending=False).head(10)[["reaction_name", "report_count", "prr"]],
                             hide_index=True, width="stretch", height=280,
                             column_config={"reaction_name": "Reaction",
                                            "report_count": st.column_config.NumberColumn("Reports", format="%d"),
                                            "prr": st.column_config.NumberColumn("PRR", format="%.2f")})


# ===========================================================================
# PAGE: TRENDS & SPIKES
# ===========================================================================
def page_trends() -> None:
    page_header("Trends & Spikes", "Monthly report volume per drug. A <b>surge spike</b> is a month whose count is at "
                                   "least 3 standard deviations above that series' historical mean (and ≥ 3 reports).")
    if trends_df.empty:
        st.info("Trends table not computed yet. Run `python src/06_trends.py`.")
        return

    spikes_all = trends_df[trends_df["is_spike"]]
    trend_drugs = sorted(trends_df["drug_name"].dropna().unique().tolist())
    spike_drugs = spikes_all.sort_values("z_score", ascending=False)["drug_name"].unique()
    default_drug = spike_drugs[0] if len(spike_drugs) else trend_drugs[0]

    with st.container(border=True, key="card_trend_pick"):
        c1, c2, c3 = st.columns([2, 2, 1], gap="medium")
        only_spiking = c3.toggle("Spiking only", value=True, help="List only drugs that have at least one Z ≥ 3 spike")
        options = sorted(spike_drugs.tolist()) if (only_spiking and len(spike_drugs)) else trend_drugs
        drug = c1.selectbox("Drug", options, index=default_index(options, default_drug))
        recs = trends_df[trends_df["drug_name"] == drug]
        rx_opts = ["All reactions (overall volume)"] + sorted(r for r in recs["reaction_name"].unique() if r != "ALL")
        rx_sel = c2.selectbox("Reaction scope", rx_opts)

    rx = "ALL" if rx_sel.startswith("All reactions") else rx_sel
    series = recs[recs["reaction_name"] == rx].sort_values("month_dt")
    if series.empty:
        st.info("No monthly series for this selection.")
        return

    mu, sd = float(series["mean_count"].iloc[0]), float(series["std_count"].iloc[0])
    thr = mu + 3 * sd
    sp = series[series["is_spike"]]

    st.write("")
    kpi_row([
        ("Months observed", fmt_int(len(series)), f"{series['month_dt'].min():%b %Y} – {series['month_dt'].max():%b %Y}", NAVY),
        ("Baseline mean (μ)", f"{mu:.1f}", "reports per month", TEAL),
        ("Spike threshold", f"{thr:.1f}", f"μ + 3σ  (σ = {sd:.1f})", AMBER),
        ("Surge spikes", fmt_int(len(sp)), f"max Z = {sp['z_score'].max():.1f}" if len(sp) else "none detected", RED),
    ])
    st.write("")

    with st.container(border=True, key="card_trend_chart"):
        scope = "all reactions" if rx == "ALL" else rx
        card_header(f"Monthly volume · {drug}", f"Scope: {esc(scope)}")
        fig = go.Figure()
        fig.add_hrect(y0=mu - sd, y1=mu + sd, fillcolor=NAVY, opacity=0.06, line_width=0)
        fig.add_trace(go.Scatter(x=series["month_dt"], y=series["report_count"], mode="lines+markers", name="Reports",
                                 line=dict(color=NAVY, width=2.4), marker=dict(size=6),
                                 hovertemplate="%{x|%b %Y}: %{y:,} reports<extra></extra>"))
        fig.add_hline(y=mu, line_dash="dash", line_color=SLATE, line_width=1,
                      annotation_text=f"μ = {mu:.1f}", annotation_font_size=11, annotation_font_color=SLATE)
        fig.add_hline(y=thr, line_dash="dot", line_color=RED, line_width=1.5,
                      annotation_text=f"+3σ = {thr:.1f}", annotation_font_size=11, annotation_font_color=RED)
        if not sp.empty:
            fig.add_trace(go.Scatter(x=sp["month_dt"], y=sp["report_count"], mode="markers", name="Surge spike (Z ≥ 3)",
                                     marker=dict(size=14, color=RED, symbol="diamond", line=dict(width=1.5, color="white")),
                                     customdata=sp["z_score"],
                                     hovertemplate="%{x|%b %Y}: %{y:,} reports<br>Z = +%{customdata:.2f}<extra></extra>"))
        fig.update_layout(xaxis_title=None, yaxis_title="Case reports", hovermode="closest")
        show_chart(style_fig(fig, 400, legend=True))

    c1, c2 = st.columns(2, gap="medium")
    with c1, st.container(border=True, key="card_trend_sp"):
        card_header("Spikes in this series")
        if sp.empty:
            st.caption("No month in this series exceeds the +3σ threshold.")
        else:
            st.dataframe(sp[["year_month", "report_count", "mean_count", "std_count", "z_score"]],
                         hide_index=True, width="stretch", height=300,
                         column_config={"year_month": "Month",
                                        "report_count": st.column_config.NumberColumn("Reports", format="%d"),
                                        "mean_count": st.column_config.NumberColumn("μ", format="%.1f"),
                                        "std_count": st.column_config.NumberColumn("σ", format="%.1f"),
                                        "z_score": st.column_config.NumberColumn("Z", format="%.2f")})
    with c2, st.container(border=True, key="card_trend_all"):
        card_header("Largest spikes across all drugs", "Top 15 by Z-score")
        st.dataframe(spikes_all.sort_values("z_score", ascending=False).head(15)
                     [["drug_name", "reaction_name", "year_month", "report_count", "z_score"]],
                     hide_index=True, width="stretch", height=300,
                     column_config={"drug_name": "Drug", "reaction_name": "Reaction", "year_month": "Month",
                                    "report_count": st.column_config.NumberColumn("Reports", format="%d"),
                                    "z_score": st.column_config.NumberColumn("Z", format="%.2f")})


# ===========================================================================
# PAGE: INTERACTION NETWORK
# ===========================================================================
@st.cache_data(show_spinner=False)
def network_layout(edges: tuple) -> dict:
    g = nx.Graph()
    g.add_edges_from(edges)
    return {k: tuple(v) for k, v in nx.spring_layout(g, k=0.6, iterations=80, seed=42).items()}


def page_network() -> None:
    page_header("Interaction Network", "Frequently co-reported drug pairs mined with Spark MLlib FP-Growth. "
                                       "Nodes are drugs, edges are co-reported pairs; red edges are high-risk pairs "
                                       "(elevated reaction rate or serious-outcome rate).")
    if interactions_df.empty:
        st.info("Interactions table not computed yet. Run `python src/05_interactions.py`.")
        return

    all_net_drugs = sorted(set(interactions_df["drug_1"]) | set(interactions_df["drug_2"]))
    c_ctrl, c_graph = st.columns([1, 3], gap="medium")
    with c_ctrl, st.container(border=True, key="card_net_ctrl"):
        card_header("Filters")
        max_co = int(interactions_df["pair_report_count"].max())
        min_co = st.slider("Minimum co-reports", 1, max(max_co, 2), min(25, max_co), 1)
        high_risk = st.toggle("High-risk pairs only", value=True)
        focal = st.selectbox("Focus on drug", ["All drugs"] + all_net_drugs)

    net = interactions_df[interactions_df["pair_report_count"] >= min_co]
    if high_risk:
        net = net[net["is_high_risk_interaction"]]
    if focal != "All drugs":
        net = net[(net["drug_1"] == focal) | (net["drug_2"] == focal)]

    with c_ctrl:
        st.write("")
        kpi("Pairs shown", fmt_int(len(net)), f"of {fmt_int(len(interactions_df))} mined pairs", AMBER)
        st.write("")
        kpi("Drugs shown", fmt_int(len(set(net["drug_1"]) | set(net["drug_2"]))), "network nodes", TEAL)

    with c_graph, st.container(border=True, key="card_net_graph"):
        if net.empty:
            card_header("Co-prescription network")
            st.warning("No pairs match these filters. Lower the minimum co-reports or turn off 'High-risk pairs only'.")
        else:
            edges = tuple(sorted(zip(net["drug_1"], net["drug_2"])))
            pos = network_layout(edges)
            g = nx.Graph()
            for r in net.itertuples():
                g.add_edge(r.drug_1, r.drug_2, w=r.pair_report_count, risk=bool(r.is_high_risk_interaction))
            card_header("Co-prescription network", f"{g.number_of_nodes()} drugs · {g.number_of_edges()} pairs · "
                                                   "node size = number of partners · labels on the 25 most connected drugs, hover for the rest")
            traces = []
            for risk, color in [(False, "#CBD5E1"), (True, "rgba(185,28,28,0.45)")]:
                ex, ey = [], []
                for u, v, d in g.edges(data=True):
                    if d["risk"] == risk:
                        ex += [pos[u][0], pos[v][0], None]
                        ey += [pos[u][1], pos[v][1], None]
                if ex:
                    traces.append(go.Scatter(x=ex, y=ey, mode="lines", hoverinfo="skip",
                                             line=dict(width=1.4, color=color)))
            # Invisible midpoint markers so edges are hoverable.
            mx, my, mt = [], [], []
            for u, v, d in g.edges(data=True):
                mx.append((pos[u][0] + pos[v][0]) / 2)
                my.append((pos[u][1] + pos[v][1]) / 2)
                mt.append(f"<b>{esc(u)} + {esc(v)}</b><br>{d['w']:,} co-reports")
            traces.append(go.Scatter(x=mx, y=my, mode="markers", marker=dict(size=8, opacity=0),
                                     hovertext=mt, hoverinfo="text"))
            nodes = list(g.nodes())
            deg = np.array([g.degree(n) for n in nodes])
            label_cut = np.sort(deg)[::-1][min(24, len(deg) - 1)] if len(deg) > 25 else 0
            traces.append(go.Scatter(
                x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes],
                mode="markers+text", text=[n if d >= label_cut else "" for n, d in zip(nodes, deg)],
                textposition="top center",
                textfont=dict(size=10, color=TEXT),
                hovertext=[f"<b>{esc(n)}</b><br>{d} partner drug(s)" for n, d in zip(nodes, deg)], hoverinfo="text",
                marker=dict(size=np.clip(10 + deg * 2.5, 10, 38), color=deg,
                            colorscale=[[0, "#9CC3E6"], [1, NAVY]], line=dict(width=1.5, color="white")),
            ))
            fig = go.Figure(traces)
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            fig = style_fig(fig, 560)
            fig.update_layout(hovermode="closest", margin=dict(l=4, r=4, t=4, b=4))
            show_chart(fig)

    if not net.empty:
        with st.container(border=True, key="card_net_table"):
            card_header("Pairs in view", "Ranked by co-reports")
            st.dataframe(
                net.sort_values("pair_report_count", ascending=False).head(50)
                [["drug_1", "drug_2", "pair_report_count", "avg_reaction_count", "reaction_rate_ratio",
                  "serious_rate", "top_reaction", "is_high_risk_interaction"]],
                hide_index=True, width="stretch", height=380,
                column_config={
                    "drug_1": "Drug 1", "drug_2": "Drug 2",
                    "pair_report_count": st.column_config.NumberColumn("Co-reports", format="%d"),
                    "avg_reaction_count": st.column_config.NumberColumn("Avg reactions", format="%.1f"),
                    "reaction_rate_ratio": st.column_config.NumberColumn("Rate ratio", format="%.2f"),
                    "serious_rate": st.column_config.ProgressColumn("Serious rate", min_value=0.0, max_value=1.0,
                                                                    format="percent"),
                    "top_reaction": "Top reaction",
                    "is_high_risk_interaction": st.column_config.CheckboxColumn("High risk"),
                },
            )


# ===========================================================================
# PAGE: ARCHITECTURE
# ===========================================================================
def page_architecture() -> None:
    page_header("Architecture", "How SafeDose turns raw FAERS JSON into the tables behind this dashboard. "
                                "Spark runs on Hadoop 3.5 client libraries (FileSystem API, output committers, "
                                "winutils on Windows) and stores everything as Snappy-compressed Parquet.")

    kpi_row([
        ("Compute engine", "Apache Spark", "PySpark, local[4]", NAVY),
        ("Hadoop client", "3.5.0", "FileSystem API + committers", TEAL),
        ("Storage format", "Parquet", "Snappy-compressed columnar", AMBER),
        ("Presentation", "Streamlit", "Plotly interactive charts", SLATE),
    ])
    st.write("")

    steps = [
        ("02", "Ingest", "src/02_download.py", "Download openFDA drug-event files into data/raw and validate counts.", None),
        ("03", "Clean & standardise", "src/03_clean.py",
         "Deduplicate reports, keep primary-suspect drugs, normalise drug/reaction names and demographics.", "clean"),
        ("04", "Signal detection", "src/04_signals.py", "PRR, ROR (95% CI) and Yates χ² per drug–reaction pair; Evans flag.", "signals"),
        ("05", "Interaction mining", "src/05_interactions.py", "FP-Growth over co-reported drugs; flags high-risk pairs.", "interactions"),
        ("06", "Trend & spike detection", "src/06_trends.py", "Monthly series, baseline μ/σ, Z-score surge flags.", "trends"),
        ("07", "Hadoop MapReduce showcase", "src/07_hadoop_showcase.py",
         "Map → shuffle/sort → reduce entity counts via the Hadoop FileSystem API.", "mapreduce"),
    ]
    c1, c2 = st.columns([3, 2], gap="medium")
    with c1, st.container(border=True, key="card_arch_steps"):
        card_header("Pipeline", "Each step reads the previous step's Parquet output from data/processed/")
        rows_html = ""
        for num, name, script, desc, table in steps:
            if table is None:
                status = f'<span class="step-status" style="color:{SLATE}">Raw input</span>'
            elif not data[table].empty:
                status = f'<span class="step-status" style="color:{TEAL}">● {fmt_int(len(data[table]))} rows</span>'
            else:
                status = f'<span class="step-status" style="color:{RED}">● Not run</span>'
            rows_html += (f'<div class="step"><div class="step-num">{num}</div><div class="step-body">'
                          f'<div class="step-name">{esc(name)} <code style="font-size:.72rem">{esc(script)}</code></div>'
                          f'<div class="step-desc">{esc(desc)}</div></div>{status}</div>')
        st.markdown(rows_html, unsafe_allow_html=True)

    with c2, st.container(border=True, key="card_arch_stack"):
        card_header("Technology stack")
        st.markdown(
            """
| Layer | Component |
| :-- | :-- |
| Presentation | Streamlit · Plotly · NetworkX |
| Analytics | Spark SQL · Spark MLlib (FP-Growth) |
| Execution | Apache Spark (in-memory DAG) |
| Storage I/O | Hadoop FileSystem API · FileOutputCommitter |
| OS layer | winutils.exe · hadoop.dll |
| Format | Apache Parquet (Snappy) |
            """
        )
        st.markdown(
            r"""
**MapReduce pattern used in step 07**

$(\text{drug}, \text{reaction}) \rightarrow (\text{key}, 1) \xrightarrow{\text{shuffle/sort}} \text{reduceByKey}(a+b)$
            """
        )

    if not mr_df.empty:
        with st.container(border=True, key="card_arch_mr"):
            card_header("MapReduce job output", "Top 25 entities by event count")
            top = mr_df.sort_values("event_count", ascending=False).head(25)
            cc1, cc2 = st.columns([3, 2], gap="medium")
            with cc1:
                show_chart(hbar(top.head(15), "event_count", "item_name", NAVY, 420, x_title="Events"))
            with cc2:
                st.dataframe(top, hide_index=True, width="stretch", height=420,
                             column_config={"item_name": "Entity (key)",
                                            "event_count": st.column_config.NumberColumn("Count (value)", format="%d")})


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
{
    "Overview": page_overview,
    "Drug Explorer": page_explorer,
    "Safety Signals": page_signals,
    "Compare Drugs": page_compare,
    "Trends & Spikes": page_trends,
    "Interaction Network": page_network,
    "Architecture": page_architecture,
}[page]()

st.markdown(
    f'<div style="margin-top:2rem;padding-top:.8rem;border-top:1px solid {GRID};font-size:.75rem;color:{SLATE}">'
    "SafeDose · Data: openFDA / FDA Adverse Event Reporting System (FAERS). "
    "For research and educational use only, not for clinical decision-making.</div>",
    unsafe_allow_html=True,
)
