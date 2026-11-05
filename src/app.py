"""
SafeDose - Drug Safety Signal Detection Dashboard
=================================================
Streamlit + Plotly dashboard over the pre-computed Parquet tables in
data/processed/ (produced by src/03_clean.py ... src/07_hadoop_showcase.py).

Run from the project root:
    streamlit run src/app.py
"""

from __future__ import annotations

import html
import json
import os
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="SafeDose | Drug Safety Signals",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
BG = "#131313"
SURFACE = "#1B1B1B"
SURFACE2 = "#222322"
LINE = "#2C2D2C"
GREEN = "#4DB51F"
GREEN2 = "#7BD84A"
TEXT = "#ECECEC"
MUTED = "#A3A3A3"
DIM = "#6E6E6E"
BLUE = "#5AA9E6"
AMBER = "#E8A23A"
RED = "#E5534B"
GRID = "#262726"
FONT = "Roboto, system-ui, sans-serif"
FONT_C = "'Roboto Condensed', Roboto, sans-serif"

DRUG_A = GREEN
DRUG_B = BLUE

PLOT_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d", "toggleSpikelines"],
}

# Inline icons (Lucide, MIT licence) so labels match the reference look.
_ICON_PATHS = {
    "file": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/>',
    "alert": '<path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "activity": '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    "link": '<path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>',
    "trend_up": '<path d="M22 7 13.5 15.5 8.5 10.5 2 17"/><path d="M16 7h6v6"/>',
    "trend_down": '<path d="M22 17 13.5 8.5 8.5 13.5 2 7"/><path d="M16 17h6v-6"/>',
    "pill": '<path d="m10.5 20.5 10-10a4.95 4.95 0 1 0-7-7l-10 10a4.95 4.95 0 1 0 7 7Z"/><path d="m8.5 8.5 7 7"/>',
    "users": '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    "user": '<path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
    "calendar": '<rect x="3" y="4" width="18" height="18" rx="2"/><path d="M16 2v4M8 2v4M3 10h18"/>',
    "chart": '<path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/>',
    "gauge": '<path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/>',
    "layers": '<path d="m12 2 10 5-10 5L2 7z"/><path d="m2 17 10 5 10-5"/><path d="m2 12 10 5 10-5"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/><path d="M3 12c0 1.66 4 3 9 3s9-1.34 9-3"/>',
    "up": '<path d="M12 19V5"/><path d="m5 12 7-7 7 7"/>',
    "down": '<path d="M12 5v14"/><path d="m19 12-7 7-7-7"/>',
    "arrow_right": '<path d="M5 12h14"/><path d="m12 5 7 7-7 7"/>',
    "ring": '<circle cx="12" cy="12" r="8"/>',
}


def icon(name: str, size: int = 15, color: str = "currentColor", stroke: float = 2.0) -> str:
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" '
            f'stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
            f'{_ICON_PATHS[name]}</svg>')


st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Condensed:wght@400;500;600;700&family=Roboto:wght@400;500;700&display=swap');

    html, body, .stApp, .stMarkdown, input, textarea, select, button, [data-testid="stWidgetLabel"] {{
        font-family: {FONT};
    }}
    .stApp {{ background: {BG}; }}
    .block-container {{ padding-top: 1rem; padding-bottom: 3rem; max-width: 1400px; }}
    [data-testid="stHeader"] {{ background: transparent; height: 0; }}
    [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {{ display: none; }}
    #MainMenu, footer {{ visibility: hidden; }}
    h1, h2, h3, h4 {{ font-family: {FONT_C} !important; }}
    [data-testid="stWidgetLabel"] p {{ color: {MUTED}; font-size: .86rem; }}

    /* ---------- top bar ---------- */
    .brand {{ display:flex; align-items:center; gap:.6rem; height:52px; }}
    .brand-mark {{ width:34px; height:34px; border-radius:50%; background:{GREEN}; display:flex; align-items:center;
                   justify-content:center; }}
    .brand-name {{ font-family:{FONT_C}; font-weight:700; font-size:1.45rem; color:{TEXT}; letter-spacing:-.01em; }}
    .st-key-navbar {{ background:{SURFACE}; border:1px solid {LINE}; border-radius:999px; padding:5px 6px; }}
    .st-key-navbar [data-testid="stButtonGroup"] {{ gap:2px; flex-wrap:wrap; }}
    .st-key-navbar button {{ border:none !important; border-radius:999px !important; background:transparent !important;
                             color:{MUTED} !important; padding:.35rem .95rem !important; min-height:38px; }}
    .st-key-navbar button:hover {{ color:{TEXT} !important; background:{SURFACE2} !important; }}
    .st-key-navbar button[aria-checked="true"] {{
        background:{GREEN} !important; color:#fff !important; }}
    .st-key-navbar button p {{ font-family:{FONT_C} !important; font-size:1rem !important; }}
    .st-key-navbar button[aria-checked="true"] p {{ font-weight:600 !important; color:#fff !important; }}
    .topgap {{ height: 1.6rem; }}

    /* ---------- buttons ---------- */
    [data-testid="stBaseButton-primary"] {{ background:{GREEN}; border-color:{GREEN}; color:#fff; border-radius:999px;
        min-height:54px; padding:0 1.6rem; }}
    [data-testid="stBaseButton-primary"]:hover {{ background:#43A019; border-color:#43A019; color:#fff; }}
    [data-testid="stBaseButton-secondary"] {{ background:{SURFACE}; border:1px solid #3A3B3A; color:{TEXT}; border-radius:999px;
        min-height:54px; padding:0 1.6rem; }}
    [data-testid="stBaseButton-secondary"]:hover {{ border-color:{GREEN}; color:{TEXT}; }}
    [data-testid="stBaseButton-primary"] p, [data-testid="stBaseButton-secondary"] p {{
        font-family:{FONT_C}; font-size:1.12rem; font-weight:500; }}
    .st-key-dl [data-testid="stBaseButton-secondary"] {{ min-height:40px; }}
    .st-key-dl [data-testid="stBaseButton-secondary"] p {{ font-size:.98rem; }}

    /* ---------- hero ---------- */
    .eyebrow {{ color:{GREEN}; font-family:{FONT_C}; font-size:.95rem; letter-spacing:.08em; text-transform:uppercase; margin:0 0 1rem; }}
    .hero-h {{ font-family:{FONT_C}; font-weight:700; color:#F2F2F2; font-size:clamp(2.4rem, 4.6vw, 4.1rem);
               line-height:1.04; letter-spacing:-.015em; margin:0 0 1.2rem; }}
    .hero-p {{ color:#BDBDBD; font-family:{FONT_C}; font-size:1.25rem; line-height:1.5; max-width:34em; margin:0 0 1.8rem; }}
    .flowstrip {{ display:flex; align-items:center; gap:1rem; flex-wrap:wrap; border:1px solid {LINE}; background:#171717;
                  border-radius:18px; padding:.85rem 1.3rem; margin-top:1.6rem; width:fit-content; max-width:100%; }}
    .flowstrip .st {{ font-family:{FONT_C}; font-weight:600; font-size:1.05rem; color:{TEXT}; line-height:1.15; }}
    .flowstrip .st small {{ display:block; font-weight:400; font-size:.82rem; color:{MUTED}; }}
    .flowstrip svg {{ flex:0 0 auto; }}

    .pulse {{ background:{SURFACE}; border:1px solid rgba(77,181,31,.55); border-radius:22px; padding:1.6rem 1.7rem 1.4rem; }}
    .pulse-top {{ display:flex; justify-content:space-between; align-items:center; gap:1rem; }}
    .pulse-k {{ font-family:{FONT_C}; color:{MUTED}; font-size:.92rem; letter-spacing:.06em; text-transform:uppercase; }}
    .live {{ display:flex; align-items:center; gap:.45rem; color:{GREEN2}; font-family:{FONT_C}; font-size:.92rem; white-space:nowrap; }}
    .live i {{ width:10px; height:10px; border-radius:50%; background:{GREEN}; box-shadow:0 0 0 3px rgba(77,181,31,.25); }}
    .pulse-mid {{ display:flex; gap:1.1rem; align-items:center; margin:1.3rem 0 1.4rem; }}
    .pulse-ic {{ flex:0 0 64px; height:64px; border-radius:50%; background:#2E2F2E; display:flex; align-items:center; justify-content:center; }}
    .pulse-h {{ font-family:{FONT_C}; font-weight:700; font-size:1.9rem; color:#F2F2F2; line-height:1.1; }}
    .pulse-s {{ color:#C9C9C9; font-family:{FONT_C}; font-size:1.02rem; margin-top:.25rem; line-height:1.4; }}
    .pulse .stat {{ padding:0 1rem; }}
    .pulse .stat:first-child {{ padding-left:0; }}

    /* ---------- stat strip ---------- */
    .stats {{ display:grid; grid-template-columns:repeat(var(--n), minmax(0,1fr)); }}
    .stats.boxed {{ background:{SURFACE}; border:1px solid rgba(77,181,31,.55); border-radius:20px; padding:1.25rem .4rem; }}
    .stat {{ padding:0 1.2rem; min-width:0; }}
    .stat + .stat {{ border-left:1px solid {LINE}; }}
    .stat-l {{ display:flex; align-items:center; gap:.4rem; color:{MUTED}; font-family:{FONT_C}; font-size:.98rem;
               white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .stat-v {{ display:flex; align-items:center; gap:.55rem; margin-top:.3rem; flex-wrap:wrap; }}
    .stat-v b {{ font-family:{FONT_C}; font-weight:700; font-size:2rem; color:#F4F4F4; line-height:1.1; white-space:nowrap; }}
    .stat-v .vs {{ color:{DIM}; font-family:{FONT_C}; font-size:1rem; }}
    .stat-d {{ display:flex; align-items:center; gap:.25rem; margin-top:.3rem; font-family:{FONT_C}; font-size:.9rem; color:{MUTED};
               white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .stat-d .up, .stat-d .down {{ display:inline-flex; align-items:center; gap:.15rem; font-weight:600; }}
    .good {{ color:{GREEN2}; }} .bad {{ color:{AMBER}; }} .neutral {{ color:{MUTED}; }}
    @media (max-width: 900px) {{
      .stats {{ grid-template-columns:repeat(2, minmax(0,1fr)); row-gap:1rem; }}
      .stat + .stat {{ border-left:none; }}
      .stat {{ padding:0 .9rem; }}
    }}

    .pill {{ display:inline-block; font-family:{FONT_C}; font-size:.85rem; font-weight:500; padding:.12rem .6rem; border-radius:8px;
             white-space:nowrap; }}
    .pill-green {{ background:rgba(77,181,31,.18); color:{GREEN2}; }}
    .pill-red {{ background:rgba(229,83,75,.18); color:#FF8A84; }}
    .pill-amber {{ background:rgba(232,162,58,.18); color:#F3B862; }}
    .pill-blue {{ background:rgba(90,169,230,.18); color:#86C3F0; }}
    .pill-grey {{ background:#2A2B2A; color:{MUTED}; }}

    /* ---------- page header ---------- */
    .ctx {{ color:{MUTED}; font-family:{FONT_C}; font-size:1rem; margin:0 0 .35rem; }}
    .ptitle {{ font-family:{FONT_C}; font-weight:700; color:#F2F2F2; font-size:clamp(2rem, 3.4vw, 2.9rem); line-height:1.08;
               letter-spacing:-.01em; margin:0 0 .8rem; }}
    .meta {{ display:flex; gap:1.6rem; flex-wrap:wrap; margin-bottom:.9rem; }}
    .meta span {{ display:flex; align-items:center; gap:.4rem; color:#CFCFCF; font-family:{FONT_C}; font-size:1.05rem; }}
    .plead {{ color:{MUTED}; font-size:.98rem; line-height:1.6; max-width:76ch; margin:0 0 1.3rem; }}

    /* ---------- cards ---------- */
    [class*="st-key-card"] {{ background:{SURFACE}; border:1px solid {LINE} !important; border-radius:18px !important; }}
    .card-t {{ font-family:{FONT_C}; font-weight:600; font-size:1.2rem; color:{TEXT}; margin:0; }}
    .card-c {{ color:{MUTED}; font-size:.86rem; margin:.15rem 0 .55rem; }}
    .section {{ font-family:{FONT_C}; font-weight:700; font-size:1.65rem; color:#F2F2F2; margin:2.2rem 0 .2rem; }}
    .section-c {{ color:{MUTED}; font-size:.95rem; margin:0 0 1rem; max-width:76ch; }}

    /* ---------- findings ---------- */
    .finds {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(330px, 1fr)); gap:14px; }}
    .find {{ background:{SURFACE}; border:1px solid {LINE}; border-radius:18px; padding:1.15rem 1.25rem 1.2rem; }}
    .find-top {{ display:flex; justify-content:space-between; align-items:center; gap:.6rem; margin-bottom:.6rem; color:{MUTED}; }}
    .find-n {{ font-family:{FONT_C}; font-weight:700; font-size:1.8rem; color:#F4F4F4; line-height:1.1; }}
    .find-h {{ font-family:{FONT_C}; font-weight:600; font-size:1.15rem; color:{TEXT}; margin:.15rem 0 .35rem; line-height:1.25; }}
    .find-p {{ color:{MUTED}; font-size:.9rem; line-height:1.55; margin:0; }}

    .note {{ display:flex; gap:.7rem; align-items:flex-start; background:#1A1712; border:1px solid #3B3020; border-radius:14px;
             padding:.8rem 1rem; color:#D7C29A; font-size:.88rem; line-height:1.5; margin:1.4rem 0 0; }}
    .note svg {{ flex:0 0 auto; margin-top:2px; }}

    .stack {{ display:grid; grid-template-columns:150px 1fr; border-top:1px solid {LINE}; }}
    .stack div {{ padding:.6rem .2rem; border-bottom:1px solid {LINE}; font-size:.9rem; color:{MUTED}; }}
    .stack .k {{ font-family:{FONT_C}; color:{TEXT}; font-weight:600; font-size:1rem; }}
    .stack b {{ color:{TEXT}; font-weight:500; }}

    .foot {{ margin-top:2.4rem; padding-top:1rem; border-top:1px solid {LINE}; font-size:.8rem; color:{DIM}; }}
    div[data-testid="stDataFrame"] {{ border-radius:10px; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def esc(v) -> str:
    return html.escape(str(v))


def fmt_int(n) -> str:
    return f"{int(n):,}"


def pct(mask: pd.Series) -> float:
    return float(mask.mean() * 100.0) if len(mask) else 0.0


def pill(text: str, tone: str = "grey") -> str:
    return f'<span class="pill pill-{tone}">{esc(text)}</span>'


def delta_html(change: float | None, unit: str, label: str, higher_is_bad: bool = True, decimals: int = 1) -> str:
    """Arrow + change text, coloured by whether the direction is good for drug safety."""
    if change is None or not np.isfinite(change):
        return f'<div class="stat-d">{esc(label)}</div>'
    direction = "up" if change > 0 else "down"
    if abs(change) < 0.05:
        tone = "neutral"
    elif higher_is_bad:
        tone = "bad" if change > 0 else "good"
    else:
        tone = "good" if change > 0 else "bad"
    sign = "+" if change > 0 else ""
    return (f'<div class="stat-d"><span class="{direction} {tone}">{icon(direction, 14)}{sign}{change:.{decimals}f}{unit}</span>'
            f'&nbsp;{esc(label)}</div>')


def stat(label: str, value: str, ic: str = "chart", pill_html: str = "", sub_html: str = "") -> str:
    return (f'<div class="stat"><div class="stat-l">{icon(ic)}{esc(label)}</div>'
            f'<div class="stat-v"><b>{esc(value)}</b>{pill_html}</div>{sub_html}</div>')


def sub(text: str) -> str:
    return f'<div class="stat-d">{esc(text)}</div>'


def stat_strip(items: list[str], boxed: bool = True) -> None:
    st.markdown(f'<div class="stats {"boxed" if boxed else ""}" style="--n:{len(items)}">{"".join(items)}</div>',
                unsafe_allow_html=True)


def page_header(context: str, title: str, meta: list[tuple[str, str]], lead: str = "") -> None:
    meta_html = "".join(f"<span>{icon(i, 17)}{esc(t)}</span>" for i, t in meta)
    st.markdown(
        f'<div class="ctx">{esc(context)}</div><div class="ptitle">{esc(title)}</div>'
        f'<div class="meta">{meta_html}</div>' + (f'<p class="plead">{lead}</p>' if lead else ""),
        unsafe_allow_html=True,
    )


def card_header(title: str, caption: str = "") -> None:
    st.markdown(f'<div class="card-t">{esc(title)}</div><div class="card-c">{caption or "&nbsp;"}</div>',
                unsafe_allow_html=True)


def gap(rem: float = 1.0) -> None:
    st.markdown(f'<div style="height:{rem}rem"></div>', unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int = 340, legend: bool = False) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=8, r=16, t=10, b=8),
        font=dict(family=FONT, size=12, color=MUTED),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=legend,
        hoverlabel=dict(bgcolor=SURFACE2, bordercolor=GREEN, font=dict(family=FONT, size=12, color=TEXT)),
        coloraxis_showscale=False,
        bargap=0.3,
    )
    if legend:
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title_text="",
                                      font=dict(color=MUTED)))
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, title_font=dict(size=11, color=DIM),
                     tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, title_font=dict(size=11, color=DIM),
                     tickfont=dict(color="#C9C9C9"))
    return fig


def show_chart(fig: go.Figure, key: str | None = None) -> None:
    st.plotly_chart(fig, width="stretch", theme=None, config=PLOT_CONFIG, key=key)


def hbar(df: pd.DataFrame, x: str, y: str, color: str, height: int = 340,
         text: str | None = None, x_title: str = "Reports") -> go.Figure:
    fig = px.bar(df, x=x, y=y, orientation="h", text=text)
    fig.update_traces(marker_color=color, marker_line_width=0, marker_cornerradius=4,
                      hovertemplate="<b>%{y}</b><br>" + x_title + ": %{x:,}<extra></extra>")
    if text:
        fig.update_traces(texttemplate="%{text}%", textposition="outside", cliponaxis=False,
                          textfont=dict(size=11, color=MUTED))
    fig.update_layout(yaxis=dict(categoryorder="total ascending", title=None), xaxis_title=x_title)
    return style_fig(fig, height)


def donut(labels, values, colors, height: int = 280) -> go.Figure:
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.66, sort=False,
                           marker=dict(colors=colors, line=dict(color=SURFACE, width=3)),
                           textinfo="percent", textfont=dict(size=12, color="#F4F4F4"),
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


def num_cols(**cols) -> dict:
    """Shorthand for dataframe column configs: name=(label, printf format or None)."""
    out = {}
    for k, (label, f) in cols.items():
        out[k] = st.column_config.NumberColumn(label, format=f) if f else label
    return out


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Set SAFEDOSE_DATA_DIR to view another dataset folder (e.g. a small sample) instead of data/.
DATA_ROOT = Path(os.environ["SAFEDOSE_DATA_DIR"]) if os.environ.get("SAFEDOSE_DATA_DIR") else PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_ROOT / "processed"
RAW_DIR = DATA_ROOT / "raw"
ASSETS_DIR = Path(__file__).resolve().parent / "assets"

TABLES = {
    "clean": PROCESSED_DIR / "clean_reports.parquet",
    "signals": PROCESSED_DIR / "signals.parquet",
    "interactions": PROCESSED_DIR / "interactions.parquet",
    "trends": PROCESSED_DIR / "trends.parquet",
    "mapreduce": PROCESSED_DIR / "hadoop_mr_counts",
}
SERIOUS_MAP = {"1": "Serious", "2": "Non-serious"}
SEX_MAP = {"1": "Male", "2": "Female", "0": "Unknown"}
# MedDRA terms that describe how a product was used rather than a clinical reaction.
PRODUCT_USE_TERMS = {
    "off label use", "drug ineffective", "product use issue", "drug dose omission by device",
    "product dose omission issue", "inappropriate schedule of product administration",
    "wrong technique in product usage process", "intentional product use issue", "product use in unapproved indication",
    "incorrect dose administered", "drug administered to patient of inappropriate age", "device difficult to use",
    "product storage error", "underdose", "no adverse event", "therapy interrupted", "therapeutic product effect incomplete",
}


@st.cache_data(show_spinner="Loading safety tables...")
def load_tables() -> dict[str, pd.DataFrame]:
    out = {name: (pd.read_parquet(p) if p.exists() else pd.DataFrame()) for name, p in TABLES.items()}
    clean = out["clean"]
    if not clean.empty:
        clean["seriousness"] = clean["seriousness"].astype("string")
        clean["sex"] = clean["sex"].astype("string")
        clean["age"] = pd.to_numeric(clean["age"], errors="coerce")
        clean["receive_dt"] = pd.to_datetime(clean["receivedate"], format="%Y%m%d", errors="coerce")
    sig = out["signals"]
    if not sig.empty:
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
    monthly = (
        reports.dropna(subset=["receive_dt"])
        .assign(month=lambda d: d["receive_dt"].dt.to_period("M").dt.to_timestamp())
        .groupby("month")
        .agg(reports=("safetyreportid", "size"), serious=("seriousness", lambda s: (s == "1").sum()))
        .reset_index()
    )
    return {
        "reports": reports,
        "drug_counts": clean.groupby("drug_name")["safetyreportid"].nunique().sort_values(ascending=False),
        "rx_counts": clean.groupby("reaction_name")["safetyreportid"].nunique().sort_values(ascending=False),
        "monthly": monthly,
        "date_min": reports["receive_dt"].min(),
        "date_max": reports["receive_dt"].max(),
    }


def dir_bytes(p: Path) -> int:
    if not p.exists():
        return 0
    return p.stat().st_size if p.is_file() else sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


data = load_tables()
clean_df, signals_df = data["clean"], data["signals"]
interactions_df, trends_df, mr_df = data["interactions"], data["trends"], data["mapreduce"]

if clean_df.empty or signals_df.empty:
    st.error("No processed tables found in data/processed. Run the pipeline from the project root, then reload:\n\n"
             "```\npython src/03_clean.py\npython src/04_signals.py\npython src/05_interactions.py\n"
             "python src/06_trends.py\npython src/07_hadoop_showcase.py\n```")
    st.stop()

D = derived(clean_df)
reports_df, drug_counts = D["reports"], D["drug_counts"]
ALL_DRUGS = drug_counts.index.tolist()
WINDOW = (f"{D['date_min']:%b %Y} to {D['date_max']:%b %Y}"
          if pd.notna(D["date_min"]) and pd.notna(D["date_max"]) else "date range unavailable")


def default_index(options: list, preferred: str, fallback: int = 0) -> int:
    return options.index(preferred) if preferred in options else min(fallback, len(options) - 1)


# ---------------------------------------------------------------------------
# Top bar navigation
# ---------------------------------------------------------------------------
PAGES = ["Overview", "Drug Explorer", "Safety Signals", "Compare Drugs", "Trends & Spikes",
         "Interaction Network", "Architecture"]

if "page" not in st.session_state:
    st.session_state.page = "Overview"
if "_last_page" not in st.session_state:
    st.session_state._last_page = "Overview"


def _keep_page() -> None:
    # Segmented controls allow deselecting; snap back to the last page instead of showing nothing.
    if st.session_state.page is None:
        st.session_state.page = st.session_state._last_page
    st.session_state._last_page = st.session_state.page


def goto(page_name: str) -> None:
    st.session_state.page = page_name
    st.session_state._last_page = page_name


bar_l, bar_r = st.columns([1, 5], vertical_alignment="center")
with bar_l:
    st.markdown(f'<div class="brand"><div class="brand-mark">{icon("pill", 18, "#fff", 2.2)}</div>'
                f'<div class="brand-name">SafeDose</div></div>', unsafe_allow_html=True)
with bar_r, st.container(key="navbar"):
    st.segmented_control("Page", PAGES, key="page", on_change=_keep_page, label_visibility="collapsed")
page = st.session_state.page or "Overview"
st.markdown('<div class="topgap"></div>', unsafe_allow_html=True)


# ===========================================================================
# OVERVIEW
# ===========================================================================
def pulse_numbers() -> dict:
    m = D["monthly"]
    out: dict = {"month": None}
    if len(m) < 2:
        return out
    last, prev = m.iloc[-1], m.iloc[-2]
    out["month"] = last["month"]
    out["prev_label"] = f"vs {prev['month']:%b}"
    out["reports"] = int(last["reports"])
    out["reports_chg"] = (last["reports"] - prev["reports"]) / max(prev["reports"], 1) * 100
    s_last = last["serious"] / max(last["reports"], 1) * 100
    s_prev = prev["serious"] / max(prev["reports"], 1) * 100
    out["serious"] = s_last
    out["serious_chg"] = s_last - s_prev
    out["spikes"] = out["spikes_prev"] = 0
    if not trends_df.empty:
        sp = trends_df[trends_df["is_spike"]]
        out["spikes"] = int((sp["month_dt"] == last["month"]).sum())
        out["spikes_prev"] = int((sp["month_dt"] == prev["month"]).sum())
    return out


def findings() -> list[dict]:
    items = []
    m = D["monthly"]
    if len(m) >= 3:
        last3 = m.tail(3)
        share = last3["reports"].sum() / max(m["reports"].sum(), 1) * 100
        items.append(dict(tag=("Data coverage", "blue"), ic="calendar", n=f"{share:.0f}%",
                          h=f"of reports arrived {last3['month'].min():%b} to {last3['month'].max():%b %Y}",
                          p="The download covers one reporting quarter. Earlier receive dates are follow-ups and "
                            "late submissions, so long-term trends before that quarter are thin."))

    s_rate = pct(reports_df["seriousness"] == "1")
    by_drug = (clean_df.drop_duplicates(["safetyreportid", "drug_name"])
               .groupby("drug_name").agg(n=("safetyreportid", "size"), s=("seriousness", lambda x: (x == "1").mean())))
    by_drug = by_drug[by_drug["n"] >= 200].sort_values("s", ascending=False)
    top_s = (f" Among drugs with 200 or more reports, {by_drug.index[0]} has the highest serious rate at "
             f"{by_drug['s'].iloc[0] * 100:.0f}%.") if len(by_drug) else ""
    items.append(dict(tag=("Seriousness", "red" if s_rate >= 50 else "amber"), ic="alert", n=f"{s_rate:.1f}%",
                      h="of reports describe a serious outcome",
                      p="Serious means death, a life-threatening event, hospital admission, disability or a birth "
                        "defect." + top_s))

    top10 = D["rx_counts"].head(10)
    pu = [r for r in top10.index if r in PRODUCT_USE_TERMS]
    if pu:
        items.append(dict(tag=("Reaction mix", "amber"), ic="activity", n=f"{len(pu)} of 10",
                          h="top reaction terms describe how a drug was used",
                          p=f"Terms such as {', '.join(pu[:3])} record use or effectiveness problems rather than a "
                            "toxic reaction. Filter them out when looking for new side effects."))

    strong = signals_df[signals_df["is_signal"] & (signals_df["report_count"] >= 20)]
    if not strong.empty:
        r = strong.sort_values("ror_lower_ci", ascending=False).iloc[0]
        items.append(dict(tag=("Signal", "red"), ic="activity", n=f"{r['prr']:.0f}x",
                          h=f"{r['drug_name']} and {r['reaction_name']}",
                          p=f"{fmt_int(r['report_count'])} of {fmt_int(r['total_drug_reports'])} reports for this drug "
                            f"mention it, {r['prr']:.0f} times the rate for all other drugs. It is the strongest signal "
                            "backed by at least 20 reports."))

    if not interactions_df.empty:
        hi = interactions_df[interactions_df["pair_report_count"] >= 25].sort_values("reaction_rate_ratio", ascending=False)
        if not hi.empty:
            r = hi.iloc[0]
            n_hr = int(interactions_df["is_high_risk_interaction"].sum())
            items.append(dict(tag=("Interaction", "amber"), ic="link", n=f"{r['reaction_rate_ratio']:.1f}x",
                              h=f"more reactions when {r['drug_1']} and {r['drug_2']} are reported together",
                              p=f"Across {fmt_int(r['pair_report_count'])} shared reports. {n_hr} of "
                                f"{len(interactions_df)} mined pairs meet the high-risk rule, so the rule flags almost "
                                "every frequent pair and works best as a starting list."))

    if not trends_df.empty and trends_df["is_spike"].any():
        r = trends_df[trends_df["is_spike"]].sort_values("z_score", ascending=False).iloc[0]
        items.append(dict(tag=("Spike", "blue"), ic="trend_up", n=f"Z {r['z_score']:.1f}",
                          h=f"{r['drug_name']} in {pd.to_datetime(r['year_month']):%B %Y}",
                          p=f"{fmt_int(r['report_count'])} reports that month against a usual {r['mean_count']:.1f}. "
                            "It is the largest jump in any monthly series."))

    ages = reports_df["age"].dropna()
    ages = ages[(ages >= 0) & (ages <= 105)]
    if len(ages):
        items.append(dict(tag=("Patients", "grey"), ic="users", n=f"{pct(reports_df['sex'] == '2'):.0f}%",
                          h=f"of patients are female, median age {ages.median():.0f}",
                          p=f"Age is recorded on {len(ages) / len(reports_df) * 100:.0f}% of reports, so age "
                            "comparisons rely on that subset."))
    return items


def page_overview() -> None:
    p = pulse_numbers()
    left, right = st.columns([1.12, 1], gap="large", vertical_alignment="center")
    with left:
        st.markdown(
            f'<div class="eyebrow">Post-market drug safety surveillance</div>'
            f'<div class="hero-h">Every report counted.<br>Every signal scored.</div>'
            f'<div class="hero-p">{fmt_int(len(reports_df))} FDA adverse event reports, cleaned with Apache Spark '
            f'and checked for reactions that turn up more often than chance would predict.</div>',
            unsafe_allow_html=True,
        )
        b1, b2, _ = st.columns([1.25, 1.05, 0.4], gap="small")
        b1.button("Review safety signals", type="primary", icon=":material/monitoring:", width="stretch",
                  on_click=goto, args=("Safety Signals",))
        b2.button("Find a drug", type="secondary", icon=":material/search:", width="stretch",
                  on_click=goto, args=("Drug Explorer",))
        arrow = icon("arrow_right", 20, GREEN)
        st.markdown(
            f'<div class="flowstrip"><span class="st">Import</span>{arrow}<span class="st">Clean</span>{arrow}'
            f'<span class="st">Score</span>{arrow}<span class="st">Review<small>in this dashboard</small></span></div>',
            unsafe_allow_html=True,
        )
    with right:
        if p["month"] is not None:
            rising, easing = p["reports_chg"] > 10, p["reports_chg"] < -10
            verdict = "Reporting is rising" if rising else ("Reporting is easing" if easing else "Reporting is steady")
            expl = ("More reports arrived than the month before, so new signals may surface." if rising
                    else ("Fewer reports arrived than the month before." if easing
                          else "Report volume is close to the month before."))
            ic = "trend_up" if rising else ("trend_down" if easing else "activity")
            st.markdown(
                f'<div class="pulse"><div class="pulse-top"><span class="pulse-k">Reporting pulse, '
                f'{p["month"]:%B %Y}</span><span class="live"><i></i>Latest month</span></div>'
                f'<div class="pulse-mid"><div class="pulse-ic">{icon(ic, 30, "#E6E6E6", 2.2)}</div>'
                f'<div><div class="pulse-h">{verdict}</div><div class="pulse-s">{expl}</div></div></div>'
                f'<div class="stats" style="--n:3">'
                + stat("Reports", fmt_int(p["reports"]), "file",
                       sub_html=delta_html(p["reports_chg"], "%", p["prev_label"]))
                + stat("Serious share", f"{p['serious']:.0f}%", "alert",
                       sub_html=delta_html(p["serious_chg"], " pts", p["prev_label"]))
                + stat("Spikes flagged", fmt_int(p["spikes"]), "trend_up",
                       sub_html=delta_html(float(p["spikes"] - p["spikes_prev"]), "", p["prev_label"], decimals=0))
                + "</div></div>",
                unsafe_allow_html=True,
            )

    gap(2.4)
    n_signals = int(signals_df["is_signal"].sum())
    n_hr = int(interactions_df["is_high_risk_interaction"].sum()) if not interactions_df.empty else 0
    n_spk = int(trends_df["is_spike"].sum()) if not trends_df.empty else 0
    s_rate = pct(reports_df["seriousness"] == "1")
    stat_strip([
        stat("Case reports", fmt_int(len(reports_df)), "file", pill("Deduplicated", "green"), sub(WINDOW)),
        stat("Serious outcomes", f"{s_rate:.1f}%", "alert",
             pill("High" if s_rate >= 50 else "Moderate", "red" if s_rate >= 50 else "amber"), sub("of all case reports")),
        stat("Suspect drugs", fmt_int(len(drug_counts)), "pill", pill("Tracked", "blue"),
             sub(f"{fmt_int(clean_df['reaction_name'].nunique())} reaction terms")),
        stat("Safety signals", fmt_int(n_signals), "activity",
             pill(f"{n_signals / len(signals_df) * 100:.0f}% of pairs", "amber"), sub("meet Evans criteria")),
        stat("Volume spikes", fmt_int(n_spk), "trend_up", pill("Z ≥ 3", "blue"), sub(f"{n_hr} high-risk drug pairs")),
    ])

    st.markdown('<div class="section">What the data shows</div>'
                '<div class="section-c">Findings calculated from the processed tables each time the page loads.</div>',
                unsafe_allow_html=True)
    cards = "".join(
        f'<div class="find"><div class="find-top">{icon(f["ic"], 18)}{pill(*f["tag"])}</div>'
        f'<div class="find-n">{esc(f["n"])}</div><div class="find-h">{esc(f["h"])}</div>'
        f'<p class="find-p">{esc(f["p"])}</p></div>'
        for f in findings()
    )
    st.markdown(f'<div class="finds">{cards}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="note">{icon("alert", 17, AMBER)}<div><b>Signals show association, not causation.</b> '
                'They come from voluntary reports and are hypotheses that need clinical review before any '
                'conclusion about a drug.</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section">Reporting at a glance</div><div class="section-c">&nbsp;</div>',
                unsafe_allow_html=True)
    with st.container(border=True, key="card_monthly"):
        card_header("Monthly reporting volume", "Unique case reports received each month, with the serious subset. "
                                                "Double-click the chart to see the full date range.")
        m = D["monthly"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=m["month"], y=m["reports"], name="All reports", mode="lines",
                                 line=dict(color=GREEN, width=2.6), fill="tozeroy", fillcolor="rgba(77,181,31,0.12)",
                                 hovertemplate="%{x|%b %Y}<br>%{y:,} reports<extra></extra>"))
        fig.add_trace(go.Scatter(x=m["month"], y=m["serious"], name="Serious", mode="lines",
                                 line=dict(color=RED, width=1.8, dash="dot"),
                                 hovertemplate="%{x|%b %Y}<br>%{y:,} serious<extra></extra>"))
        active = m[m["reports"] >= max(5, 0.02 * m["reports"].max())]
        if not active.empty:
            fig.update_xaxes(range=[active["month"].min() - pd.DateOffset(months=1),
                                    m["month"].max() + pd.DateOffset(days=15)])
        fig.update_layout(yaxis_title="Reports", xaxis_title=None, hovermode="x unified")
        show_chart(style_fig(fig, 280, legend=True))

    c1, c2 = st.columns(2, gap="medium")
    with c1, st.container(border=True, key="card_topdrugs"):
        card_header("Most reported suspect drugs", "Top 10 by unique case reports")
        top = drug_counts.head(10).rename_axis("drug").reset_index(name="reports")
        show_chart(hbar(top, "reports", "drug", GREEN, 360))
    with c2, st.container(border=True, key="card_toprx"):
        card_header("Most reported reactions", "Top 10 MedDRA terms by unique case reports")
        top = D["rx_counts"].head(10).rename_axis("reaction").reset_index(name="reports")
        show_chart(hbar(top, "reports", "reaction", BLUE, 360))

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1, st.container(border=True, key="card_serious"):
        card_header("Outcome seriousness", "Share of case reports")
        s = reports_df["seriousness"].map(SERIOUS_MAP).fillna("Not specified").value_counts()
        s = s.reindex([k for k in ["Serious", "Non-serious", "Not specified"] if k in s.index])
        show_chart(donut(s.index, s.values, [RED, GREEN, "#4A4B4A"]))
    with c2, st.container(border=True, key="card_sex"):
        card_header("Patient sex", "Share of case reports")
        s = reports_df["sex"].map(SEX_MAP).fillna("Unknown").value_counts()
        s = s.reindex([k for k in ["Female", "Male", "Unknown"] if k in s.index])
        show_chart(donut(s.index, s.values, [BLUE, GREEN, "#4A4B4A"]))
    with c3, st.container(border=True, key="card_age"):
        ages = reports_df["age"].dropna()
        ages = ages[(ages >= 0) & (ages <= 105)]
        card_header("Patient age", f"{fmt_int(len(ages))} reports with age, median {ages.median():.0f} years"
                    if len(ages) else "No age recorded")
        show_chart(age_hist(ages, GREEN))


# ===========================================================================
# DRUG EXPLORER
# ===========================================================================
def page_explorer() -> None:
    with st.container(border=True, key="card_pick_drug"):
        drug = st.selectbox("Choose a drug", ALL_DRUGS, index=default_index(ALL_DRUGS, "adalimumab"),
                            help="Most reported drugs are listed first. Type to search.")
    gap(1)
    recs = clean_df[clean_df["drug_name"] == drug]
    reps = recs.drop_duplicates("safetyreportid")
    n = len(reps)
    ages = reps["age"].dropna()
    ages = ages[(ages >= 0) & (ages <= 105)]
    drug_sig = signals_df[(signals_df["drug_name"] == drug) & signals_df["is_signal"]]
    rank = ALL_DRUGS.index(drug) + 1

    page_header(f"Drug profile, rank {rank:,} of {len(ALL_DRUGS):,} by report volume", drug.title(), [
        ("ring", f"{fmt_int(n)} case reports"),
        ("activity", f"{fmt_int(len(drug_sig))} safety signals"),
        ("calendar", WINDOW),
    ])
    serious = pct(reps["seriousness"] == "1")
    overall = pct(reports_df["seriousness"] == "1")
    if serious > overall + 5:
        s_pill = pill("Above average", "red")
    elif serious < overall - 5:
        s_pill = pill("Below average", "green")
    else:
        s_pill = pill("Average", "blue")
    stat_strip([
        stat("Case reports", fmt_int(n), "file", pill(f"{fmt_int(recs['reaction_name'].nunique())} reactions", "grey")),
        stat("Serious rate", f"{serious:.1f}%", "alert", s_pill, sub(f"all drugs: {overall:.1f}%")),
        stat("Female patients", f"{pct(reps['sex'] == '2'):.0f}%", "users", sub_html=sub(f"male {pct(reps['sex'] == '1'):.0f}%")),
        stat("Mean age", f"{ages.mean():.0f} yrs" if len(ages) else "n/a", "user",
             sub_html=sub(f"{fmt_int(len(ages))} reports with age")),
        stat("Safety signals", fmt_int(len(drug_sig)), "activity",
             pill("Signals found" if len(drug_sig) else "None", "amber" if len(drug_sig) else "green")),
    ])
    gap(1.2)

    c1, c2 = st.columns([3, 2], gap="medium")
    with c1, st.container(border=True, key="card_drug_rx"):
        card_header("Top 15 reactions", "Label shows the share of this drug's case reports")
        rx = (recs.groupby("reaction_name")["safetyreportid"].nunique().sort_values(ascending=False).head(15)
              .rename_axis("reaction").reset_index(name="reports"))
        rx["share"] = (rx["reports"] / max(n, 1) * 100).round(1)
        show_chart(hbar(rx, "reports", "reaction", GREEN, 545, text="share"))
    with c2:
        with st.container(border=True, key="card_drug_age"):
            card_header("Patient age", f"Median {ages.median():.0f} years" if len(ages) else "No age recorded")
            if len(ages):
                show_chart(age_hist(ages, BLUE, 190, nbins=20))
            else:
                st.caption("Reports for this drug do not include patient age.")
        with st.container(border=True, key="card_drug_co"):
            card_header("Often reported with", "Drug pairs found by FP-Growth")
            co = pd.DataFrame()
            if not interactions_df.empty:
                co = interactions_df[(interactions_df["drug_1"] == drug) | (interactions_df["drug_2"] == drug)].copy()
            if co.empty:
                st.caption("No other drug appears with this one often enough to form a frequent pair.")
            else:
                co["partner"] = np.where(co["drug_1"] == drug, co["drug_2"], co["drug_1"])
                co = co.sort_values("pair_report_count", ascending=False).head(6)
                st.dataframe(co[["partner", "pair_report_count", "reaction_rate_ratio", "top_reaction"]],
                             hide_index=True, width="stretch", height=250,
                             column_config=num_cols(partner=("Drug", None), pair_report_count=("Shared reports", "%d"),
                                                    reaction_rate_ratio=("Rate ratio", "%.2f"),
                                                    top_reaction=("Top reaction", None)))

    with st.container(border=True, key="card_drug_sig"):
        card_header("Safety signals for this drug", "Reactions meeting the Evans criteria, strongest first")
        if drug_sig.empty:
            st.caption("No reaction for this drug meets the Evans criteria.")
        else:
            st.dataframe(drug_sig.sort_values("prr", ascending=False)
                         [["reaction_name", "report_count", "prr", "ror", "ror_lower_ci", "ror_upper_ci", "chi_square"]],
                         hide_index=True, width="stretch", height=min(38 + 35 * len(drug_sig), 380),
                         column_config=num_cols(reaction_name=("Reaction", None), report_count=("Reports", "%d"),
                                                prr=("PRR", "%.2f"), ror=("ROR", "%.2f"),
                                                ror_lower_ci=("ROR 95% low", "%.2f"),
                                                ror_upper_ci=("ROR 95% high", "%.2f"), chi_square=("Chi-square", "%.1f")))


# ===========================================================================
# SAFETY SIGNALS
# ===========================================================================
def page_signals() -> None:
    page_header("Disproportionality analysis", "Safety signals", [
        ("ring", f"{fmt_int(len(signals_df))} pairs scored"),
        ("activity", f"{fmt_int(signals_df['is_signal'].sum())} meet Evans criteria"),
    ], lead="Each pair compares how often a reaction is reported with one drug against every other drug. "
            "A signal needs a PRR of at least 2, a chi-square of at least 4 and at least 3 reports. "
            "Change the thresholds to look at weaker or stronger associations.")

    with st.container(border=True, key="card_sig_filters"):
        f1, f2, f3, f4 = st.columns(4, gap="medium")
        drug_q = f1.text_input("Drug name contains", placeholder="adalimumab")
        rx_q = f2.text_input("Reaction contains", placeholder="infection")
        min_prr = f3.slider("Minimum PRR", 1.0, 20.0, 2.0, 0.5)
        min_a = f4.slider("Minimum reports", 1, 50, 3, 1)

    sig = signals_df
    if drug_q.strip():
        sig = sig[sig["drug_name"].str.contains(drug_q.strip().lower(), na=False, regex=False)]
    if rx_q.strip():
        sig = sig[sig["reaction_name"].str.contains(rx_q.strip().lower(), na=False, regex=False)]
    sig = sig[(sig["prr"] >= min_prr) & (sig["chi_square"] >= 4.0) & (sig["report_count"] >= min_a)]
    sig = sig.sort_values("chi_square", ascending=False)

    gap(1.2)
    med = sig["prr"].median() if len(sig) else None
    stat_strip([
        stat("Matching signals", fmt_int(len(sig)), "activity", pill("Filtered", "amber")),
        stat("Drugs", fmt_int(sig["drug_name"].nunique()), "pill"),
        stat("Reactions", fmt_int(sig["reaction_name"].nunique()), "layers"),
        stat("Median PRR", f"{med:.1f}" if med is not None else "n/a", "gauge",
             pill("Strong", "red") if med is not None and med >= 10 else pill("Moderate", "blue")),
    ])
    gap(1.2)
    if sig.empty:
        st.info("No pairs match these filters. Lower the minimum PRR or the minimum reports.")
        return

    with st.container(border=True, key="card_volcano"):
        card_header("Signal map", f"All {len(sig):,} matching pairs. Further right means a larger effect, higher means "
                                  "stronger statistical evidence. Hover for details, drag to zoom.")
        p = sig.copy()
        p["log2_prr"] = np.log2(p["prr"].clip(0.1, 1e5))
        p["log10_chi"] = np.log10(p["chi_square"].clip(1, 1e6))
        tiers = ["PRR 2 to 5", "PRR 5 to 10", "PRR 10 to 100", "PRR 100+"]
        p["strength"] = pd.cut(p["prr"], [0, 5, 10, 100, np.inf], labels=tiers, right=False)
        fig = px.scatter(p, x="log2_prr", y="log10_chi", color="strength", render_mode="webgl",
                         color_discrete_map=dict(zip(tiers, [BLUE, GREEN, AMBER, RED])),
                         category_orders={"strength": tiers},
                         custom_data=["drug_name", "reaction_name", "report_count", "prr", "ror", "chi_square"])
        fig.update_traces(marker=dict(size=6, opacity=0.6, line=dict(width=0)),
                          hovertemplate="<b>%{customdata[0]}</b>: %{customdata[1]}<br>Reports %{customdata[2]:,}"
                                        "<br>PRR %{customdata[3]:.2f}, ROR %{customdata[4]:.2f}"
                                        "<br>Chi-square %{customdata[5]:.1f}<extra></extra>")
        fig.add_vline(x=1.0, line_dash="dash", line_color=DIM, line_width=1, annotation_text="PRR 2",
                      annotation_font_color=MUTED, annotation_font_size=11)
        fig.add_hline(y=np.log10(4.0), line_dash="dash", line_color=DIM, line_width=1, annotation_text="Chi-square 4",
                      annotation_font_color=MUTED, annotation_font_size=11, annotation_position="bottom right")
        fig.update_layout(xaxis_title="log2(PRR), effect size", yaxis_title="log10(chi-square), evidence")
        show_chart(style_fig(fig, 440, legend=True))

    with st.container(border=True, key="card_sig_table"):
        h1, h2 = st.columns([4, 1], vertical_alignment="bottom")
        with h1:
            shown = (f"All {len(sig):,} matching pairs" if len(sig) <= 200
                     else f"Top 200 of {len(sig):,} matching pairs")
            card_header("Signal table", f"{shown}, strongest evidence first")
        with h2, st.container(key="dl"):
            st.download_button("Download CSV", sig.drop(columns=["is_signal"], errors="ignore").to_csv(index=False),
                               file_name="safedose_signals.csv", mime="text/csv", width="stretch",
                               icon=":material/download:")
        gap(0.8)  # room for the table's hover toolbar so it doesn't cover the button
        st.dataframe(sig.head(200)[["drug_name", "reaction_name", "report_count", "prr", "ror", "ror_lower_ci",
                                    "ror_upper_ci", "chi_square"]],
                     hide_index=True, width="stretch", height=420,
                     column_config=num_cols(drug_name=("Drug", None), reaction_name=("Reaction", None),
                                            report_count=("Reports", "%d"), prr=("PRR", "%.2f"), ror=("ROR", "%.2f"),
                                            ror_lower_ci=("ROR 95% low", "%.2f"), ror_upper_ci=("ROR 95% high", "%.2f"),
                                            chi_square=("Chi-square", "%.1f")))


# ===========================================================================
# COMPARE DRUGS
# ===========================================================================
def page_compare() -> None:
    with st.container(border=True, key="card_cmp_pick"):
        c1, c2 = st.columns(2, gap="medium")
        drug_a = c1.selectbox("First drug", ALL_DRUGS, index=default_index(ALL_DRUGS, "adalimumab"), key="cmp_a")
        drug_b = c2.selectbox("Second drug", ALL_DRUGS, index=default_index(ALL_DRUGS, "methotrexate", 1), key="cmp_b")
    gap(1)
    if drug_a == drug_b:
        st.warning("Choose two different drugs to compare.")
        return

    rec_a, rec_b = clean_df[clean_df["drug_name"] == drug_a], clean_df[clean_df["drug_name"] == drug_b]
    rep_a, rep_b = rec_a.drop_duplicates("safetyreportid"), rec_b.drop_duplicates("safetyreportid")

    def mean_age(r):
        a = r["age"].dropna()
        a = a[(a >= 0) & (a <= 105)]
        return f"{a.mean():.0f}" if len(a) else "n/a"

    def dot(c):
        return f'<span style="display:inline-block;width:11px;height:11px;border-radius:50%;background:{c}"></span>'

    def two(label, ic, a, b):
        return (f'<div class="stat"><div class="stat-l">{icon(ic)}{esc(label)}</div><div class="stat-v">'
                f'<b style="color:{DRUG_A}">{esc(a)}</b><span class="vs">vs</span><b style="color:{DRUG_B}">{esc(b)}</b>'
                f'</div></div>')

    page_header("Side-by-side comparison", f"{drug_a.title()} vs {drug_b.title()}", [
        ("ring", f"{fmt_int(len(rep_a))} and {fmt_int(len(rep_b))} case reports"),
    ], lead="Reaction rates are shown as a share of each drug's own reports, so drugs with different volumes "
            "can be compared fairly.")
    stat_strip([
        two("Case reports", "file", fmt_int(len(rep_a)), fmt_int(len(rep_b))),
        two("Serious rate", "alert", f"{pct(rep_a['seriousness'] == '1'):.0f}%", f"{pct(rep_b['seriousness'] == '1'):.0f}%"),
        two("Female share", "users", f"{pct(rep_a['sex'] == '2'):.0f}%", f"{pct(rep_b['sex'] == '2'):.0f}%"),
        two("Mean age", "user", mean_age(rep_a), mean_age(rep_b)),
    ])
    st.markdown(f'<div style="display:flex;gap:1.4rem;margin:.8rem 0 0 .2rem;font-family:{FONT_C};color:{MUTED}">'
                f'<span>{dot(DRUG_A)} {esc(drug_a)}</span><span>{dot(DRUG_B)} {esc(drug_b)}</span></div>',
                unsafe_allow_html=True)
    gap(1)

    def top_rx(rec, k=10):
        return rec.groupby("reaction_name")["safetyreportid"].nunique().sort_values(ascending=False).head(k)

    def share(rec, rep, terms):
        return (rec[rec["reaction_name"].isin(terms)].groupby("reaction_name")["safetyreportid"].nunique()
                / max(len(rep), 1) * 100).round(1)

    union = list(dict.fromkeys(list(top_rx(rec_a).index) + list(top_rx(rec_b).index)))
    comp = pd.DataFrame({"a": share(rec_a, rep_a, union).reindex(union).fillna(0),
                         "b": share(rec_b, rep_b, union).reindex(union).fillna(0)})
    comp = comp.assign(mx=comp.max(axis=1)).sort_values("mx").tail(15)

    with st.container(border=True, key="card_cmp_chart"):
        card_header("Reaction rates", "Share of each drug's reports that mention the reaction, from each drug's top 10")
        fig = go.Figure()
        fig.add_trace(go.Bar(y=comp.index, x=comp["a"], name=drug_a, orientation="h", marker_color=DRUG_A,
                             marker_cornerradius=3, hovertemplate="%{y}<br>" + esc(drug_a) + ": %{x:.1f}%<extra></extra>"))
        fig.add_trace(go.Bar(y=comp.index, x=comp["b"], name=drug_b, orientation="h", marker_color=DRUG_B,
                             marker_cornerradius=3, hovertemplate="%{y}<br>" + esc(drug_b) + ": %{x:.1f}%<extra></extra>"))
        fig.update_layout(barmode="group", xaxis_title="% of case reports", yaxis_title=None, bargroupgap=0.08)
        show_chart(style_fig(fig, max(380, 30 * len(comp) + 80), legend=True))

    c1, c2 = st.columns(2, gap="medium")
    for col, d, key in [(c1, drug_a, "card_cmp_a"), (c2, drug_b, "card_cmp_b")]:
        s = signals_df[(signals_df["drug_name"] == d) & signals_df["is_signal"]]
        with col, st.container(border=True, key=key):
            card_header(f"Strongest signals for {d}", f"{len(s):,} reactions meet the Evans criteria")
            if s.empty:
                st.caption("No reaction for this drug meets the Evans criteria.")
            else:
                st.dataframe(s.sort_values("prr", ascending=False).head(10)[["reaction_name", "report_count", "prr"]],
                             hide_index=True, width="stretch", height=280,
                             column_config=num_cols(reaction_name=("Reaction", None), report_count=("Reports", "%d"),
                                                    prr=("PRR", "%.2f")))


# ===========================================================================
# TRENDS & SPIKES
# ===========================================================================
def page_trends() -> None:
    if trends_df.empty:
        st.info("The trends table is missing. Run `python src/06_trends.py`, then reload this page.")
        return
    spikes_all = trends_df[trends_df["is_spike"]]
    trend_drugs = sorted(trends_df["drug_name"].dropna().unique().tolist())
    spike_drugs = spikes_all.sort_values("z_score", ascending=False)["drug_name"].unique()
    default_drug = spike_drugs[0] if len(spike_drugs) else trend_drugs[0]

    with st.container(border=True, key="card_trend_pick"):
        c1, c2, c3 = st.columns([2, 2, 1], gap="medium", vertical_alignment="bottom")
        only = c3.toggle("Spiking drugs only", value=True)
        options = sorted(spike_drugs.tolist()) if (only and len(spike_drugs)) else trend_drugs
        drug = c1.selectbox("Drug", options, index=default_index(options, default_drug))
        recs = trends_df[trends_df["drug_name"] == drug]
        rx_opts = ["All reactions"] + sorted(r for r in recs["reaction_name"].unique() if r != "ALL")
        rx_sel = c2.selectbox("Reaction", rx_opts)
    gap(1)

    rx = "ALL" if rx_sel == "All reactions" else rx_sel
    series = recs[recs["reaction_name"] == rx].sort_values("month_dt")
    if series.empty:
        st.info("This drug has no monthly series for that reaction. Choose All reactions.")
        return
    mu, sd = float(series["mean_count"].iloc[0]), float(series["std_count"].iloc[0])
    thr = mu + 3 * sd
    sp = series[series["is_spike"]]

    page_header("Monthly volume and surge detection", drug.title(), [
        ("ring", "All reactions" if rx == "ALL" else rx),
        ("calendar", f"{series['month_dt'].min():%b %Y} to {series['month_dt'].max():%b %Y}"),
    ], lead="A spike is a month at least three standard deviations above the series' own average, with at least "
            "three reports. Sudden jumps can point to a new problem, a product recall or a reporting campaign.")
    stat_strip([
        stat("Months observed", fmt_int(len(series)), "calendar"),
        stat("Usual month", f"{mu:.1f}", "chart", sub_html=sub("mean reports per month")),
        stat("Spike line", f"{thr:.1f}", "gauge", sub_html=sub(f"mean + 3 SD, SD {sd:.1f}")),
        stat("Spikes", fmt_int(len(sp)), "trend_up", pill("Surge", "red") if len(sp) else pill("None", "green"),
             sub(f"largest Z {sp['z_score'].max():.1f}" if len(sp) else "no month above the line")),
    ])
    gap(1.2)

    with st.container(border=True, key="card_trend_chart"):
        card_header("Monthly reports", "Shaded band shows the usual range, one standard deviation either side of the mean")
        fig = go.Figure()
        fig.add_hrect(y0=max(mu - sd, 0), y1=mu + sd, fillcolor=GREEN, opacity=0.07, line_width=0)
        fig.add_trace(go.Scatter(x=series["month_dt"], y=series["report_count"], mode="lines+markers", name="Reports",
                                 line=dict(color=GREEN, width=2.4), marker=dict(size=6),
                                 hovertemplate="%{x|%b %Y}: %{y:,} reports<extra></extra>"))
        fig.add_hline(y=mu, line_dash="dash", line_color=DIM, line_width=1, annotation_text=f"mean {mu:.1f}",
                      annotation_font_size=11, annotation_font_color=MUTED)
        fig.add_hline(y=thr, line_dash="dot", line_color=RED, line_width=1.5, annotation_text=f"spike line {thr:.1f}",
                      annotation_font_size=11, annotation_font_color=RED)
        if not sp.empty:
            fig.add_trace(go.Scatter(x=sp["month_dt"], y=sp["report_count"], mode="markers", name="Spike",
                                     marker=dict(size=14, color=RED, symbol="diamond", line=dict(width=1.5, color=SURFACE)),
                                     customdata=sp["z_score"],
                                     hovertemplate="%{x|%b %Y}: %{y:,} reports<br>Z +%{customdata:.2f}<extra></extra>"))
        fig.update_layout(xaxis_title=None, yaxis_title="Case reports", hovermode="closest")
        show_chart(style_fig(fig, 400, legend=True))

    c1, c2 = st.columns(2, gap="medium")
    with c1, st.container(border=True, key="card_trend_sp"):
        card_header("Spikes in this series")
        if sp.empty:
            st.caption("No month in this series rises above the spike line.")
        else:
            st.dataframe(sp[["year_month", "report_count", "mean_count", "std_count", "z_score"]],
                         hide_index=True, width="stretch", height=min(38 + 35 * len(sp), 300),
                         column_config=num_cols(year_month=("Month", None), report_count=("Reports", "%d"),
                                                mean_count=("Mean", "%.1f"), std_count=("SD", "%.1f"),
                                                z_score=("Z", "%.2f")))
    with c2, st.container(border=True, key="card_trend_all"):
        card_header("Largest spikes across all drugs", "Top 15 by Z-score")
        st.dataframe(spikes_all.sort_values("z_score", ascending=False).head(15)
                     [["drug_name", "reaction_name", "year_month", "report_count", "z_score"]],
                     hide_index=True, width="stretch", height=300,
                     column_config=num_cols(drug_name=("Drug", None), reaction_name=("Reaction", None),
                                            year_month=("Month", None), report_count=("Reports", "%d"),
                                            z_score=("Z", "%.2f")))


# ===========================================================================
# INTERACTION NETWORK
# ===========================================================================
@st.cache_data(show_spinner=False)
def network_layout(edges: tuple) -> dict:
    g = nx.Graph()
    g.add_edges_from(edges)
    return {k: tuple(v) for k, v in nx.spring_layout(g, k=0.6, iterations=80, seed=42).items()}


def page_network() -> None:
    if interactions_df.empty:
        st.info("The interactions table is missing. Run `python src/05_interactions.py`, then reload this page.")
        return
    n_hr = int(interactions_df["is_high_risk_interaction"].sum())
    page_header("Drugs reported together", "Interaction network", [
        ("ring", f"{len(interactions_df)} frequent pairs"),
        ("alert", f"{n_hr} high-risk pairs"),
    ], lead="Each dot is a drug and each line joins two drugs that often appear in the same report. Red lines mark "
            "high-risk pairs: more reactions per report than either drug alone, or at least 70% serious outcomes.")

    all_net = sorted(set(interactions_df["drug_1"]) | set(interactions_df["drug_2"]))
    c_ctrl, c_graph = st.columns([1, 3], gap="medium")
    with c_ctrl, st.container(border=True, key="card_net_ctrl"):
        card_header("Filters")
        max_co = int(interactions_df["pair_report_count"].max())
        min_co = st.slider("Minimum shared reports", 1, max(max_co, 2), min(25, max_co), 1)
        high_risk = st.toggle("High-risk pairs only", value=True)
        focal = st.selectbox("Focus on one drug", ["All drugs"] + all_net)

    net = interactions_df[interactions_df["pair_report_count"] >= min_co]
    if high_risk:
        net = net[net["is_high_risk_interaction"]]
    if focal != "All drugs":
        net = net[(net["drug_1"] == focal) | (net["drug_2"] == focal)]

    with c_ctrl:
        gap(.6)
        st.markdown('<div class="stats boxed" style="--n:1">'
                    + stat("Pairs shown", fmt_int(len(net)), "link", sub_html=sub(f"of {len(interactions_df)} pairs"))
                    + "</div>", unsafe_allow_html=True)
        gap(.6)
        st.markdown('<div class="stats boxed" style="--n:1">'
                    + stat("Drugs shown", fmt_int(len(set(net["drug_1"]) | set(net["drug_2"]))), "pill")
                    + "</div>", unsafe_allow_html=True)

    with c_graph, st.container(border=True, key="card_net_graph"):
        if net.empty:
            card_header("Network")
            st.warning("No pairs match these filters. Lower the minimum shared reports or switch off High-risk pairs only.")
        else:
            pos = network_layout(tuple(sorted(zip(net["drug_1"], net["drug_2"]))))
            g = nx.Graph()
            for r in net.itertuples():
                g.add_edge(r.drug_1, r.drug_2, w=r.pair_report_count, risk=bool(r.is_high_risk_interaction))
            card_header("Network", f"{g.number_of_nodes()} drugs and {g.number_of_edges()} pairs. Bigger dots have more "
                                   "partners. Labels show the 25 most connected drugs; hover for the rest.")
            traces = []
            for risk, color in [(False, "#3A3B3A"), (True, "rgba(229,83,75,0.45)")]:
                ex, ey = [], []
                for u, v, d in g.edges(data=True):
                    if d["risk"] == risk:
                        ex += [pos[u][0], pos[v][0], None]
                        ey += [pos[u][1], pos[v][1], None]
                if ex:
                    traces.append(go.Scatter(x=ex, y=ey, mode="lines", hoverinfo="skip", line=dict(width=1.3, color=color)))
            mx, my, mt = [], [], []
            for u, v, d in g.edges(data=True):
                mx.append((pos[u][0] + pos[v][0]) / 2)
                my.append((pos[u][1] + pos[v][1]) / 2)
                mt.append(f"<b>{esc(u)} + {esc(v)}</b><br>{d['w']:,} shared reports")
            traces.append(go.Scatter(x=mx, y=my, mode="markers", marker=dict(size=8, opacity=0), hovertext=mt, hoverinfo="text"))
            nodes = list(g.nodes())
            deg = np.array([g.degree(n) for n in nodes])
            cut = np.sort(deg)[::-1][min(24, len(deg) - 1)] if len(deg) > 25 else 0
            traces.append(go.Scatter(
                x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers+text",
                text=[n if d >= cut else "" for n, d in zip(nodes, deg)], textposition="top center",
                textfont=dict(size=10.5, color="#D6D6D6"),
                hovertext=[f"<b>{esc(n)}</b><br>{d} partner drugs" for n, d in zip(nodes, deg)], hoverinfo="text",
                marker=dict(size=np.clip(9 + deg * 2.4, 9, 36), color=deg, colorscale=[[0, "#2F5A1C"], [1, GREEN2]],
                            line=dict(width=1.5, color=SURFACE)),
            ))
            fig = go.Figure(traces)
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            fig = style_fig(fig, 560)
            fig.update_layout(hovermode="closest", margin=dict(l=4, r=4, t=4, b=4))
            show_chart(fig)

    if not net.empty:
        with st.container(border=True, key="card_net_table"):
            card_header("Pairs in view", "Most shared reports first")
            st.dataframe(net.sort_values("pair_report_count", ascending=False).head(50)
                         [["drug_1", "drug_2", "pair_report_count", "avg_reaction_count", "reaction_rate_ratio",
                           "serious_rate", "top_reaction", "is_high_risk_interaction"]],
                         hide_index=True, width="stretch", height=380,
                         column_config={
                             **num_cols(drug_1=("Drug 1", None), drug_2=("Drug 2", None),
                                        pair_report_count=("Shared reports", "%d"),
                                        avg_reaction_count=("Reactions per report", "%.1f"),
                                        reaction_rate_ratio=("Rate ratio", "%.2f"), top_reaction=("Top reaction", None)),
                             "serious_rate": st.column_config.ProgressColumn("Serious rate", min_value=0.0, max_value=1.0,
                                                                             format="percent"),
                             "is_high_risk_interaction": st.column_config.CheckboxColumn("High risk"),
                         })


# ===========================================================================
# ARCHITECTURE
# ===========================================================================
@st.cache_data(show_spinner=False)
def architecture_stats() -> dict:
    raw = sorted([f for f in RAW_DIR.rglob("*.json") if f.is_file()], key=lambda f: f.name)
    sig = signals_df
    strong = sig[sig["is_signal"] & (sig["report_count"] >= 20)].sort_values("ror_lower_ci", ascending=False)
    ex = strong.iloc[0] if not strong.empty else sig.sort_values("chi_square", ascending=False).iloc[0]
    out = {
        "raw_files": [{"name": f.name, "bytes": f.stat().st_size} for f in raw] or [{"name": "no raw files found", "bytes": 1}],
        "parquet_bytes": sum(dir_bytes(p) for p in TABLES.values()),
        "counts": {
            "reports": int(len(reports_df)), "clean_rows": int(len(clean_df)), "pairs": int(len(sig)),
            "signals": int(sig["is_signal"].sum()), "interactions": int(len(interactions_df)),
            "high_risk": int(interactions_df["is_high_risk_interaction"].sum()) if not interactions_df.empty else 0,
            "trend_rows": int(len(trends_df)),
            "spikes": int(trends_df["is_spike"].sum()) if not trends_df.empty else 0, "mr": int(len(mr_df)),
        },
        "example": {"drug": str(ex["drug_name"]), "reaction": str(ex["reaction_name"]), "a": int(ex["report_count"]),
                    "b": int(ex["b"]), "c": int(ex["c"]), "d": int(ex["d"]), "prr": float(ex["prr"]),
                    "ror": float(ex["ror"]) if pd.notna(ex["ror"]) else 0.0, "chi": float(ex["chi_square"])},
        "top_pair": {"d1": "n/a", "d2": "n/a", "co": 0, "support": 0.0, "ratio": 0.0},
        "mr_top": [["n/a", 0]] * 3,
        "spike": {"drug": "n/a", "months": [], "counts": [0], "mu": 0.0, "thr": 0.0, "z": 0.0, "spike_month": ""},
    }
    if not interactions_df.empty:
        r = interactions_df.sort_values("pair_report_count", ascending=False).iloc[0]
        out["top_pair"] = {"d1": str(r["drug_1"]), "d2": str(r["drug_2"]), "co": int(r["pair_report_count"]),
                           "support": float(r["support"]), "ratio": float(r["reaction_rate_ratio"])}
    if not mr_df.empty:
        top = mr_df.sort_values("event_count", ascending=False).head(3)
        out["mr_top"] = [[str(a), int(b)] for a, b in zip(top["item_name"], top["event_count"])]
    if not trends_df.empty and trends_df["is_spike"].any():
        top = trends_df[trends_df["is_spike"]].sort_values("z_score", ascending=False).iloc[0]
        s = trends_df[(trends_df["drug_name"] == top["drug_name"]) & (trends_df["reaction_name"] == top["reaction_name"])]
        s = s.sort_values("month_dt")
        out["spike"] = {"drug": str(top["drug_name"]), "months": s["year_month"].tolist(),
                        "counts": [int(c) for c in s["report_count"]], "mu": float(top["mean_count"]),
                        "thr": float(top["mean_count"] + 3 * top["std_count"]), "z": float(top["z_score"]),
                        "spike_month": str(top["year_month"])}
    return out


def page_architecture() -> None:
    stats = architecture_stats()
    raw_gb = sum(f["bytes"] for f in stats["raw_files"]) / 1073741824
    page_header("System architecture", "From dataset import to analysis", [
        ("database", f"{raw_gb:.2f} GB raw JSON"),
        ("layers", "Hadoop 3.5 and Apache Spark"),
        ("chart", "10 pipeline stages"),
    ])

    tpl = (ASSETS_DIR / "architecture_flow.html").read_text(encoding="utf-8")
    payload = json.dumps(stats).replace("</", "<\\/")  # keep data from closing the <script> block
    st.iframe(tpl.replace("/*__STATS__*/{}", payload), height=1010)

    gap(1.4)
    c = stats["counts"]
    out_mb = stats["parquet_bytes"] / 1048576
    stat_strip([
        stat("Raw input", f"{raw_gb:.2f} GB", "database", pill(f"{len(stats['raw_files'])} files", "grey")),
        stat("Clean rows", fmt_int(c["clean_rows"]), "layers", sub_html=sub(f"from {fmt_int(c['reports'])} reports")),
        stat("Pairs scored", fmt_int(c["pairs"]), "activity", sub_html=sub(f"{fmt_int(c['signals'])} signals")),
        stat("Parquet output", f"{out_mb:.1f} MB", "file",
             pill(f"{raw_gb * 1024 / max(out_mb, 0.01):.0f}x smaller", "green")),
    ])
    gap(1.4)

    c1, c2 = st.columns([1, 1], gap="medium")
    with c1, st.container(border=True, key="card_stack"):
        card_header("Technology stack", "Each layer and the job it does in SafeDose")
        rows = [
            ("Dataset", "<b>openFDA FAERS</b> drug adverse event bulk files, multi-line JSON"),
            ("Storage", "<b>Hadoop FileSystem API</b> in local mode (same API as HDFS); winutils.exe and hadoop.dll on Windows"),
            ("Compute", "<b>Apache Spark</b> via PySpark, local[4]: four parallel tasks on one machine"),
            ("Processing", "<b>Spark SQL</b> DataFrames for cleaning, joins and window functions"),
            ("Machine learning", "<b>Spark MLlib FP-Growth</b> for frequent drug pairs"),
            ("Classic model", "<b>MapReduce</b> pattern with map and reduceByKey on RDDs"),
            ("File format", "<b>Apache Parquet</b> with Snappy compression"),
            ("Dashboard", "<b>Streamlit</b> pages, <b>Plotly</b> charts, <b>NetworkX</b> graph layout"),
        ]
        st.markdown('<div class="stack">' + "".join(f'<div class="k">{k}</div><div>{v}</div>' for k, v in rows) + "</div>",
                    unsafe_allow_html=True)
    with c2, st.container(border=True, key="card_tables"):
        card_header("Tables produced", "Written to data/processed and read by the dashboard")
        tbl = pd.DataFrame([
            ("clean_reports", "03_clean.py", len(clean_df), dir_bytes(TABLES["clean"]), "every page"),
            ("signals", "04_signals.py", len(signals_df), dir_bytes(TABLES["signals"]), "Safety Signals, Drug Explorer"),
            ("interactions", "05_interactions.py", len(interactions_df), dir_bytes(TABLES["interactions"]), "Interaction Network"),
            ("trends", "06_trends.py", len(trends_df), dir_bytes(TABLES["trends"]), "Trends & Spikes"),
            ("hadoop_mr_counts", "07_hadoop_showcase.py", len(mr_df), dir_bytes(TABLES["mapreduce"]), "Architecture"),
        ], columns=["table", "script", "rows", "bytes", "used_by"])
        tbl["size"] = (tbl["bytes"] / 1024).round(0)
        st.dataframe(tbl[["table", "script", "rows", "size", "used_by"]], hide_index=True, width="stretch",
                     column_config=num_cols(table=("Table", None), script=("Written by", None), rows=("Rows", "%d"),
                                            size=("Size (KB)", "%d"), used_by=("Used on", None)))
        st.markdown('<div class="card-t" style="font-size:1.05rem;margin-top:.6rem">Run the pipeline</div>',
                    unsafe_allow_html=True)
        st.code("python src/02_download.py\npython src/03_clean.py\npython src/04_signals.py\n"
                "python src/05_interactions.py\npython src/06_trends.py\npython src/07_hadoop_showcase.py\n"
                "streamlit run src/app.py", language="bash")

    if not mr_df.empty:
        with st.container(border=True, key="card_mr"):
            card_header("MapReduce output", "Most frequent drugs counted with map and reduceByKey")
            top = mr_df.sort_values("event_count", ascending=False).head(15)
            show_chart(hbar(top, "event_count", "item_name", GREEN, 420, x_title="Drug and reaction rows"))


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

st.markdown('<div class="foot">Data from openFDA, the public copy of the FDA Adverse Event Reporting System. '
            'For research and teaching, not for clinical decisions.</div>', unsafe_allow_html=True)
