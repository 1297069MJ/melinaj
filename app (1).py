from __future__ import annotations

import ast
import io
import math
import re
from datetime import date
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import sympy as sp

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
)

st.set_page_config(page_title="Hispanic Heritage Historical Data Explorer",
                   page_icon="📊", layout="wide")

# Local, built-in theme choices: no image files or external assets are required.
FONT_OPTIONS = {
    "Nunito (soft + rounded)": ("Nunito", "Nunito:wght@400;600;700;800"),
    "Poppins (cute + modern)": ("Poppins", "Poppins:wght@400;500;600;700;800"),
    "Quicksand (playful)": ("Quicksand", "Quicksand:wght@400;500;600;700"),
    "Playfair Display (elegant)": ("Playfair Display", "Playfair+Display:wght@400;500;600;700"),
    "Lora (classic)": ("Lora", "Lora:wght@400;500;600;700"),
}
BACKGROUND_OPTIONS = {
    "Cotton candy": ("#fff0f6", "#f3e8ff", "#fff7ed", "#7c3aed"),
    "Sage garden": ("#edf7ef", "#e4f1e8", "#f7f8ed", "#28704a"),
    "Peach lemonade": ("#fff1e6", "#ffe8d6", "#fff8e7", "#b45309"),
    "Lavender sky": ("#f1edff", "#e9e7ff", "#f7f3ff", "#6d5bd0"),
    "Ocean breeze": ("#e8f6fb", "#e6f0ff", "#f3fbff", "#176b87"),
    "Clean paper": ("#f8fafc", "#f1f5f9", "#ffffff", "#475569"),
}

# Spanish-speaking sovereign countries (20), Puerto Rico as a territory,
# and Brazil as a non-Hispanic Latin American comparison.
COUNTRIES = {
    "Argentina": "ARG", "Bolivia": "BOL", "Chile": "CHL",
    "Colombia": "COL", "Costa Rica": "CRI", "Cuba": "CUB",
    "Dominican Republic": "DOM", "Ecuador": "ECU", "El Salvador": "SLV",
    "Equatorial Guinea": "GNQ", "Guatemala": "GTM", "Honduras": "HND",
    "Mexico": "MEX", "Nicaragua": "NIC", "Panama": "PAN",
    "Paraguay": "PRY", "Peru": "PER", "Spain": "ESP",
    "Uruguay": "URY", "Venezuela": "VEN",
    "Puerto Rico (territory)": "PRI",
    "Brazil (Latin American comparison; not Hispanic)": "BRA",
}

INDICATORS = {
    "Life expectancy at birth": ("SP.DYN.LE00.IN", "years", "Average number of years a newborn is expected to live if current age-specific mortality rates persist.", "nonnegative"),
    "Unemployment, national estimate": ("SL.UEM.TOTL.NE.ZS", "% of labor force", "Share of the labor force that is unemployed, using national estimates.", "percent"),
    "GDP per capita": ("NY.GDP.PCAP.CD", "current US$ per person", "Gross domestic product divided by midyear population. This is not average wealth or household income.", "nonnegative"),
    "Crude birth rate": ("SP.DYN.CBRT.IN", "births per 1,000 people per year", "Annual live births per 1,000 people; this is a rate, not a birth count.", "nonnegative"),
    "Net migration": ("SM.POP.NETM", "people", "Net number of migrants (immigrants minus emigrants) over the period; it is not an emigration count.", "any"),
    "Intentional homicide rate": ("VC.IHR.PSRC.P5", "per 100,000 people", "Intentional homicides per 100,000 people.", "nonnegative"),
    "Consumer price inflation": ("FP.CPI.TOTL.ZG", "% annual", "Annual percentage change in the consumer price index.", "any"),
    "Central government debt": ("GC.DOD.TOTL.GD.ZS", "% of GDP", "Central government debt as a percentage of GDP.", "nonnegative"),
    "Total reserves including gold": ("FI.RES.TOTL.CD", "current US$", "Total reserves including gold, in current US dollars.", "nonnegative"),
    "External balance on goods and services": ("NE.RSB.GNFS.ZS", "% of GDP", "External balance on goods and services as a percentage of GDP.", "any"),
}

CONTEXTS = {
    "Business": ("revenue", "dollars", "time", "years", "a business revenue model"),
    "Engineering": ("displacement", "meters", "time", "seconds", "an engineering motion model"),
    "Medical": ("concentration", "mg/L", "time", "hours", "a simplified concentration model (not medical advice)"),
    "Sports science": ("distance", "meters", "time", "seconds", "a sports performance model"),
    "Biology": ("population", "organisms", "time", "days", "a population model"),
    "Bioengineering": ("cell_count", "cells", "time", "hours", "a bioengineering model"),
    "Economic forecasting": ("index", "index points", "time", "years", "an economic indicator model"),
    "Weather": ("temperature", "°C", "time", "days", "a weather trend model"),
    "Custom": ("y", "units", "x", "units", "a custom model"),
}
SUGGESTED_FUNCTIONS = {
    "Business": "a*x**2 + b*x + c",
    "Engineering": "a*sin(b*x) + c",
    "Medical": "a*exp(-b*x) + c",
    "Sports science": "a*x + b",
    "Biology": "a*exp(b*x)",
    "Bioengineering": "a*x**2 + b*x + c",
    "Economic forecasting": "a*x**2 + b*x + c",
    "Weather": "a*sin(b*x) + c",
    "Custom": "a*x + b",
}

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_world_bank(country_code: str, indicator_code: str, start_year: int, end_year: int) -> pd.DataFrame:
    """Retrieve World Bank annual observations. Empty results remain empty; no synthetic data."""
    url = (
        f"https://api.worldbank.org/v2/country/{country_code}/indicator/"
        f"{indicator_code}?date={start_year}:{end_year}&format=json&per_page=20000"
    )
    response = requests.get(url, timeout=25)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
        return pd.DataFrame(columns=["year", "value"])
    rows = []
    for item in payload[1]:
        value = item.get("value")
        year = item.get("date")
        if value is None or year is None:
            continue
        try:
            rows.append({"year": int(year), "value": float(value)})
        except (TypeError, ValueError):
            continue
    df = pd.DataFrame(rows, columns=["year", "value"])
    if not df.empty:
        df = df.sort_values("year").reset_index(drop=True)
    return df

def validate_data(df: pd.DataFrame, constraint: str) -> tuple[pd.DataFrame, list[str]]:
    errors = []
    if not {"year", "value"}.issubset(df.columns):
        return pd.DataFrame(columns=["year", "value"]), ["Table must contain year and value columns."]
    work = df[["year", "value"]].copy()
    work["year"] = pd.to_numeric(work["year"], errors="coerce")
    work["value"] = pd.to_numeric(work["value"], errors="coerce")
    if work.isna().any().any():
        errors.append("Years and values must be numeric and cannot be blank.")
        work = work.dropna()
    if not work.empty:
        if ((work["year"] % 1) != 0).any():
            errors.append("Years must be whole numbers.")
        work = work[(work["year"] % 1) == 0]
        if work["year"].duplicated().any():
            errors.append("Duplicate years are not allowed. Keep one row per year.")
        if not work["year"].between(1900, 2100).all():
            errors.append("Years must be between 1900 and 2100.")
        if not np.isfinite(work["value"].astype(float)).all():
            errors.append("Values must be finite numbers.")
        if constraint == "nonnegative" and (work["value"] < 0).any():
            errors.append("This indicator cannot have negative values.")
        if constraint == "percent" and ((work["value"] < 0) | (work["value"] > 100)).any():
            errors.append("Unemployment percentage must be between 0 and 100.")
        work = work.sort_values("year").reset_index(drop=True)
    return work, errors

def fit_polynomial(df: pd.DataFrame, degree: int):
    clean = df.dropna().sort_values("year")
    if clean["year"].nunique() < degree + 1:
        raise ValueError(f"Degree {degree} requires at least {degree + 1} distinct valid years.")
    base_year = int(clean["year"].min())
    t = clean["year"].to_numpy(dtype=float) - base_year
    y = clean["value"].to_numpy(dtype=float)
    coeff = np.polyfit(t, y, degree)
    model = np.poly1d(coeff)
    predicted = model(t)
    residuals = y - predicted
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    mae = float(np.mean(np.abs(residuals)))
    return base_year, model, coeff, r2, mae, residuals, clean

def safe_sympy_expression(source: str, x: sp.Symbol):
    """Parse a limited math grammar; never evaluate arbitrary Python code."""
    source = source.strip().replace("^", "**")
    if not source or len(source) > 180:
        raise ValueError("Enter a function no longer than 180 characters.")
    if not re.fullmatch(r"[A-Za-z0-9_+\-*/().,\s**]+", source):
        raise ValueError("Use only numbers, x, a, b, c, pi, e, arithmetic, and supported functions.")
    tree = ast.parse(source, mode="eval")
    allowed_names = {"x": x, "a": sp.Symbol("a"), "b": sp.Symbol("b"),
                     "c": sp.Symbol("c"), "pi": sp.pi, "e": sp.E}
    allowed_funcs = {"sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
                     "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt,
                     "abs": sp.Abs}
    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return sp.Float(node.value)
        if isinstance(node, ast.Name):
            if node.id not in allowed_names:
                raise ValueError(f"Unknown name: {node.id}. Use x, a, b, c, pi, or e.")
            return allowed_names[node.id]
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            ops = {ast.Add: lambda: left + right, ast.Sub: lambda: left - right,
                   ast.Mult: lambda: left * right, ast.Div: lambda: left / right,
                   ast.Pow: lambda: left ** right}
            if type(node.op) not in ops:
                raise ValueError("Unsupported operator.")
            return ops[type(node.op)]()
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name not in allowed_funcs or len(node.args) != 1 or node.keywords:
                raise ValueError("Supported one-input functions: sin, cos, tan, exp, log, sqrt, abs.")
            return allowed_funcs[name](visit(node.args[0]))
        raise ValueError("Unsupported expression. Use standard arithmetic and supported functions.")
    return sp.simplify(visit(tree))

def make_chart(df, base_year, model, sample_step, single_year=None, batch=None):
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.scatter(df["year"], df["value"], label="Observed data", s=24)
    lo, hi = int(df["year"].min()), int(df["year"].max())
    grid_hist = np.linspace(lo, hi, 500)
    ax.plot(grid_hist, model(grid_hist - base_year), label="Polynomial fit")
    if single_year is not None:
        ax.scatter([single_year], [model(single_year-base_year)], marker="*", s=150, label="Single-year estimate")
    if batch is not None and not batch.empty:
        ax.scatter(batch["year"], batch["estimate"], marker="x", s=55, label="Batch estimates")
    sample_years = np.arange(lo, hi + 1, sample_step)
    ax.scatter(sample_years, model(sample_years-base_year), marker=".", alpha=.55, label=f"Model markers every {sample_step} year(s)")
    ax.axvline(hi, linestyle="--", linewidth=1, label="Historical boundary")
    ax.set_xlabel("Year")
    ax.set_ylabel(st.session_state.get("units", "Indicator value"))
    ax.set_title("Historical observations and polynomial model")
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    return fig

def pdf_report(metadata: dict, equation: str, metrics: str, chart_bytes: bytes,
               preview: pd.DataFrame, results: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    doc = SimpleDocTemplate(output, pagesize=letter, rightMargin=36, leftMargin=36)
    styles = getSampleStyleSheet()
    story = [Paragraph("Hispanic Heritage Historical Data Explorer", styles["Title"]),
             Spacer(1, 10)]
    for key, value in metadata.items():
        story.append(Paragraph(f"<b>{key}:</b> {value}", styles["BodyText"]))
    story += [Spacer(1, 8), Paragraph("<b>Model equation</b>", styles["Heading2"]),
              Paragraph(equation.replace("<", "&lt;"), styles["BodyText"]),
              Paragraph(metrics, styles["BodyText"]), Spacer(1, 10)]
    chart_file = io.BytesIO(chart_bytes)
    story.append(RLImage(chart_file, width=500, height=260))
    story.append(Paragraph("Data preview", styles["Heading2"]))
    data = [["Year", "Value"]] + [[str(int(r.year)), f"{r.value:.6g}"] for r in preview.itertuples()]
    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
                               ("GRID", (0,0), (-1,-1), .4, colors.grey),
                               ("FONTSIZE", (0,0), (-1,-1), 8)]))
    story.append(table)
    if not results.empty:
        story.append(Paragraph("Calculated-year results", styles["Heading2"]))
        res_data = [["Year", "Estimate", "Classification"]] + [
            [str(int(r.year)), f"{r.estimate:.6g}", r.classification]
            for r in results.itertuples()
        ]
        t2 = Table(res_data, repeatRows=1)
        t2.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
                                ("GRID", (0,0), (-1,-1), .4, colors.grey),
                                ("FONTSIZE", (0,0), (-1,-1), 8)]))
        story.append(t2)
    doc.build(story)
    return output.getvalue()

st.title("🌎 Hispanic Heritage Historical Data Explorer")
st.caption("Explore World Bank historical indicators, polynomial models, and function behavior.")
if st.session_state.get("printer_mode", False):
    st.markdown(
        "<style>@media print { section[data-testid='stSidebar'] {display:none;} "
        "button, [data-testid='stDownloadButton'] {display:none;} "
        ".block-container {padding-top:1rem;} }</style>",
        unsafe_allow_html=True,
    )
    st.info("Printer-friendly view is on: use your browser's Print command. Interactive controls and download buttons are hidden in the printed page.")
st.info("Country coverage: 20 Spanish-speaking sovereign countries, Puerto Rico as a separate territory, and Brazil as a non-Hispanic Latin American comparison.")
with st.expander("See the full country list"):
    st.write("**20 Spanish-speaking sovereign countries:** Argentina, Bolivia, Chile, Colombia, Costa Rica, Cuba, Dominican Republic, Ecuador, El Salvador, Equatorial Guinea, Guatemala, Honduras, Mexico, Nicaragua, Panama, Paraguay, Peru, Spain, Uruguay, and Venezuela.")
    st.write("**Additional options:** Puerto Rico (territory) and Brazil (Latin American comparison; not Hispanic).")

with st.sidebar:
    st.header("Data selection")
    country_name = st.selectbox("Country", list(COUNTRIES.keys()), index=list(COUNTRIES).index("Mexico"))
    indicator_name = st.selectbox("Indicator", list(INDICATORS.keys()))
    degree = st.slider("Polynomial degree", 3, 8, 3)
    sample_step = st.slider("Graph sample increment (years)", 1, 10, 2)
    forward_horizon = st.slider("Forward display horizon (years)", 0, 50, 10)
    comparison_names = st.multiselect("Comparison countries", [n for n in COUNTRIES if n != country_name], default=[])
    printer_mode = st.checkbox("Printer-friendly view", value=False)
    st.session_state["printer_mode"] = printer_mode
    st.divider()
    st.header("✨ Personalize your website")
    font_choice = st.selectbox("Cute font", list(FONT_OPTIONS.keys()), index=0)
    background_choice = st.selectbox("Background style", list(BACKGROUND_OPTIONS.keys()), index=0)

font_name, google_font = FONT_OPTIONS[font_choice]
bg1, bg2, bg3, accent = BACKGROUND_OPTIONS[background_choice]
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family={google_font}&display=swap');
    html, body, [class*="css"] {{
        font-family: '{font_name}', 'Trebuchet MS', sans-serif;
    }}
    .stApp {{
        background: linear-gradient(135deg, {bg1} 0%, {bg2} 52%, {bg3} 100%);
        background-attachment: fixed;
    }}
    [data-testid="stHeader"] {{ background: transparent; }}
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {bg2} 0%, {bg3} 100%);
        border-right: 1px solid rgba(120, 100, 150, .14);
    }}
    [data-testid="stMetric"] {{
        background: rgba(255,255,255,.66);
        border: 1px solid rgba(120,100,150,.13);
        border-radius: 16px;
        padding: 12px 14px;
    }}
    div.stButton > button, div.stDownloadButton > button {{
        border-radius: 14px;
        border: 1px solid {accent};
        font-weight: 700;
    }}
    h1, h2, h3 {{ letter-spacing: -0.025em; }}
    .stAlert {{ border-radius: 14px; }}
    @media print {{
        .stApp {{ background: white !important; }}
        [data-testid="stSidebar"] {{ display: none; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)
st.caption(f"Theme: {font_choice} · {background_choice}")

indicator_code, units, definition, constraint = INDICATORS[indicator_name]
st.session_state["units"] = units
current_year = date.today().year
start_year, end_year = current_year - 70, current_year - 1
st.write(f"**Research window:** {start_year}–{end_year} inclusive (requested 70-year window).")
st.write(f"**Definition:** {definition}  \n**Units:** {units}  \n**Source:** [World Bank indicator page](https://data.worldbank.org/indicator/{indicator_code})")

try:
    source_df = fetch_world_bank(COUNTRIES[country_name], indicator_code, start_year, end_year)
    retrieval_error = None
except requests.RequestException as exc:
    source_df = pd.DataFrame(columns=["year", "value"])
    retrieval_error = f"World Bank request failed: {exc}"
except (ValueError, TypeError) as exc:
    source_df = pd.DataFrame(columns=["year", "value"])
    retrieval_error = f"Could not read World Bank response: {exc}"

if retrieval_error:
    st.error(retrieval_error)
elif source_df.empty:
    st.warning("The World Bank API returned no usable observations for this selection. No replacement or random data was created.")
else:
    st.success(f"Retrieved {len(source_df)} observations, covering {source_df.year.min()}–{source_df.year.max()}.")

st.caption(f"Retrieved on: {date.today().isoformat()} | API: https://api.worldbank.org/v2/country/{COUNTRIES[country_name]}/indicator/{indicator_code}")

if "editor_revision" not in st.session_state:
    st.session_state["editor_revision"] = 0

if "original_df" not in st.session_state or st.session_state.get("source_key") != (country_name, indicator_name):
    st.session_state["editor_revision"] += 1
    st.session_state["original_df"] = source_df.copy()
    st.session_state["source_key"] = (country_name, indicator_name)
    st.session_state["working_df"] = source_df.copy()
    st.session_state["edited"] = False
    st.session_state["calc_results"] = pd.DataFrame(columns=["year", "estimate", "classification"])
    st.session_state["single_result"] = None

left, right = st.columns([1.15, 1])
with left:
    st.subheader("Validated, editable data")
    st.caption("Edit the year/value table to test a scenario. Reset restores the original API observations.")
    edited_table = st.data_editor(
        st.session_state["working_df"], num_rows="dynamic",
        use_container_width=True,
        key=f"editable_data_{st.session_state['editor_revision']}"
    )
    validated_df, validation_errors = validate_data(edited_table, constraint)
    for error in validation_errors:
        st.error(error)
    if not validation_errors:
        st.session_state["working_df"] = validated_df
        st.session_state["edited"] = not validated_df.equals(st.session_state["original_df"].reset_index(drop=True))
    if st.button("Reset to source"):
        st.session_state["working_df"] = st.session_state["original_df"].copy()
        st.session_state["edited"] = False
        st.session_state["calc_results"] = pd.DataFrame(columns=["year", "estimate", "classification"])
        st.session_state["single_result"] = None
        st.session_state["editor_revision"] = st.session_state.get("editor_revision", 0) + 1
        st.rerun()
    if st.session_state["edited"]:
        st.warning("User-modified scenario: calculations and exports use edited data.")
    else:
        st.caption("Using original source observations.")

with right:
    st.subheader("Coverage summary")
    data = st.session_state["working_df"]
    if not data.empty:
        st.metric("Valid observations", len(data))
        st.metric("Actual coverage", f"{int(data.year.min())}–{int(data.year.max())}")
        st.metric("Mean", f"{data.value.mean():,.4g} {units}")
        st.dataframe(data.describe().rename_axis("Statistic"), use_container_width=True)
    else:
        st.info("No valid observations are available yet.")

data = st.session_state["working_df"]
model_state_key = (country_name, indicator_name, degree, tuple(map(tuple, data[["year", "value"]].to_numpy().tolist())) if not data.empty else ())
if st.session_state.get("last_model_state_key") != model_state_key:
    st.session_state["single_result"] = None
    st.session_state["calc_results"] = pd.DataFrame(columns=["year", "estimate", "classification"])
    st.session_state["last_model_state_key"] = model_state_key

if not data.empty and not validation_errors:
    try:
        base_year, model, coeff, r2, mae, residuals, clean = fit_polynomial(data, degree)
        coeff_text = " + ".join(
            [f"({c:.10g})·(year − {base_year})^{degree-i}" for i, c in enumerate(coeff)]
        )
        st.subheader("Polynomial model")
        st.latex("t = \\text{year} - " + str(base_year))
        st.code(f"f(t) = {np.poly1d(coeff)}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Degree", degree)
        m2.metric("R² (in-sample)", f"{r2:.5f}" if np.isfinite(r2) else "Undefined")
        m3.metric("MAE (in-sample)", f"{mae:.5g} {units}")
        m4.metric("Observation count", len(clean))
        st.caption(f"Base year: {base_year} | Historical range: {int(clean.year.min())}–{int(clean.year.max())} | Mean: {clean.value.mean():.6g} {units}")
        with st.expander("Residuals and chronological holdout"):
            residual_table = clean.copy()
            residual_table["fitted"] = model(clean.year.to_numpy()-base_year)
            residual_table["residual"] = residuals
            st.dataframe(residual_table, use_container_width=True)
            if len(clean) >= max(8, degree + 3):
                split = max(degree + 1, int(len(clean) * .8))
                train = clean.iloc[:split]
                test = clean.iloc[split:]
                if len(test):
                    hb = int(train.year.min())
                    hm = np.poly1d(np.polyfit(train.year.to_numpy()-hb, train.value.to_numpy(), degree))
                    holdout_mae = float(np.mean(np.abs(test.value.to_numpy()-hm(test.year.to_numpy()-hb))))
                    st.write(f"Chronological holdout MAE: **{holdout_mae:.6g} {units}** (separate from in-sample MAE; test years {int(test.year.min())}–{int(test.year.max())}).")
            else:
                st.write("Not enough observations for a useful chronological holdout at this degree.")
        st.subheader("Calculate a single year")
        chosen_year = st.number_input("Year (1950–2100)", min_value=1950, max_value=2100, value=min(max(current_year, 1950), 2100), step=1)
        if st.button("Calculate Value"):
            estimate = float(model(chosen_year-base_year))
            classification = "within fitted historical range" if clean.year.min() <= chosen_year <= clean.year.max() else "outside fitted historical range (extrapolation)"
            st.session_state["single_result"] = {"year": int(chosen_year), "estimate": estimate, "classification": classification}
        if st.session_state.get("single_result"):
            result = st.session_state["single_result"]
            st.write(f"**{result['year']}:** {result['estimate']:,.6g} {units} — {result['classification']}.")
            if result["classification"].startswith("outside"):
                st.warning("Polynomial extrapolation can become implausible quickly. This is a mathematical estimate, not a validated forecast.")
            else:
                st.caption("This is a fitted model value, not the actual observed value.")
        st.subheader("Batch calculations")
        batch_mode = st.radio("Input method", ["Year range", "Custom years"], horizontal=True)
        batch_df = pd.DataFrame(columns=["year", "estimate", "classification"])
        if batch_mode == "Year range":
            c1, c2, c3 = st.columns(3)
            range_start = c1.number_input("Start year", 1950, 2100, int(clean.year.min()), key="range_start")
            range_end = c2.number_input("End year", 1950, 2100, min(int(clean.year.max())+5, 2100), key="range_end")
            range_step = c3.slider("Step", 1, 10, 1, key="range_step")
            requested = list(range(int(range_start), int(range_end)+1, int(range_step))) if range_end >= range_start else []
        else:
            custom = st.text_input("Comma-separated years", f"{int(clean.year.min())}, {int(clean.year.max())}")
            requested, invalid_tokens = [], []
            for token in custom.split(","):
                token = token.strip()
                if not token:
                    continue
                if token.isdigit() and 1950 <= int(token) <= 2100:
                    requested.append(int(token))
                else:
                    invalid_tokens.append(token)
            if invalid_tokens:
                st.warning("Ignored invalid year entries: " + ", ".join(invalid_tokens))
        requested = list(dict.fromkeys(requested))
        if requested:
            batch_df = pd.DataFrame({"year": requested})
            batch_df["estimate"] = [float(model(y-base_year)) for y in requested]
            batch_df["classification"] = batch_df.year.apply(
                lambda y: "within fitted historical range" if clean.year.min() <= y <= clean.year.max()
                else "outside fitted historical range (extrapolation)"
            )
            st.session_state["calc_results"] = batch_df.copy()
            st.dataframe(batch_df, use_container_width=True)
        else:
            st.info("Enter at least one valid year. Batch results use the same fitted model as the single-year calculation.")
        st.subheader("Historical graph")
        fig = make_chart(clean, base_year, model, sample_step,
                         st.session_state.get("single_result", {}).get("year") if st.session_state.get("single_result") else None,
                         batch_df)
        if forward_horizon > 0:
            future_x = np.linspace(int(clean.year.max()), min(2100, int(clean.year.max())+forward_horizon), 100)
            fig.axes[0].plot(future_x, model(future_x-base_year), linestyle=":", label="Extrapolated display segment")
            fig.axes[0].legend()
        st.pyplot(fig, use_container_width=True)
        chart_buffer = io.BytesIO()
        fig.savefig(chart_buffer, format="png", dpi=160, bbox_inches="tight")
        chart_bytes = chart_buffer.getvalue()
        plt.close(fig)

        st.subheader("Country comparison")
        comparison_frames = []
        for comp_name in comparison_names:
            try:
                comp = fetch_world_bank(COUNTRIES[comp_name], indicator_code, start_year, end_year)
                if not comp.empty:
                    comp["country"] = comp_name
                    comparison_frames.append(comp)
            except requests.RequestException:
                st.warning(f"Could not retrieve comparison data for {comp_name}.")
        if comparison_frames:
            compare_df = pd.concat(comparison_frames, ignore_index=True)
            st.line_chart(compare_df.pivot(index="year", columns="country", values="value"))
            st.caption("Countries may have different observation coverage and missing years; compare only overlapping years with care.")
        elif comparison_names:
            st.info("No comparison observations were returned for the selected countries.")

        # Additional user-selectable mathematical context and function lab.
        st.subheader("Function and calculus laboratory")
        course = st.selectbox("Mathematics pathway", ["Honors Algebra 2", "Precalculus", "Calculus"])
        context = st.selectbox("Context", list(CONTEXTS.keys()))
        default_y, default_yu, default_x, default_xu, context_desc = CONTEXTS[context]
        c1, c2 = st.columns(2)
        y_name = c1.text_input("Dependent variable", default_y)
        y_units = c2.text_input("Dependent-variable units", default_yu)
        x_name = c1.text_input("Independent variable", default_x)
        x_units = c2.text_input("Independent-variable units", default_xu)
        description = st.text_input("Context description", context_desc)
        function_choice = st.radio("Function source", ["Use fitted data curve", "Suggested function", "Enter my own function"], horizontal=True)
        x_symbol = sp.Symbol("x", real=True)
        expression = None
        if function_choice == "Use fitted data curve":
            # Show the polynomial in centered-time form as a symbolic expression.
            expression = sum(sp.Float(float(c)) * (x_symbol-base_year)**(degree-i)
                             for i, c in enumerate(coeff))
            st.caption("The fitted World Bank polynomial is used, with x representing calendar year.")
        elif function_choice == "Suggested function":
            suggested = SUGGESTED_FUNCTIONS[context]
            function_text = st.text_input("Suggested function (edit parameters)", suggested)
            try:
                expression = safe_sympy_expression(function_text, x_symbol)
            except Exception as exc:
                st.error(str(exc))
        else:
            function_text = st.text_input("Function f(x)", "a*x**2 + b*x + c")
            try:
                expression = safe_sympy_expression(function_text, x_symbol)
            except Exception as exc:
                st.error(str(exc))
        if expression is not None:
            parameter_cols = st.columns(3)
            a_val = parameter_cols[0].number_input("Parameter a", value=1.0, key="param_a")
            b_val = parameter_cols[1].number_input("Parameter b", value=0.5, key="param_b")
            c_val = parameter_cols[2].number_input("Parameter c", value=0.0, key="param_c")
            expression = expression.subs({sp.Symbol("a"): a_val, sp.Symbol("b"): b_val, sp.Symbol("c"): c_val})
            derivative = sp.diff(expression, x_symbol)
            second_derivative = sp.diff(derivative, x_symbol)
            st.latex(sp.latex(sp.Eq(sp.Function("f")(x_symbol), expression)))
            if course == "Calculus":
                st.write("First derivative:")
                st.latex(sp.latex(derivative))
                st.write("Second derivative:")
                st.latex(sp.latex(second_derivative))
            elif course == "Precalculus":
                st.write("The first derivative describes the instantaneous rate of change; the second derivative describes how that rate changes.")
            else:
                st.write("Focus: domain/range, graph shape, end behavior, and average rate of change.")
            x_min, x_max = st.slider("Analysis interval", -20.0, 20.0, (-5.0, 5.0), 0.5)
            if x_max <= x_min:
                st.error("The analysis interval must have a larger right endpoint.")
            else:
                f_num = sp.lambdify(x_symbol, expression, "numpy")
                df_num = sp.lambdify(x_symbol, derivative, "numpy")
                d2_num = sp.lambdify(x_symbol, second_derivative, "numpy")
                x0 = st.slider("Current input x", float(x_min), float(x_max), float((x_min+x_max)/2))
                u1, u2 = st.columns(2)
                sec_a = u1.number_input("Secant interval start", value=float(x_min))
                sec_b = u2.number_input("Secant interval end", value=float(x_max))
                try:
                    fx = float(f_num(x0))
                    dfx = float(df_num(x0))
                    if not (math.isfinite(fx) and math.isfinite(dfx)):
                        raise ValueError("Function is undefined at the selected input.")
                    st.write(f"Context: {description}.")
                    st.write(f"At {x_name} = {x0:g}, {y_name} = {fx:.6g} {y_units}; rate of change = {dfx:.6g} {y_units}/{x_units}.")
                    secant_slope = None
                    secant_start_value = None
                    if sec_b != sec_a:
                        fa, fb = float(f_num(sec_a)), float(f_num(sec_b))
                        if not (math.isfinite(fa) and math.isfinite(fb)):
                            raise ValueError("The function is undefined at one of the secant endpoints.")
                        avg_rate = (fb-fa)/(sec_b-sec_a)
                        secant_slope = avg_rate
                        secant_start_value = fa
                        st.write(f"Average rate of change on [{sec_a:g}, {sec_b:g}] = **{avg_rate:.6g} {y_units}/{x_units}**.")
                    else:
                        st.warning("Choose different secant endpoints.")
                    xx = np.linspace(float(x_min), float(x_max), 500)
                    yy = np.array([float(f_num(v)) if np.isfinite(float(f_num(v))) else np.nan for v in xx])
                    show_f = st.checkbox("Show function", True)
                    show_d1 = st.checkbox("Show first derivative", course == "Calculus")
                    show_d2 = st.checkbox("Show second derivative", False)
                    fig2, ax2 = plt.subplots(figsize=(9, 4.5))
                    if show_f:
                        ax2.plot(xx, yy, label="f(x)")
                        ax2.scatter([x0], [fx], label="Current point")
                        tangent = fx + dfx*(xx-x0)
                        ax2.plot(xx, tangent, linestyle="--", alpha=.7, label="Tangent line at x")
                        if secant_slope is not None:
                            secant_line = secant_start_value + secant_slope*(xx-sec_a)
                            ax2.plot(xx, secant_line, linestyle=":", alpha=.8, label="Secant line")
                    if show_d1:
                        dyy = np.array([float(df_num(v)) if np.isfinite(float(df_num(v))) else np.nan for v in xx])
                        ax2.plot(xx, dyy, label="f'(x)")
                    if show_d2:
                        d2yy = np.array([float(d2_num(v)) if np.isfinite(float(d2_num(v))) else np.nan for v in xx])
                        ax2.plot(xx, d2yy, label="f''(x)")
                    ax2.axhline(0, linewidth=.7)
                    ax2.grid(alpha=.2)
                    ax2.legend()
                    st.pyplot(fig2, use_container_width=True)
                    plt.close(fig2)
                    if course == "Calculus":
                        critical = sp.solveset(derivative, x_symbol, domain=sp.Interval(x_min, x_max))
                        st.write(f"Critical-number candidates in interval: `{critical}`. Candidates are not automatically maxima or minima.")
                        st.write(f"Concavity is determined by the sign of: `{second_derivative}`. A zero of f'' is only an inflection candidate; concavity must change.")
                        st.caption("For complicated functions, exact root classification may not be available in closed form; numerical graphing is an approximation.")
                except (ValueError, TypeError, OverflowError, ZeroDivisionError) as exc:
                    st.warning(f"Could not evaluate at the selected point/interval: {exc}")
    except (ValueError, np.linalg.LinAlgError) as exc:
        st.error(f"Model cannot be fit: {exc}")
else:
    st.info("Load valid observations or correct the editable table before fitting a model.")

# Downloads use the current data and calculations, even if model fitting was unavailable.
st.subheader("Export current work")
export_df = st.session_state["working_df"].copy()
csv_bytes = export_df.to_csv(index=False).encode("utf-8")
st.download_button("Download working data (CSV)", csv_bytes, "working_data.csv", "text/csv")
try:
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        export_df.to_excel(writer, sheet_name="Working Data", index=False)
        pd.DataFrame([{"Country": country_name, "Indicator": indicator_name,
                       "World Bank code": indicator_code, "Units": units,
                       "Edited scenario": st.session_state.get("edited", False),
                       "Retrieved": date.today().isoformat()}]).to_excel(writer, sheet_name="Summary", index=False)
        st.session_state.get("calc_results", pd.DataFrame()).to_excel(writer, sheet_name="Calculated Years", index=False)
    st.download_button("Download Excel workbook (.xlsx)", excel_buffer.getvalue(),
                       "hispanic_heritage_analysis.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
except Exception as exc:
    st.warning(f"Excel export is unavailable: {exc}")

if not data.empty and "chart_bytes" in locals():
    try:
        metadata = {"Country": country_name, "Indicator": indicator_name,
                    "World Bank code": indicator_code, "Units": units,
                    "Retrieved": date.today().isoformat(),
                    "Data status": "User-modified scenario" if st.session_state.get("edited") else "Original API data"}
        pdf_bytes = pdf_report(metadata, str(np.poly1d(coeff)),
                               f"In-sample R-squared: {r2:.6g}; in-sample MAE: {mae:.6g} {units}",
                               chart_bytes, clean.head(20), st.session_state.get("calc_results", pd.DataFrame()))
        st.download_button("Download PDF report", pdf_bytes, "hispanic_heritage_report.pdf", "application/pdf")
    except Exception as exc:
        st.warning(f"PDF export is unavailable: {exc}")

st.divider()
st.caption("Data source: World Bank API. Historical associations and polynomial fits do not establish causes or guarantee future outcomes. Extrapolations are especially uncertain. Medical-context functions are classroom mathematics examples, not treatment guidance.")
