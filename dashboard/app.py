"""
Streamlit website for PS4: "Credit Supply and the Price of Housing" (Favara & Imbs 2015).

PS4 requirements (Problem_Set_3_4.pdf, section 5):
  [x] paper summary box in non-technical language     -> section 1
  [x] intuitive headline for every section
  [x] >= 1 chart with dynamic functionality           -> section 2 (map + year slider)
  [x] >= 1 table                                       -> section 3 (regression results, toggle paper vs. ours)
  [x] second interactive chart                         -> section 3b (Figure 3, choose credit measure)
  [x] 3 exam-style questions below the charts          -> section 4
Hosted on Streamlit Community Cloud under a team pseudonym.

Data: only small aggregates in dashboard/data/ (written by scripts/01 and 02).
Run locally:  streamlit run dashboard/app.py
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA = Path(__file__).resolve().parent / "data"
DEREG_FILE = DATA / "deregulation_state_year.csv"   # state_fips, year, dereg_index
REG_FILE = DATA / "regression_results.csv"          # table, column, coef_ours, coef_paper, se_ours, ...
IRF_FILE = DATA / "irf_house_prices.csv"            # series, horizon, coef, se (Figure 3)

FIPS_TO_ABBR = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT",
    "10": "DE", "11": "DC", "12": "FL", "13": "GA", "15": "HI", "16": "ID", "17": "IL",
    "18": "IN", "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME", "24": "MD",
    "25": "MA", "26": "MI", "27": "MN", "28": "MS", "29": "MO", "30": "MT", "31": "NE",
    "32": "NV", "33": "NH", "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
    "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD",
    "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA", "54": "WV",
    "55": "WI", "56": "WY",
}

st.set_page_config(page_title="Credit Supply & House Prices", layout="wide")


@st.cache_data
def load_dereg() -> pd.DataFrame | None:
    if not DEREG_FILE.exists():
        return None
    df = pd.read_csv(DEREG_FILE, dtype={"state_fips": str})
    df["state_fips"] = df["state_fips"].str.zfill(2)
    df["state"] = df["state_fips"].map(FIPS_TO_ABBR)
    return df


@st.cache_data
def load_regressions() -> pd.DataFrame | None:
    return pd.read_csv(REG_FILE) if REG_FILE.exists() else None


@st.cache_data
def load_irf() -> pd.DataFrame | None:
    return pd.read_csv(IRF_FILE) if IRF_FILE.exists() else None


# --- Header + 1. Summary box ------------------------------------------------
st.title("Does easier access to bank credit push up house prices?")
st.caption("Replication of Favara & Imbs (2015), American Economic Review · [Team name]")

with st.container(border=True):
    st.subheader("The paper in 30 seconds")
    st.markdown(
        "**Question:** [one sentence, non-technical]\n\n"
        "**Idea:** [how US banking deregulation acts as a natural experiment]\n\n"
        "**Takeaway:** [main finding in plain words]"
    )

# --- 2. Interactive map -----------------------------------------------------
st.header("Which states opened their doors to out-of-state banks, and when?")
dereg = load_dereg()
if dereg is None:
    st.info(f"Map data missing: run `scripts/01_data_cleaning.py` to create `{DEREG_FILE.name}`.")
else:
    year = st.slider("Year", min_value=1994, max_value=2005, value=1994, step=1)
    d = dereg[dereg["year"] == year]
    fig = px.choropleth(
        d, locations="state", locationmode="USA-states", scope="usa",
        color="dereg_index", range_color=(0, 4),
        color_continuous_scale="Blues",
        labels={"dereg_index": "Deregulation index (0–4)"},
        hover_name="state",
    )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=480)
    st.plotly_chart(fig, width="stretch")
    st.caption("Deregulation index = 4 minus the Rice & Strahan (2010) index: the number of the four "
               "interstate branching restrictions a state has lifted (0 = most restricted, 4 = fully open).")

# --- 3. Regression table ----------------------------------------------------
st.header("How much do house prices react to more credit?")
reg = load_regressions()
if reg is None:
    st.info("Results missing: run `scripts/02_iv_regression.py`.")
else:
    t5 = reg[reg["table"] == "Table 5"]
    show_paper = st.toggle("Compare with the published numbers", value=True)
    out = pd.DataFrame({
        "Credit measure": t5["column"].str.replace(r"^\(\d\) ", "", regex=True),
        "Effect (our replication)": t5["coef_ours"].round(3),
        "Std. error": t5["se_ours"].round(3),
    })
    if show_paper:
        out["Effect (paper)"] = t5["coef_paper"].round(3)
        out["Std. error (paper)"] = t5["se_paper"].round(3)
    out["Observations"] = t5["N_ours"].map("{:,}".format)
    st.dataframe(out, hide_index=True, width="stretch")
    st.caption("Reading example: 1 percent faster growth in mortgage lending, caused by deregulation, "
               "raises house price growth by about 0.12 to 0.14 percent in the same year.")

# --- 3b. Interactive impulse response (Figure 3) ------------------------------
st.header("How long does the effect on house prices last?")
irf = load_irf()
if irf is None:
    st.info("Results missing: run `scripts/02b_impulse_responses.py`.")
else:
    series = st.radio("Measure of credit", sorted(irf["series"].unique()), horizontal=True)
    e = irf[irf["series"] == series].sort_values("horizon")
    lo, hi = e["coef"] - 1.64 * e["se"], e["coef"] + 1.64 * e["se"]
    fig = go.Figure([
        go.Scatter(x=list(e["horizon"]) + list(e["horizon"])[::-1], y=list(hi) + list(lo)[::-1],
                   fill="toself", fillcolor="rgba(70,110,180,0.18)", line=dict(width=0),
                   hoverinfo="skip", name="90% confidence band"),
        go.Scatter(x=e["horizon"], y=e["coef"], mode="lines+markers", name="Effect",
                   line=dict(color="rgb(40,80,160)", width=2.5)),
    ])
    fig.add_hline(y=0, line_color="grey", line_width=1)
    fig.update_layout(xaxis_title="Years after the credit shock", yaxis_title="Effect on house price growth",
                      height=420, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
    st.plotly_chart(fig, width="stretch")
    st.caption("Replication of Figure 3 in Favara & Imbs (2015), local projections. "
               "The shaded band is the 90 percent confidence interval.")

# --- 4. Exam-style questions ------------------------------------------------
st.header("Test yourself")
QUESTIONS = [  # written by the team, getting harder from 1 to 3
    ("Question 1: [easy: interpretation of β]", "[answer]"),
    ("Question 2: [medium: relevance / first stage]", "[answer]"),
    ("Question 3: [hard: exclusion restriction / OVB direction]", "[answer]"),
]
for q, a in QUESTIONS:
    st.markdown(f"**{q}**")
    with st.expander("Show answer"):
        st.markdown(a)
