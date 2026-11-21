"""
CHD microsimulation explorer, birth through age 17.

Reads the tidy files that collect_app_data.py writes on the server
(copy that app_data/ folder next to this file):
    app_data/cumulative_by_rep.csv
    app_data/prevalence_by_age.csv
    app_data/costs_by_age.csv      (optional)
    app_data/survival_by_rep.csv   (optional)
    app_data/cumulative_by_age_by_rep.csv   (optional)
    app_data/meta.json

Run locally with:  streamlit run app.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_DIR = Path(__file__).parent / "app_data"

HS_LABELS = {0: "No CHD", 2: "Mild", 3: "Moderate", 4: "Severe"}
HS_COLORS = {0: "#7f7f7f", 2: "#4c9be8", 3: "#e8a33d", 4: "#c8433b"}
CHD_STATES = [2, 3, 4]

# Binary outcomes simulated annually (or at birth) through 17.
CONDITION_LABELS = {
    "dev_delay": "Developmental delay",
    "int_dis_modsev": "Intellectual disability (mod/severe)",
    "anxiety": "Anxiety",
    "depression": "Depression",
    "behavioral_problems": "Behavioral problems",
    "spec_care": "Special health care needs",
    "er_vis": "Any ER visit (parent report)",
    "spec_vis": "Any specialist visit (parent report)",
    "stopwork_ind": "Parent stopped working",
    "essentials_difficulty": "Difficulty affording essentials",
    "access_to_care": "Access to care problem",
    "no_exercise": "No exercise",
    "low_grades": "Low grades",
    "school_absence": "School absence",
}

COST_LABELS = {
    "tot_pay": "Insurer payments, total",
    "ip_pay": "Insurer payments, inpatient",
    "noned_pay": "Insurer payments, outpatient (non-ED)",
    "ed_pay": "Insurer payments, ED",
    "rx_pay": "Insurer payments, prescriptions",
    "oop_total": "Out-of-pocket spending",
    "n_ed_visits": "ED visits (claims)",
    "n_inpatient_visits": "Inpatient admissions (claims)",
}
DOLLAR_COSTS = {"tot_pay", "ip_pay", "noned_pay", "ed_pay", "rx_pay", "oop_total"}

# The MEPS expenditure model (totexp) is retired in favor of the MarketScan
# claims models. collect_app_data.py drops it; this also hides it if an older
# app_data folder still has it.
RETIRED = ("totexp",)

# Claims costs come in three payer versions: the expected cost under the
# simulated payer mix (no suffix), and the cost if every child were
# priced as commercial (_comm) or as Medicaid (_mcaid). The app shows
# one version at a time, chosen with a "Price as" dropdown shown next to cost
# outcomes.
PAYER_OPTIONS = {
    "Payer mix (NSCH)": "",
    "Everyone commercial": "_comm",
    "Everyone Medicaid": "_mcaid",
}
PAYER_SUFFIXES = ("_comm", "_mcaid")

INTERVENTION_LABELS = {"mort": "Mortality"}
INTERVENTION_LABELS.update(CONDITION_LABELS)


def cumulative_label(col):
    if col == "life_years":
        return "Life years, 0-17"
    if col == "wvas_0_17":
        return "QALYs (GRACE), 0-17"
    if col == "died_before_18":
        return "Died before 18 (%)"
    if col.startswith("years_"):
        base = col[len("years_"):]
        lab = CONDITION_LABELS.get(base, base)
        return "Years with {}, 0-17".format(lab[:1].lower() + lab[1:])
    if col.startswith("total_"):
        base = col[len("total_"):]
        unit = " ($)" if base in DOLLAR_COSTS else ""
        return "{}, 0-17{}".format(COST_LABELS.get(base, base), unit)
    return col


def scale(col):
    """Display multiplier: shares become percentages."""
    return 100.0 if col == "died_before_18" else 1.0


# Display precision. The model can't support more digits than this, so
# dollars round to the nearest $100, shares and counts to one decimal,
# years and QALYs to two.
def _base(col):
    return col[len("total_"):] if col.startswith("total_") else col


def is_dollar(col):
    return _base(col) in DOLLAR_COSTS


def decimals(col):
    if is_dollar(col):
        return 0
    if col == "died_before_18" or _base(col) in ("n_ed_visits", "n_inpatient_visits"):
        return 1
    return 2


def rnd(v, col):
    """Round a value (or array) to its display precision."""
    return np.round(v, -2) if is_dollar(col) else np.round(v, decimals(col))


def fmt(v, col, signed=False):
    if pd.isna(v):
        return "n/a"
    v = float(rnd(v, col))
    if is_dollar(col):
        sign = ("+" if v >= 0 else "-") if signed else ("-" if v < 0 else "")
        return "{}${:,.0f}".format(sign, abs(v))
    return ("{:+,.%df}" % decimals(col) if signed else "{:,.%df}" % decimals(col)).format(v)


def hover_y(col):
    """Plotly hover format for a value already passed through rnd()."""
    return ("$%{y:,.0f}" if is_dollar(col) else "%{y:,." + str(decimals(col)) + "f}")


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

def data_stamp():
    """File names and modification times in app_data/, so replacing the data
    invalidates Streamlit's cache without a restart."""
    if not DATA_DIR.exists():
        return ()
    return tuple(sorted((p.name, p.stat().st_mtime) for p in DATA_DIR.iterdir()))


@st.cache_data
def load_data(stamp):
    def read(name, required=True):
        path = DATA_DIR / name
        if not path.exists():
            if required:
                raise FileNotFoundError(str(path))
            return None
        df = pd.read_csv(path)
        df = df[[c for c in df.columns if not any(r in c for r in RETIRED)]]
        df["iv_outcome"] = df["iv_outcome"].fillna("")
        return df

    cum = read("cumulative_by_rep.csv")
    prev = read("prevalence_by_age.csv")
    costs = read("costs_by_age.csv", required=False)
    surv = read("survival_by_rep.csv", required=False)
    cumage = read("cumulative_by_age_by_rep.csv", required=False)
    with open(DATA_DIR / "meta.json") as f:
        meta = json.load(f)
    return cum, prev, costs, surv, cumage, meta


def has_payer_versions(df):
    return df is not None and any(sfx in c for c in df.columns for sfx in PAYER_SUFFIXES)


def payer_view(df, suffix):
    """The chosen payer version under the plain column names, with the other
    versions dropped, so the rest of the app never sees payer suffixes.
    e.g. suffix "_mcaid": total_tot_pay <- total_tot_pay_mcaid and
    cum_tot_pay_surv <- cum_tot_pay_mcaid_surv."""
    if df is None:
        return None
    out = df.copy()
    for c in df.columns:
        for sfx in PAYER_SUFFIXES:
            if sfx in c:
                if sfx == suffix:
                    out[c.replace(sfx, "", 1)] = df[c]
                out = out.drop(columns=c)
    return out


@st.cache_data
def load_views(stamp, suffix):
    cum, prev, costs, surv, cumage, meta = load_data(stamp)
    return (payer_view(cum, suffix), prev, payer_view(costs, suffix), surv,
            payer_view(cumage, suffix), meta)


def summarize_levels(df, col):
    """Mean across reps and 95% Monte Carlo interval half-width, by heart_state."""
    g = df.groupby("heart_state")[col]
    out = pd.DataFrame({"mean": g.mean(), "sd": g.std(ddof=1), "n": g.count()})
    out["hw"] = 1.96 * out["sd"] / np.sqrt(out["n"])
    return out.reset_index()


def paired(df, scenario, col, keys):
    """Per-rep scenario minus baseline, matched on sim + keys, then mean and
    95% Monte Carlo half-width. Baseline and scenarios share random draws,
    so this is far tighter than differencing the level SDs."""
    cols = ["sim"] + keys + [col]
    base = df[df["scenario"] == "baseline"][cols]
    scen = df[df["scenario"] == scenario][cols]
    m = scen.merge(base, on=["sim"] + keys, suffixes=("_s", "_b"))
    m["d"] = m[col + "_s"] - m[col + "_b"]
    g = m.groupby(keys)
    out = pd.DataFrame({
        "base": g[col + "_b"].mean(),
        "level": g[col + "_s"].mean(),
        "delta": g["d"].mean(),
        "hw": 1.96 * g["d"].std(ddof=1) / np.sqrt(g["d"].count()),
    })
    return out.reset_index()


def intervention_grid(df, iv):
    """[(pct_reduction, scenario)] for one intervention, baseline at 0."""
    rows = df[df["iv_outcome"] == iv][["pct_reduction", "scenario"]].drop_duplicates()
    rows = rows.sort_values("pct_reduction")
    return [(0.0, "baseline")] + list(rows.itertuples(index=False, name=None))


def unaffected(df, grid, col):
    """True if the intervention leaves this outcome exactly unchanged for every
    person-group, rep, and simulated intensity. Baseline and intervention runs
    share every random draw, so an outcome the intervention cannot reach (given
    the model's order) comes out bit-for-bit identical; any reachable outcome
    moves at least a little. This reads that off the data instead of hard-coding
    the model's structure, so it stays correct as the model changes."""
    base = df[df["scenario"] == "baseline"][["sim", "heart_state", col]]
    for _, scen in grid[1:]:
        m = df[df["scenario"] == scen][["sim", "heart_state", col]].merge(
            base, on=["sim", "heart_state"], suffixes=("_s", "_b"))
        d = (m[col + "_s"] - m[col + "_b"]).abs()
        if (d.fillna(0) > 1e-12).any():
            return False
    return True


def bracket(grid, x):
    """The two simulated scenarios around x and the weight on the upper one."""
    pts = [p for p, _ in grid]
    x = min(max(x, pts[0]), pts[-1])
    j = int(np.searchsorted(pts, x, side="left"))
    if j < len(pts) and np.isclose(pts[j], x):
        return grid[j][1], grid[j][1], 0.0
    lo, hi = grid[j - 1], grid[j]
    return lo[1], hi[1], (x - lo[0]) / (hi[0] - lo[0])


def effects_at(df, grid, x, col, keys):
    """Paired effects at reduction x, linearly interpolated between the two
    nearest simulated intensities. Interval half-widths are interpolated the
    same way, which slightly overstates them between grid points."""
    s0, s1, t = bracket(grid, x)
    e0 = paired(df, s0, col, keys).set_index(keys)
    if t == 0.0:
        out = e0
    else:
        e1 = paired(df, s1, col, keys).set_index(keys)
        out = (1 - t) * e0 + t * e1
    out = out.reset_index()
    out["pct"] = 100.0 * out["delta"] / out["base"]
    out["pct_hw"] = 100.0 * out["hw"] / out["base"].abs()
    return out


def means_at(data, grid, x, col, keys):
    """Scenario means at reduction x, linearly interpolated (no per-rep data)."""
    s0, s1, t = bracket(grid, x)
    m0 = data[data["scenario"] == s0].set_index(keys)[col]
    if t == 0.0:
        return m0.reset_index()
    m1 = data[data["scenario"] == s1].set_index(keys)[col]
    return ((1 - t) * m0 + t * m1).reset_index()


def intensity_input(grid, key, label="Reduction in risk (%)"):
    """Continuous intensity, bounded by the simulated grid (no extrapolation)."""
    pts = [p for p, _ in grid]
    hi = float(pts[-1])
    x = st.number_input(label, min_value=0.0, max_value=hi, value=min(10.0, hi),
                        step=0.25, format="%.2f", key=key)
    st.caption("Simulated at {}%. Values in between are linearly interpolated; values above "
               "{:g}% are not available.".format(", ".join("{:g}".format(p) for p in pts[1:]), hi))
    return x


def group_picker(key, default):
    groups = st.multiselect("Groups", [HS_LABELS[h] for h in [0] + CHD_STATES],
                            default=[HS_LABELS[h] for h in default], key=key)
    return [h for h, l in HS_LABELS.items() if l in groups]


# ---------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------

st.set_page_config(page_title="CHD microsimulation explorer", layout="wide")

try:
    _raw = load_data(data_stamp())
except FileNotFoundError as e:
    st.error("Could not find the app data ({}). Copy the app_data/ folder from the "
             "server next to app.py.".format(e))
    st.stop()

# One view per payer pricing. The app's structure (outcome lists, grids)
# comes from the default payer-mix view; the "Price as" pickers next to
# cost outcomes swap in another view for that tab only.
HAS_PAYER = has_payer_versions(_raw[0])
views = {sfx: load_views(data_stamp(), sfx)
         for sfx in (PAYER_OPTIONS.values() if HAS_PAYER else [""])}
cum, prev, costs, surv, cumage, meta = views[""]

PAYER_HELP = ("Payer mix weights each year's commercial and Medicaid cost by the chance the child "
              "is covered by each, from NSCH by age, CHD severity, and family characteristics. "
              "Children with both kinds of coverage are priced as commercial, and uninsured children "
              "with CHD as Medicaid. The other two choices price every child as one payer.")


def payer_picker(container, key):
    """'Price as' dropdown for a cost outcome; returns the payer suffix."""
    if not HAS_PAYER:
        return ""
    label = container.selectbox("Price as", list(PAYER_OPTIONS.keys()), key=key, help=PAYER_HELP)
    return PAYER_OPTIONS[label]


def cost_note(suffix=""):
    """Caption text for any cost view, naming the payer pricing in use."""
    if not HAS_PAYER:
        return "Claims costs are from commercial claims models."
    return {"": "Claims costs are priced at the simulated payer mix (commercial and Medicaid).",
            "_comm": "Claims costs are priced as if every child had commercial coverage.",
            "_mcaid": "Claims costs are priced as if every child had Medicaid (fee-for-service)."
            }[suffix]

st.title("Congenital heart disease: birth through age 17")
st.caption(
    "Microsimulation of children born with and without CHD, by severity. Values are per child "
    "born, weighted to NSCH, averaged over {} Monte Carlo replications.".format(meta.get("nsim", "?"))
)

value_cols = [c for c in cum.columns
              if c not in ("scenario", "iv_outcome", "rr", "pct_reduction", "sim", "heart_state",
                           "n_unweighted")]
# Baseline summary table: (section, columns), in display order.
HEADLINE_SECTIONS = [
    ("Survival and quality of life", ["life_years", "wvas_0_17", "died_before_18"]),
    ("Development and health", ["years_dev_delay", "years_spec_care", "years_anxiety",
                                "years_depression"]),
    ("Family", ["years_stopwork_ind"]),
    ("Health care use", ["total_n_ed_visits", "total_n_inpatient_visits"]),
    ("Spending", ["total_tot_pay", "total_oop_total"]),
]
HEADLINE_SECTION = {c: sec for sec, cols in HEADLINE_SECTIONS for c in cols}
headline = [c for _, cols in HEADLINE_SECTIONS for c in cols if c in value_cols]
ordered_cols = headline + [c for c in value_cols if c not in headline]
label_to_col = {cumulative_label(c): c for c in ordered_cols}

interventions = [iv for iv in cum["iv_outcome"].unique() if iv]
iv_label_to_key = {INTERVENTION_LABELS.get(iv, iv): iv for iv in interventions}

tab_about, tab_base, tab_iv, tab_age = st.tabs(
    ["About", "Baseline", "Interventions", "By age"])

# ---------------------------------------------------------------------
# About (landing page)
# ---------------------------------------------------------------------
FLOW_SVG = Path(__file__).parent / "model_flow.svg"   # rendered from model_flow.dot


with tab_about:
    st.subheader("About this model")
    st.markdown(
        "This explorer shows results from a microsimulation of children born in the United States "
        "with and without congenital heart disease (CHD). It follows each simulated child year by "
        "year from birth, tracking health, health care use, effects on the family, quality of life, "
        "and medical spending, separately for children born with **mild**, **moderate**, or "
        "**severe** CHD and children born **without CHD**.\n\n"
        "This version covers **birth through age 17**, with survival through age 24. The childhood "
        "model is the first part of a life-course model that continues into adulthood (ages 25 and "
        "older) through the Future Adult Model. Lifetime results will be added here as that link "
        "is completed."
    )

    st.markdown("#### How it works")
    st.markdown(
        "- **Starting population.** Infants in the National Survey of Children's Health (NSCH), "
        "2016-2024, weighted to represent U.S. births. A child's CHD severity is set at birth and "
        "does not change.\n"
        "- **Each year of childhood.** Outcomes are drawn from statistical models estimated on NSCH "
        "children of all ages. Some outcomes are drawn jointly so that conditions that tend to occur "
        "together do so in the simulation as well.\n"
        "- **Mortality.** U.S. life table death rates, multiplied by relative risks that depend on "
        "CHD severity and age.\n"
        "- **Costs.** Annual spending and utilization from models estimated on commercial and "
        "Medicaid insurance claims for children with CHD, combined according to each child's "
        "chance of being covered by each at that age (from NSCH).\n"
        "- **Quality of life.** Health status is converted to quality-adjusted life years (QALYs) "
        "using the GRACE utility transform.\n"
        "- **Repetition.** The whole population is simulated many times with different random draws, "
        "and results are averaged across those replications."
    )

    st.markdown("#### The model runs in a fixed order")
    if FLOW_SVG.exists():
        flow_col, _ = st.columns([6, 1])
        flow_col.image(str(FLOW_SVG), width="stretch")
    st.markdown(
        "Each child starts with a set of **initial conditions** drawn at birth. Some are known "
        "right away; others, such as an eventual anxiety or depression diagnosis, are drawn at "
        "birth together with the age at which they appear, and are revealed when the child "
        "reaches that age. Drawing these jointly keeps conditions that tend to occur together "
        "linked in the simulation.\n\n"
        "Then, **each year of age**, the model steps through the cycle in order. Each step can "
        "use the results of earlier steps, but never later ones, and nothing feeds back into an "
        "earlier step the following year. **Interventions therefore flow downstream only.** "
        "For example:\n\n"
        "- Reducing **depression** can change ER and specialist visits, impacts on parents, "
        "school outcomes, self-rated health, and QALYs.\n"
        "- Reducing **ER visits** can change impacts on parents, but cannot change depression, "
        "which comes earlier.\n"
        "- **Survival** depends only on age and CHD severity. Only the mortality intervention "
        "changes it; reducing a condition does not make a child more likely to survive in "
        "this version.\n"
        "- **Costs** currently depend only on age, severity, sex, and the year of death. "
        "Condition interventions therefore do not change costs. Mortality interventions do, "
        "because more children are alive and receiving care."
    )

    st.markdown("#### What the model is, and what it isn't")
    st.markdown(
        "- **The relationships are associations, not proven causal effects.** They come from survey "
        "data. An intervention result answers the question: *if this risk were lower, and "
        "everything downstream responded as the estimated relationships imply, what would change?*\n"
        "- **Interventions are hypothetical.** Each one lowers the probability of a single outcome "
        "by a fixed percentage, for every child, at every age that outcome is simulated. They are "
        "not tied to a specific program, its reach, or its cost, and the cost of delivering an "
        "intervention is not included.\n"
        "- **Treatment is not modeled directly.** Surgery and other CHD care are reflected in "
        "outcomes and costs as they occur in the data, not as separate choices that can be changed.\n"
        "- **Survival reflects current death rates.** It uses today's life tables, not the history "
        "of older birth cohorts, so it can differ from long-term follow-up studies of children born "
        "decades ago.\n"
        "- **Costs are preliminary.** They are not yet estimated for children without CHD. Medicaid "
        "costs come from fee-for-service claims, do not yet vary by race, and use one model for "
        "the year of death and other years. Insurance coverage after 17 is held at its age 14-17 "
        "pattern.\n"
        "- **Ages 18 to 24** carry mortality and costs forward. Other outcomes stop at 17.\n"
        "- **Uncertainty shown is simulation noise only.** The intervals reflect random variation "
        "across replications. They do not include uncertainty in the estimated model parameters, "
        "so they understate total uncertainty."
    )

    st.markdown("#### How the outcomes relate to the ICHOM Standard Set")
    st.markdown(
        "The International Consortium for Health Outcomes Measurement (ICHOM) Standard Set for "
        "Congenital Heart Disease defines the outcomes that matter most to patients and families, "
        "in four domains. The table shows where the simulated childhood outcomes fit. The model's "
        "measures are parent-reported survey indicators from the NSCH, not the clinical tools or "
        "questionnaires ICHOM recommends, so the correspondence is conceptual rather than exact."
    )
    ichom_map = pd.DataFrame([
        ["Overall health", "Health-related quality of life / perceived health status",
         "Self-rated health; QALYs (GRACE)", "Yes"],
        ["Overall health", "Quality of life (life satisfaction)", "", "Not modeled"],
        ["Physical health", "Survival", "Survival; died before 18; life years", "Yes"],
        ["Physical health", "Development (developmental delay)",
         "Developmental delay; intellectual disability", "Yes"],
        ["Physical health", "Activity level", "No exercise", "Partial"],
        ["Physical health", "Growth, heart failure, arrhythmias", "", "Not modeled"],
        ["Social health", "Productivity (school days missed)", "School absence; low grades", "Yes"],
        ["Social health", "Family quality of life", "Parent stopped working", "Partial"],
        ["Social health", "Financial burden",
         "Affording essentials; access to care; out-of-pocket costs", "Yes"],
        ["Mental health", "Development (mental health diagnoses)",
         "Anxiety; depression; behavioral problems", "Yes"],
    ], columns=["ICHOM domain", "ICHOM outcome", "Model outcomes", "Coverage"])
    st.dataframe(ichom_map, width="stretch", hide_index=True)
    st.markdown(
        "The model also tracks outcomes that ICHOM does not include as patient outcomes but that "
        "matter for costs and planning: **ER and specialist visits, special health care needs, and "
        "medical spending**. ICHOM's adult outcomes (adult quality of life, work productivity, "
        "adult depression and anxiety, cognition, pregnancy) belong to the adult part of the "
        "life-course model and are not shown here."
    )
    st.caption("ICHOM Congenital Heart Disease Standard Set, version 1.0, April 2020. "
               "[ichom.org](https://www.ichom.org/portfolio/congenital-heart-disease/)")

    st.markdown("#### Reading the results")
    st.markdown(
        "- Results are **per child born**, including children who die during childhood. A "
        "mortality intervention can therefore increase both life years *and* years lived with a "
        "condition, since more children are alive to experience it.\n"
        "- Intervention results are compared with **the same group's own baseline**, run with the "
        "same random draws, so the difference reflects the intervention rather than chance.\n"
        "- Numbers are rounded to avoid implying more precision than the model supports."
    )



# ---------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------
with tab_base:
    st.subheader("Childhood outcomes by CHD severity")
    base = cum[cum["scenario"] == "baseline"]

    rows = []
    for col in headline:
        s = summarize_levels(base, col)
        for _, r in s.iterrows():
            rows.append({"Outcome": cumulative_label(col),
                         "heart_state": int(r["heart_state"]),
                         "value": fmt(r["mean"] * scale(col), col)})
    if rows:
        t = pd.DataFrame(rows).pivot(index="Outcome", columns="heart_state", values="value")
        t = t.reindex(index=[cumulative_label(c) for c in headline])
        t.columns = [HS_LABELS.get(c, c) for c in t.columns]
        t = t.fillna("n/a").reset_index()
        # Section label on the first row of each section only.
        secs = [HEADLINE_SECTION[c] for c in headline]
        t.insert(0, "", [sec if i == 0 or sec != secs[i - 1] else "" for i, sec in enumerate(secs)])
        st.dataframe(t, width="stretch", hide_index=True, height=35 * (len(t) + 1) + 3)
        if any(c.startswith("total_") for c in headline):
            st.caption(cost_note())

    b1, b2, _ = st.columns(3)
    choice = b1.selectbox("Outcome", list(label_to_col.keys()), key="base_outcome")
    col = label_to_col[choice]
    base_payer = payer_picker(b2, "base_payer") if col.startswith("total_") else ""
    base_v = views[base_payer][0]
    base_v = base_v[base_v["scenario"] == "baseline"]
    s = summarize_levels(base_v, col).dropna(subset=["mean"])
    s = s[s["heart_state"].isin(HS_LABELS)]
    fig = go.Figure(go.Bar(
        x=[HS_LABELS[h] for h in s["heart_state"]],
        y=rnd(s["mean"] * scale(col), col),
        error_y=dict(type="data", array=s["hw"] * scale(col)),
        marker_color=[HS_COLORS[h] for h in s["heart_state"]],
        hovertemplate="%{x}: " + hover_y(col) + "<extra></extra>",
    ))
    fig.update_layout(yaxis_title=choice, template="plotly_white", height=400,
                      margin=dict(t=30), bargap=0.6)
    chart_col, _ = st.columns([1, 1])
    chart_col.plotly_chart(fig, width="stretch")
    if col.startswith("total_"):
        st.caption(cost_note(base_payer) + " Costs come from claims models estimated on children with CHD, "
                   "so they are not defined for the no-CHD group.")

# ---------------------------------------------------------------------
# Interventions
# ---------------------------------------------------------------------
with tab_iv:
    st.subheader("What an intervention changes, relative to each group's own baseline")
    if not interventions:
        st.info("No intervention scenarios in this data.")
    else:
        c1, c2, c3 = st.columns(3)
        iv_label = c1.selectbox("Intervention", list(iv_label_to_key.keys()), key="iv_sel")
        iv = iv_label_to_key[iv_label]
        grid = intervention_grid(cum, iv)
        untouched_cols = {c for c in ordered_cols if unaffected(cum, grid, c)}

        # The intervention's own target first, then everything it reaches,
        # then (marked) the outcomes it cannot reach in this model.
        target = "died_before_18" if iv == "mort" else "years_" + iv
        reached = [c for c in ordered_cols if c not in untouched_cols]
        reached = [c for c in reached if c == target] + [c for c in reached if c != target]
        opt_cols = reached + [c for c in ordered_cols if c in untouched_cols]
        opt_labels = {c: cumulative_label(c) + (" (not affected)" if c in untouched_cols else "")
                      for c in opt_cols}
        col = c2.selectbox("Outcome", opt_cols, format_func=lambda c: opt_labels[c],
                           key="iv_outcome_sel_" + iv)
        out_label = cumulative_label(col)
        mode = c3.radio("Show", ["Change from baseline", "Percent change", "Level"])
        iv_payer = payer_picker(st.columns(3)[1], "iv_payer") if col.startswith("total_") else ""
        cum_iv = views[iv_payer][0]
        hs_sel = group_picker("iv_groups", CHD_STATES)

        x = intensity_input(grid, key="iv_x")

        if col in untouched_cols:
            st.info("{} is not affected by the {} intervention in the current model. It either comes "
                    "before {} in the model's order or does not depend on it, so the intervention "
                    "leaves it exactly unchanged.".format(out_label, iv_label.lower(), iv_label.lower()))
        else:
            frames = []
            for pct, scen in grid:
                e = effects_at(cum_iv, grid, pct, col, ["heart_state"])
                e["pct_reduction"] = pct
                frames.append(e)
            eff = pd.concat(frames, ignore_index=True)
            here = effects_at(cum_iv, grid, x, col, ["heart_state"])

            def pick(df):
                if mode == "Change from baseline":
                    return rnd(df["delta"] * scale(col), col), df["hw"] * scale(col), "Change in " + out_label
                if mode == "Percent change":
                    return np.round(df["pct"], 1), df["pct_hw"], "% change in " + out_label
                return rnd(df["level"] * scale(col), col), None, out_label

            hv = "%{y:+.1f}%" if mode == "Percent change" else hover_y(col)

            fig = go.Figure()
            ytitle = out_label
            for h in hs_sel:
                sub = eff[eff["heart_state"] == h].sort_values("pct_reduction")
                if sub.empty or sub["level"].isna().all():
                    continue
                y, hw, ytitle = pick(sub)
                fig.add_trace(go.Scatter(
                    x=sub["pct_reduction"], y=y, mode="lines+markers", name=HS_LABELS[h],
                    line=dict(color=HS_COLORS[h]),
                    error_y=dict(type="data", array=hw) if hw is not None else None,
                    hovertemplate=HS_LABELS[h] + ": " + hv + "<extra></extra>",
                ))
                pt = here[here["heart_state"] == h]
                yp, _, _ = pick(pt)
                fig.add_trace(go.Scatter(
                    x=[x], y=yp, mode="markers", showlegend=False,
                    marker=dict(color=HS_COLORS[h], size=13, symbol="diamond",
                                line=dict(color="white", width=1)),
                    hovertemplate=HS_LABELS[h] + " at %{x:.2f}%: " + hv + "<extra></extra>",
                ))
            fig.add_vline(x=x, line_color="#999999", line_dash="dot", line_width=1)
            fig.update_layout(
                xaxis_title="Reduction in {} risk (%)".format(iv_label.lower()),
                yaxis_title=ytitle, template="plotly_white", height=450, margin=dict(t=30),
                legend_title="Severity",
            )
            if mode != "Level":
                fig.add_hline(y=0, line_color="#bbbbbb", line_width=1)
            st.plotly_chart(fig, width="stretch")

        st.markdown("**All outcomes at {:g}% reduction**".format(x))
        rows, shown, untouched = [], [], []
        for c in ordered_cols:
            if c in untouched_cols:
                untouched.append(cumulative_label(c))
                continue
            e = effects_at(cum_iv, grid, x, c, ["heart_state"])
            for _, r in e.iterrows():
                h = int(r["heart_state"])
                if h not in hs_sel or pd.isna(r["delta"]):
                    continue
                vals = [r["base"] * scale(c), r["level"] * scale(c),
                        r["delta"] * scale(c), r["hw"] * scale(c)]
                rows.append({"Outcome": cumulative_label(c), "Group": HS_LABELS.get(h, h),
                             "Baseline": rnd(vals[0], c), "With intervention": rnd(vals[1], c),
                             "Change": rnd(vals[2], c), "95% MC ±": rnd(vals[3], c),
                             "% change": round(r["pct"], 1)})
                shown.append({"Outcome": cumulative_label(c), "Group": HS_LABELS.get(h, h),
                              "Baseline": fmt(vals[0], c), "With intervention": fmt(vals[1], c),
                              "Change": fmt(vals[2], c, signed=True), "95% MC ±": fmt(vals[3], c),
                              "% change": "n/a" if pd.isna(r["pct"]) else "{:+.1f}%".format(r["pct"])})
        tbl = pd.DataFrame(rows)
        if len(tbl):
            st.dataframe(pd.DataFrame(shown), width="stretch", hide_index=True, height=420)
            st.download_button("Download table (CSV)", tbl.to_csv(index=False).encode("utf-8"),
                               file_name="chd_{}_{:g}pct.csv".format(iv, x), mime="text/csv")
        if untouched:
            st.markdown("**Not affected by this intervention in the current model** (they come "
                        "before it in the model's order, or do not depend on it): " +
                        "; ".join(untouched) + ".")

        st.caption(
            cost_note(iv_payer) + " " +
            "Each intervention multiplies the predicted probability of its outcome by (1 - reduction) "
            "for every child, at every age it is simulated. Intervals are 95% Monte Carlo intervals "
            "for the paired difference (baseline and intervention runs share random draws); they do "
            "not reflect uncertainty in the estimated model parameters."
        )

# ---------------------------------------------------------------------
# Survival (shown in the By age tab)
# ---------------------------------------------------------------------
def show_survival(iv_label_s, hs_s):
    """Survival curves for the By age tab (Measure = Survival)."""
    mode_s = st.radio("Show", ["Survival", "Change (percentage points)"], horizontal=True,
                      key="surv_mode")
    if iv_label_s == "(none)":
        grid_s, x_s = [(0.0, "baseline")], 0.0
    else:
        grid_s = intervention_grid(surv, iv_label_to_key[iv_label_s])
        x_s = intensity_input(grid_s, key="surv_x")

    eff_s = effects_at(surv, grid_s, x_s, "survival", ["heart_state", "age"])
    fig = go.Figure()
    for h in hs_s:
        sub = eff_s[eff_s["heart_state"] == h].sort_values("age")
        if sub.empty:
            continue
        if mode_s == "Survival":
            fig.add_trace(go.Scatter(
                x=sub["age"], y=100 * sub["base"], mode="lines", name=HS_LABELS[h],
                line=dict(color=HS_COLORS[h]),
                hovertemplate=HS_LABELS[h] + ", age %{x}: %{y:.1f}%<extra></extra>"))
            if x_s > 0:
                fig.add_trace(go.Scatter(
                    x=sub["age"], y=100 * sub["level"], mode="lines",
                    name=HS_LABELS[h] + " ({:g}% reduction)".format(x_s),
                    line=dict(color=HS_COLORS[h], dash="dash"),
                    hovertemplate=HS_LABELS[h] + " with intervention, age %{x}: %{y:.1f}%<extra></extra>"))
        else:
            fig.add_trace(go.Scatter(
                x=sub["age"], y=100 * sub["delta"], mode="lines+markers", name=HS_LABELS[h],
                line=dict(color=HS_COLORS[h]),
                error_y=dict(type="data", array=100 * sub["hw"]),
                hovertemplate=HS_LABELS[h] + ", age %{x}: %{y:+.1f} pp<extra></extra>"))
    if mode_s == "Survival":
        fig.update_layout(yaxis_title="Alive at age (%)", yaxis_range=[0, 101])
    else:
        fig.update_layout(yaxis_title="Change in % alive (percentage points)")
        fig.add_hline(y=0, line_color="#bbbbbb", line_width=1)
    fig.update_layout(xaxis_title="Age", template="plotly_white", height=470, margin=dict(t=30))
    st.plotly_chart(fig, width="stretch")

    milestones = [a for a in (1, 5, 18, 24) if a in set(eff_s["age"])]
    rows = []
    for h in hs_s:
        for a in milestones:
            r = eff_s[(eff_s["heart_state"] == h) & (eff_s["age"] == a)]
            if r.empty:
                continue
            r = r.iloc[0]
            row = {"Group": HS_LABELS[h], "Age": a, "Baseline (%)": 100 * r["base"]}
            if x_s > 0:
                row.update({"With intervention (%)": 100 * r["level"],
                            "Change (pp)": 100 * r["delta"], "95% MC ± (pp)": 100 * r["hw"]})
            rows.append(row)
    if rows:
        fmt = {"Baseline (%)": "{:.1f}", "With intervention (%)": "{:.1f}",
               "Change (pp)": "{:+.1f}", "95% MC ± (pp)": "{:.1f}"}
        tbl = pd.DataFrame(rows)
        st.dataframe(tbl.style.format({k: v for k, v in fmt.items() if k in tbl.columns}),
                     width="stretch", hide_index=True)
    st.caption("Share of each birth cohort alive at each birthday. Mortality is simulated through "
               "age 24 (life-table rates with CHD severity multipliers); other childhood outcomes "
               "stop at 17.")


# ---------------------------------------------------------------------
# By age
# ---------------------------------------------------------------------
with tab_age:
    st.subheader("Age profiles, baseline vs. intervention")
    sources = {}
    if surv is not None:
        sources["Survival (% alive)"] = ("surv", "survival")
    for c in CONDITION_LABELS:
        if c in prev.columns:
            sources["Prevalence: " + CONDITION_LABELS[c]] = ("prev", c)
    if costs is not None:
        for c in COST_LABELS:
            if c in costs.columns:
                sources["Annual: " + COST_LABELS[c]] = ("cost", c)
    if cumage is not None:
        for c in COST_LABELS:
            if "cum_" + c in cumage.columns:
                sources["Running total: " + COST_LABELS[c]] = ("cum", c)

    c1, c2 = st.columns(2)
    src_label = c1.selectbox("Measure", list(sources.keys()))
    kind, col = sources[src_label]
    iv_label_a = c2.selectbox("Intervention", ["(none)"] + list(iv_label_to_key.keys()), key="age_iv")
    hs_a = group_picker("age_groups", [0] + CHD_STATES)
    if kind == "surv":
        show_survival(iv_label_a, hs_a)
    else:
        per_survivor = False
        if kind == "cum":
            per_survivor = st.radio("Per", ["Child born", "Child alive at that age"], horizontal=True,
                                    key="age_per") == "Child alive at that age"
        age_payer = ""
        if kind in ("cost", "cum"):
            age_payer = payer_picker(st.columns(2)[0], "age_payer")
        _, _, costs_a, _, cumage_a, _ = views[age_payer]
        data = {"prev": prev, "cost": costs_a, "cum": cumage_a}[kind]
        x_a, grid_a = 0.0, None
        if iv_label_a != "(none)":
            grid_a = intervention_grid(data, iv_label_to_key[iv_label_a])
            x_a = intensity_input(grid_a, key="age_x")

        if kind == "cum":
            # Running totals carry every rep, so the base and intervention lines
            # come from the same paired comparison as the Interventions tab.
            dcol = "cum_" + col + ("_surv" if per_survivor else "")
            grid_c = grid_a if grid_a is not None else [(0.0, "baseline")]
            e = effects_at(data, grid_c, x_a, dcol, ["heart_state", "age"])
            base_a = e[["heart_state", "age"]].assign(**{col: e["base"]})
            scen_a = e[["heart_state", "age"]].assign(**{col: e["level"]}) if x_a > 0 else None
        else:
            base_a = data[data["scenario"] == "baseline"]
            scen_a = means_at(data, grid_a, x_a, col, ["heart_state", "age"]) if x_a > 0 else None
        mult = 100.0 if kind == "prev" else 1.0
        fig = go.Figure()
        for h in hs_a:
            for df, dash, suffix in [(base_a, "solid", ""),
                                     (scen_a, "dash", " ({:g}% reduction)".format(x_a))]:
                if df is None:
                    continue
                sub = df[df["heart_state"] == h].sort_values("age")
                if sub.empty or sub[col].isna().all():
                    continue
                yv = np.round(sub[col] * mult, 1) if kind == "prev" else rnd(sub[col], col)
                hy = "%{y:.1f}%" if kind == "prev" else hover_y(col)
                fig.add_trace(go.Scatter(
                    x=sub["age"], y=yv, mode="lines", name=HS_LABELS[h] + suffix,
                    line=dict(color=HS_COLORS[h], dash=dash),
                    hovertemplate=HS_LABELS[h] + suffix + ", age %{x}: " + hy + "<extra></extra>",
                ))
        fig.update_layout(xaxis_title="Age", yaxis_title=src_label.split(": ", 1)[1] +
                          (" (%)" if kind == "prev" else ""),
                          template="plotly_white", height=450, margin=dict(t=30))
        st.plotly_chart(fig, width="stretch")
        if kind == "cost":
            st.caption("Mean annual value per child alive at the start of each year of age. " + cost_note(age_payer))
        elif kind == "cum":
            st.caption(
                "Total from birth through each age. " +
                ("Among children alive at that age." if per_survivor else
                 "Per child born, so children who died earlier count what they had used up to their "
                 "death.") + " The year of death counts in full. " + cost_note(age_payer))
        else:
            st.caption("Weighted prevalence at each age, as in the NSCH validation tables.")

st.divider()
st.caption("Data generated {} on {}, revision {}.".format(
    meta.get("generated", "?"), meta.get("host", "?"), meta.get("revision", "?")))
