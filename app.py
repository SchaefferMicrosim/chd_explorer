"""
CHD microsimulation explorer: childhood (birth through 17) and the
whole life course (childhood simulation linked to the Future Adult Model).

Reads the tidy files that collect_app_data.py writes on the server
(copy that app_data/ folder next to this file):
    app_data/cumulative_by_rep.csv
    app_data/prevalence_by_age.csv
    app_data/costs_by_age.csv      (optional)
    app_data/survival_by_rep.csv   (optional)
    app_data/cumulative_by_age_by_rep.csv   (optional)
    app_data/meta.json
    app_data/lifetime_summary.csv   (optional; burden and What if? tabs)
    app_data/lifetime_diff.csv      (optional; What if? tab)

Run locally with:  streamlit run app.py
"""

import json
import re
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
    "ed_pay": "Insurer payments, ED",
    "n_inpatient_visits": "Inpatient admissions (claims)",
    "n_ed_visits": "ED visits (claims)",
    "n_op_clinic_visits": "Outpatient clinic visits (claims)",
}
DOLLAR_COSTS = {"tot_pay", "ip_pay", "ed_pay"}

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
    if col == "died_before_18" or _base(col) in ("n_ed_visits", "n_inpatient_visits",
                                                 "n_op_clinic_visits"):
        return 1
    return 2


def rnd(v, col):
    """Round a value (or array) to its display precision."""
    return np.round(v, -2) if is_dollar(col) else np.round(v, decimals(col))


def fmt(v, col, signed=False, extra=0):
    """extra: additional decimals, used for changes so a small change that is
    real does not print as +0.00 next to a nonzero percent change."""
    if pd.isna(v):
        return "n/a"
    if is_dollar(col):
        v = float(rnd(v, col))
        sign = ("+" if v >= 0 else "-") if signed else ("-" if v < 0 else "")
        return "{}${:,.0f}".format(sign, abs(v))
    d = decimals(col) + extra
    v = float(np.round(v, d))
    return ("{:+,.%df}" % d if signed else "{:,.%df}" % d).format(v)


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


def intensity_input(grid, key, label="Reduction in risk (%)", default=10.0):
    """Continuous intensity slider, bounded by the simulated grid (no extrapolation)."""
    pts = [p for p, _ in grid]
    hi = float(pts[-1])
    x = st.slider(label, min_value=0.0, max_value=hi, value=min(float(default), hi),
                  step=1.0, format="%.0f%%", key=key)
    st.caption("Simulated at {}%; in between, results are interpolated.".format(
        ", ".join("{:g}".format(p) for p in pts[1:])))
    return x


def group_picker(key, default):
    groups = st.multiselect("Groups", [HS_LABELS[h] for h in [0] + CHD_STATES],
                            default=[HS_LABELS[h] for h in default], key=key)
    return [h for h, l in HS_LABELS.items() if l in groups]


# ---------------------------------------------------------------------
# Lifetime (childhood simulation linked to FAM)
# ---------------------------------------------------------------------

# (column, label, unit). unit: "yr" years, "q" QALYs, "pct" share shown
# as %, "usd" dollars, "n" count. Per-birth outcomes count every child
# born (children who die before 25 contribute what they had); per-adult
# outcomes are averages over those alive at 25.
LIFE_PER_BIRTH = [
    ("LE", "Life expectancy (years)", "yr"),
    ("share_reach_25", "Alive at 25 (%)", "pct"),
    ("YPLL", "Years of life lost before 75", "yr"),
    ("WVAS_LIFETIME", "Lifetime QALYs (GRACE)", "q"),
    ("WVAS_LIFETIME_DIS", "Lifetime QALYs (GRACE), discounted", "q"),
    ("WVAS_0_24", "QALYs (GRACE), ages 0-24", "q"),
    ("TOTPAY_0_24", "Insurer payments, ages 0-24", "usd"),
    ("TOTPAY_0_24_DIS", "Insurer payments, ages 0-24, discounted", "usd"),
    ("TOTMD_DIS_PB", "Medical spending, ages 25+, discounted", "usd"),
    ("MCARE_DIS_PB", "Medicare spending, ages 25+, discounted", "usd"),
    ("CAIDMD_DIS_PB", "Medicaid spending, ages 25+, discounted", "usd"),
    ("IEARN_DIS_PB", "Lifetime earnings, discounted", "usd"),
]
LIFE_PER_ADULT = [
    ("WVAS_UNDIS", "QALYs (GRACE) from age 25", "q"),
    ("FIRST_EDUC3", "College degree at 25 (%)", "pct"),
    ("FIRST_EDUC4", "Beyond college at 25 (%)", "pct"),
    ("FIRST_HEARTE", "Heart disease at 25 (%)", "pct"),
    ("HEARTE_UNDIS", "Years with heart disease", "yr"),
    ("STROKE_UNDIS", "Years with stroke", "yr"),
    ("HIBPE_UNDIS", "Years with hypertension", "yr"),
    ("DIABE_UNDIS", "Years with diabetes", "yr"),
    ("LUNGE_UNDIS", "Years with lung disease", "yr"),
    ("CANCRE_UNDIS", "Years with cancer", "yr"),
    ("K6SEVERE_UNDIS", "Years with severe psychological distress", "yr"),
    ("DICLAIM_UNDIS", "Years claiming disability insurance", "yr"),
    ("TOTMD_DIS", "Medical spending from 25, discounted to 25", "usd"),
]
FAM_IV_LABELS = {"pdied": "Adult mortality", "phearte": "Adult heart disease",
                 "plunge": "Adult lung disease", "phibpe": "Adult hypertension",
                 "pdiabe": "Adult diabetes", "pstroke": "Adult stroke"}
CF_LABELS = {"as_mod": "Severe CHD as moderate", "as_mild": "Severe CHD as mild",
             "as_none": "Severe CHD as no CHD",
             "mort_as_mod": "Severe CHD with moderate-CHD mortality only",
             "mort_as_mild": "Severe CHD with mild-CHD mortality only",
             "mort_as_none": "Severe CHD with no-CHD mortality only"}
LIFE_KEYS = ["child_scenario", "fam_scenario", "heart_state"]


@st.cache_data
def load_lifetime(stamp):
    """(levels, diffs) indexed by (child_scenario, fam_scenario,
    heart_state), or None if the lifetime files are absent."""
    lv, df = DATA_DIR / "lifetime_summary.csv", DATA_DIR / "lifetime_diff.csv"
    if not (lv.exists() and df.exists()):
        return None
    out = []
    for path in (lv, df):
        d = pd.read_csv(path, low_memory=False)
        for c in ("child_iv", "fam_iv", "child_kind"):
            d[c] = d[c].fillna("")
        out.append(d)
    return tuple(out)


US_BIRTHS = 3606400   # CDC/NCHS, Births: Provisional Data for 2025 (VSRR 43)
# What "a case" is for each childhood lever, for the per-case basis.
CASE_WORDS = {
    "mort": ("death before 25", "averts", "a death before 25"),
    "dev_delay": ("case", "prevents", "developmental delay"),
    "int_dis_modsev": ("case", "prevents", "moderate/severe intellectual disability"),
    "anxiety": ("case", "prevents", "anxiety"),
    "depression": ("case", "prevents", "depression"),
    "behavioral_problems": ("case", "prevents", "behavioral problems"),
}


@st.cache_data
def load_ever_share(stamp):
    path = DATA_DIR / "ever_share.csv"
    return pd.read_csv(path) if path.exists() else None


def zero_effect(df_all, iv):
    """True if a childhood lever changes no lifetime outcome for anyone (e.g.
    spec_care once it has no pathway); such levers are hidden."""
    r = df_all[(df_all["child_iv"] == iv) & (df_all["fam_scenario"] == "chd_base")]
    cols = [c for c in ("LE", "WVAS_LIFETIME", "TOTPAY_0_24") if c in r.columns]
    return len(r) > 0 and float(np.nanmax(np.abs(r[cols].to_numpy(dtype=float)))) < 1e-9


def us_births(meta):
    """{heart_state: U.S. births per year} from the NSCH-weighted birth shares."""
    sh = meta.get("birth_share")
    if not sh:
        return None
    return {int(h): US_BIRTHS * v for h, v in sh.items()}


def cases_prevented(civ, child_w, dmean):
    """Share of births whose outcome the childhood lever changes, by
    heart_state: deaths averted before 25 for mortality, otherwise the drop
    in the share ever having the condition (interpolated like everything
    else). None if not available."""
    if civ is None or civ not in CASE_WORDS:
        return None
    unit, verb, what = CASE_WORDS[civ]
    if civ == "mort":
        if "share_reach_25" not in dmean.columns:
            return None
        dp = {int(h): float(v) for h, v in dmean["share_reach_25"].items()}
    else:
        ev = load_ever_share(data_stamp())
        if ev is None:
            return None
        e = ev[ev["condition"] == civ].set_index(["child_scenario", "heart_state"])["ever_share"]
        try:
            base = e.xs("baseline", level="child_scenario")
            scen = sum(w * e.xs(c, level="child_scenario") for c, w in child_w)
        except KeyError:
            return None
        dp = {int(h): float(v) for h, v in (base - scen).items()}
    return {"dp": dp, "unit": unit, "verb": verb, "what": what}


def agg_fmt(v, unit, signed=False):
    """Totals for a birth cohort: dollars in $M/$B, counts and years whole."""
    if pd.isna(v):
        return "n/a"
    sign = ("+" if v >= 0 else "-") if signed else ("-" if v < 0 else "")
    a = abs(v)
    if unit == "usd":
        return "{}${:,.1f}B".format(sign, a / 1e9) if a >= 1e9 else "{}${:,.1f}M".format(sign, a / 1e6)
    return "{}{:,.0f}".format(sign, a)


def life_unit(col):
    for c, _, u in LIFE_PER_BIRTH + LIFE_PER_ADULT:
        if c == col:
            return u
    return "n"


def life_fmt(v, unit, signed=False):
    if pd.isna(v):
        return "n/a"
    if unit == "usd":
        v = float(np.round(v, -2))
        sign = ("+" if v >= 0 else "-") if signed else ("-" if v < 0 else "")
        return "{}${:,.0f}".format(sign, abs(v))
    if unit == "pct":
        v = 100.0 * v
    d = 1 if unit == "pct" else 2
    return ("{:+,.%df}" % d if signed else "{:,.%df}" % d).format(v)


def child_grid(life_df, iv):
    """[(pct, child_scenario)] for one childhood intervention, baseline at 0;
    the cure (100%) is kept out of the grid (see cure_scenario)."""
    r = life_df[(life_df["child_kind"] == "intervention") & (life_df["child_iv"] == iv)]
    r = r[["child_pct", "child_scenario"]].drop_duplicates().sort_values("child_pct")
    return [(0.0, "baseline")] + list(r.itertuples(index=False, name=None))


def cure_scenario(life_df, iv):
    r = life_df[(life_df["child_kind"] == "cure") & (life_df["child_iv"] == iv)]
    return r["child_scenario"].iloc[0] if len(r) else None


def fam_grid(life_df, iv):
    r = life_df[life_df["fam_iv"] == iv][["fam_pct", "fam_scenario"]].drop_duplicates()
    return [(0.0, "chd_base")] + list(r.sort_values("fam_pct").itertuples(index=False, name=None))


def weights(grid, x):
    """[(scenario, weight)] for linear interpolation at x on grid."""
    s0, s1, t = bracket(grid, x)
    return [(s0, 1.0)] if t == 0.0 else [(s0, 1.0 - t), (s1, t)]


def life_at(table, child_w, fam_w, cols):
    """Bilinear combination over the child x FAM corners; rows by heart_state.
    Means are exact (paired differences are linear in the corners); SDs
    are interpolated the same way, an approximation between grid points."""
    idx = table.set_index(LIFE_KEYS)
    acc = None
    for c, wc in child_w:
        for f, wf in fam_w:
            part = idx.xs((c, f), level=("child_scenario", "fam_scenario"))[cols] * (wc * wf)
            acc = part if acc is None else acc + part
    return acc


def life_halfwidth(diff_row_sd, diff_row_bootsd, nsim):
    """95% half-width combining Monte Carlo error of the mean with
    donor-pool (bootstrap) uncertainty, treated as independent."""
    mc = diff_row_sd / np.sqrt(nsim)
    bs = 0.0 if diff_row_bootsd is None else diff_row_bootsd
    return 1.96 * np.sqrt(mc ** 2 + np.square(bs))


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

LOGO = Path(__file__).parent / "usc_schaeffer_logo.png"
_t_col, _l_col = st.columns([4, 1], vertical_alignment="center")
_t_col.title("Congenital heart disease across the life course")
if LOGO.exists():
    _l_col.image(str(LOGO), width="stretch")
_t_col.caption(
    "Microsimulation of children born with and without CHD, by severity, from birth to death. "
    "Values are weighted to NSCH births and averaged over {} Monte Carlo replications.".format(
        meta.get("nsim", "?"))
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
    ("Health care use", ["total_n_inpatient_visits", "total_n_ed_visits",
                         "total_n_op_clinic_visits"]),
    ("Spending", ["total_tot_pay"]),
]
HEADLINE_SECTION = {c: sec for sec, cols in HEADLINE_SECTIONS for c in cols}
headline = [c for _, cols in HEADLINE_SECTIONS for c in cols if c in value_cols]
ordered_cols = headline + [c for c in value_cols if c not in headline]
label_to_col = {cumulative_label(c): c for c in ordered_cols}

interventions = [iv for iv in cum["iv_outcome"].unique() if iv]
iv_label_to_key = {INTERVENTION_LABELS.get(iv, iv): iv for iv in interventions}

LIFE = load_lifetime(data_stamp())
_vp = DATA_DIR / "visit_prices.csv"
USE_PRICES = pd.read_csv(_vp) if _vp.exists() else None
COSTS_RAW = _raw[2]   # with the _comm / _mcaid columns (the payer views drop them)
HAS_USE = (USE_PRICES is not None and COSTS_RAW is not None and surv is not None
           and "n_inpatient_visits_mcaid" in COSTS_RAW.columns)
# Order of the story: what the model is, how heavy the burden is, what could
# change it; the childhood-only detail sits last for analysts.
tab_names = ["About"] + (["The burden of CHD", "What if?"] if LIFE is not None else []) + \
            (["Health care use"] if HAS_USE else []) + ["Details"]
_tabs = dict(zip(tab_names, st.tabs(tab_names)))
tab_about = _tabs["About"]
tab_burden = _tabs.get("The burden of CHD")
tab_life = _tabs.get("What if?")
tab_use = _tabs.get("Health care use")
with _tabs["Details"]:
    st.caption("Childhood detail: outcomes from birth to 18, interventions on them, and profiles by "
               "age through 24.")
    tab_base, tab_iv, tab_age = st.tabs(["Baseline (0-17)", "Interventions (0-17)", "By age (0-24)"])

# ---------------------------------------------------------------------
# About (landing page)
# ---------------------------------------------------------------------
FLOW_SVG = Path(__file__).parent / "model_flow.svg"   # rendered from model_flow.dot


with tab_about:
  about_col, _ = st.columns([3, 1])
  with about_col:
    st.subheader("About this model")
    st.markdown(
        "This explorer shows results from a microsimulation of children born in the United States "
        "with and without congenital heart disease (CHD). It follows each simulated child year by "
        "year from birth, tracking health, health care use, effects on the family, quality of life, "
        "and medical spending, separately for children born with **mild**, **moderate**, or "
        "**severe** CHD and children born **without CHD**.\n\n"
        "The **childhood model** covers birth through age 17, with survival and costs through "
        "age 24. Each child alive at 25 is then matched to adults in the Panel Study of Income "
        "Dynamics (PSID) who were simulated from age 25 to death in the Future Adult Model (FAM), "
        "so results run from birth to death.\n\n"
        "- **The burden of CHD** compares children born with CHD to children born without it, "
        "over a lifetime, per child and for a year of U.S. births.\n"
        "- **What if?** asks how much of that burden could change: if severe CHD carried the risks "
        "of a milder form, or if childhood or adult risks were reduced.\n"
        "- **Health care use** shows what fewer hospitalizations, ED visits or clinic visits "
        "would save.\n"
        "- **Details** has the childhood model on its own: outcomes from birth to 18, "
        "interventions on them, and profiles by age."
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
        "- **Costs.** Each year, the model predicts inpatient admissions, ED visits and outpatient "
        "clinic visits, then insurer payments given that use, from models estimated on commercial "
        "and Medicaid claims (MarketScan) for children with and without CHD. Commercial and "
        "Medicaid values are combined according to each child's chance of being covered by each "
        "at that age (from NSCH).\n"
        "- **Quality of life.** Health status is converted to quality-adjusted life years (QALYs) "
        "using the GRACE utility transform.\n"
        "- **Repetition.** The whole population is simulated many times with different random draws, "
        "and results are averaged across those replications."
    )

    st.markdown("#### The model runs in a fixed order")
    if FLOW_SVG.exists():
        # Inline (not st.image) so the SVG's text uses the page's web font.
        # Collapsed to one line first: Markdown would otherwise read the
        # indented, blank-line-separated SVG source as a code block.
        _svg = re.sub(r"<!--.*?-->", "", FLOW_SVG.read_text(encoding="utf-8"), flags=re.S)
        # The prose column is 3/4 of the page; let the diagram use the full
        # width so its labels stay legible.
        st.markdown('<div style="width:133%">' +
                    " ".join(line.strip() for line in _svg.splitlines() if line.strip()) +
                    "</div>", unsafe_allow_html=True)
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
        "- **Costs** depend on age, severity, sex, and moderate/severe intellectual disability. "
        "Of the condition interventions, only intellectual disability changes costs. Mortality "
        "interventions also do, because more children are alive and receiving care."
    )

    _exp_is = st.expander("What the model is, and what it isn't")
    _exp_is.markdown(
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
        "- **Costs are preliminary.** Medicaid costs do not yet vary by race, and the year of death "
        "is priced like any other year. Insurance coverage after 17 is held at its age 14-17 "
        "pattern.\n"
        "- **Ages 18 to 24** carry mortality and costs forward. Other outcomes stop at 17.\n"
        "- **Uncertainty shown is partial.** In the childhood tabs, intervals reflect random "
        "variation across replications. In the What if? tab they also include uncertainty from the "
        "small pool of PSID adults with childhood heart conditions. Neither includes uncertainty in "
        "the estimated model parameters, so both understate total uncertainty."
    )

    _exp_ichom = st.expander("How the outcomes relate to the ICHOM Standard Set")
    _exp_ichom.markdown(
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
         "Affording essentials; access to care", "Partial"],
        ["Mental health", "Development (mental health diagnoses)",
         "Anxiety; depression; behavioral problems", "Yes"],
    ], columns=["ICHOM domain", "ICHOM outcome", "Model outcomes", "Coverage"])
    _exp_ichom.dataframe(ichom_map, width="stretch", hide_index=True)
    _exp_ichom.markdown(
        "The model also tracks outcomes that ICHOM does not include as patient outcomes but that "
        "matter for costs and planning: **ER and specialist visits, special health care needs, and "
        "medical spending**. ICHOM's adult outcomes (adult quality of life, work productivity, "
        "adult depression and anxiety, cognition, pregnancy) are partly covered in the lifetime "
        "results through the Future Adult Model (work and earnings, severe psychological distress)."
    )
    _exp_ichom.caption("ICHOM Congenital Heart Disease Standard Set, version 1.0, April 2020. "
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
    st.plotly_chart(fig, width="stretch")
    if col.startswith("total_"):
        st.caption(cost_note(base_payer))

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
                              "Change": fmt(vals[2], c, signed=True, extra=1),
                              "95% MC ±": fmt(vals[3], c, extra=1),
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
                line=(dict(color="#222222", dash="dot", width=1.5) if h == 0
                      else dict(color=HS_COLORS[h])),
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
        lo = 100 * eff_s.loc[eff_s["heart_state"].isin(hs_s), ["base", "level"]].min().min()
        fig.update_layout(yaxis_title="Alive at age (%)",
                          yaxis_range=[max(0.0, 5 * np.floor((lo - 2) / 5)), 100.5])
    else:
        fig.update_layout(yaxis_title="Change in % alive (percentage points)")
        fig.add_hline(y=0, line_color="#bbbbbb", line_width=1)
    fig.update_layout(xaxis_title="Age", xaxis_range=[0, 24], template="plotly_white", height=470,
                      margin=dict(t=30))
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

# ---------------------------------------------------------------------
# The burden of CHD (baseline, lifetime): CHD groups vs. no CHD
# ---------------------------------------------------------------------
def tile_fmt(v, unit, signed=False):
    """Headline tiles: one decimal for years and QALYs (tables keep two)."""
    if unit in ("yr", "q") and not pd.isna(v):
        return ("{:+,.1f}" if signed else "{:,.1f}").format(v)
    return life_fmt(v, unit, signed)


# ICHOM Standard Set domains on the burden page. Headline items per domain:
# (source, column, label, unit, higher_is_worse). Source "life" = lifetime
# linked results; "child" = childhood totals, birth to 17.
ICHOM_HEADLINE = [
    ("Overall health", "Health-related quality of life", [
        ("life", "WVAS_LIFETIME", "Lifetime QALYs (GRACE)", "q", False),
        ("life", "WVAS_0_24", "QALYs (GRACE), ages 0-24", "q", False)]),
    ("Physical health", "Survival and development", [
        ("life", "LE", "Life expectancy (years)", "yr", False),
        ("child", "years_dev_delay", "Years with developmental delay, 0-17", "yr", True),
        ("child", "years_int_dis_modsev", "Years with intellectual disability, 0-17", "yr", True)]),
    ("Mental health", "Mental health diagnoses", [
        ("child", "years_anxiety", "Years with anxiety, 0-17", "yr", True),
        ("child", "years_depression", "Years with depression, 0-17", "yr", True),
        ("child", "years_behavioral_problems", "Years with behavioral problems, 0-17", "yr", True)]),
    ("Social health", "School, family and work", [
        ("child", "years_school_absence", "Years with a week+ of school missed, 0-17", "yr", True),
        ("child", "years_stopwork_ind", "Years a parent stopped working, 0-17", "yr", True),
        ("life", "FIRST_EDUC3", "College degree by 25 (of those alive)", "pct", False)]),
]
# Every childhood total on the burden page, by ICHOM domain (expander table).
ICHOM_CHILD = [
    ("Overall health", ["wvas_0_17"]),
    ("Physical health", ["died_before_18", "life_years", "years_dev_delay", "years_int_dis_modsev",
                         "years_spec_care", "years_no_exercise"]),
    ("Mental health", ["years_anxiety", "years_depression", "years_behavioral_problems"]),
    ("Social health", ["years_school_absence", "years_low_grades", "years_stopwork_ind",
                       "years_essentials_difficulty", "years_access_to_care"]),
]
LIFE_DOMAIN = {"LE": "Physical health", "share_reach_25": "Physical health", "YPLL": "Physical health",
               "WVAS_LIFETIME": "Overall health", "WVAS_LIFETIME_DIS": "Overall health", "WVAS_0_24": "Overall health",
               "WVAS_UNDIS": "Overall health", "IEARN_DIS_PB": "Social health", "FIRST_EDUC3": "Social health",
               "FIRST_EDUC4": "Social health", "DICLAIM_UNDIS": "Social health", "K6SEVERE_UNDIS": "Mental health",
               "FIRST_HEARTE": "Physical health", "HEARTE_UNDIS": "Physical health", "STROKE_UNDIS": "Physical health",
               "HIBPE_UNDIS": "Physical health", "DIABE_UNDIS": "Physical health", "LUNGE_UNDIS": "Physical health",
               "CANCRE_UNDIS": "Physical health"}

TILE_HELP = {"TOTMD_DIS": "Average over adults alive at 25, discounted at 3% to age 25. Per child "
                          "born it is lower for severe CHD, because fewer children reach 25."}
# Totals for a year of births must count every birth, so they use the per-birth version.
US_TILE_COL = {"TOTMD_DIS": ("TOTMD_DIS_PB", "Medical spending from 25, discounted")}

BURDEN_TILES = [("LE", "Life expectancy (years)", "yr", False),
                ("WVAS_LIFETIME", "Lifetime QALYs (GRACE)", "q", False),
                ("TOTPAY_0_24", "Insurer payments, ages 0-24", "usd", True),
                # Per adult alive at 25: per child born, fewer severe-CHD children reach 25,
                # which outweighs their higher spending and reads as a saving.
                ("TOTMD_DIS", "Adult medical spending", "usd", True),
                ("IEARN_DIS_PB", "Lifetime earnings, discounted", "usd", False)]

if tab_burden is not None:
    with tab_burden:
        lv_b = LIFE[0]
        st.subheader("The burden of CHD over a lifetime")
        st.markdown(
            "Children born with CHD, by severity, compared with children born without it, from "
            "birth to death. Each value is per child born, so children who die young count what "
            "they lived. The **excess** is the difference from children without CHD.")
        b_cols = [c for c, _, _ in LIFE_PER_BIRTH + LIFE_PER_ADULT if c in lv_b.columns]
        bl = life_at(lv_b, [("baseline", 1.0)], [("chd_base", 1.0)], b_cols)
        births_b = us_births(meta)
        b_basis = st.radio("Show", ["Per child born"] + (["U.S. births per year"] if births_b else []),
                           horizontal=True, key="burden_basis",
                           help="U.S. births per year multiplies the excess per child by the number "
                                "born with each severity in a year.")
        us_mode = b_basis.startswith("U.S.")
        groups = [h for h in CHD_STATES if h in bl.index]
        tiles_b = [t for t in BURDEN_TILES if t[0] in bl.columns]

        if 0 in bl.index:
            if not us_mode:
                tc = st.columns([1.2] + [2] * len(tiles_b))
                tc[0].markdown("<div style='padding-top:1.6rem;font-weight:600;color:{}'>No CHD</div>"
                               .format(HS_COLORS[0]), unsafe_allow_html=True)
                for i, (c, lab, u, _) in enumerate(tiles_b):
                    tc[i + 1].metric(lab, tile_fmt(bl.loc[0, c], u))
            for h in groups:
                tc = st.columns([1.2] + [2] * len(tiles_b))
                tc[0].markdown("<div style='padding-top:1.6rem;font-weight:600;color:{}'>{}</div>"
                               .format(HS_COLORS[h], HS_LABELS[h]), unsafe_allow_html=True)
                for i, (c, lab, u, cost) in enumerate(tiles_b):
                    ex = bl.loc[h, c] - bl.loc[0, c]
                    if us_mode:
                        c2, lab2 = US_TILE_COL.get(c, (c, "Life-years" if c == "LE" else lab))
                        ex = bl.loc[h, c2] - bl.loc[0, c2]
                        tc[i + 1].metric(lab2 + ", excess", agg_fmt(ex * births_b[h], u, signed=True))
                    else:
                        tc[i + 1].metric(lab, tile_fmt(bl.loc[h, c], u), tile_fmt(ex, u, signed=True),
                                         delta_color="inverse" if cost else "normal",
                                         help=TILE_HELP.get(c))
            if us_mode:
                tot = st.columns([1.2] + [2] * len(tiles_b))
                tot[0].markdown("<div style='padding-top:1.6rem;font-weight:600'>All CHD</div>",
                                unsafe_allow_html=True)
                for i, (c, lab, u, _) in enumerate(tiles_b):
                    c2, lab2 = US_TILE_COL.get(c, (c, "Life-years" if c == "LE" else lab))
                    tot[i + 1].metric(lab2 + ", excess", agg_fmt(
                        sum((bl.loc[h, c2] - bl.loc[0, c2]) * births_b[h] for h in groups), u, signed=True))
                st.markdown("**U.S. births per year** (2025): " + " · ".join(
                    "{} {:,.0f}".format(HS_LABELS[h].lower(), births_b[h]) for h in groups) +
                    ", from {:,.0f} total births and the model's NSCH-weighted severity shares.".format(
                        US_BIRTHS))
            else:
                st.caption("Small figures under each value: the excess over children without CHD.")

            # ---- ICHOM domains: one group at a time, a few outcomes per domain
            st.markdown("#### What CHD changes, by ICHOM domain")
            st.caption("The outcome domains of the ICHOM Standard Set for congenital heart disease. "
                       "See About for how the model's measures map to them.")
            ch_b = cum[cum["scenario"] == "baseline"].groupby("heart_state").mean(numeric_only=True)
            w_b = None
            if births_b:
                tot_b = sum(births_b[h] for h in groups)
                w_b = {h: births_b[h] / tot_b for h in groups}
            g_opts = [HS_LABELS[h] for h in groups] + (["All CHD"] if w_b else [])
            g_sel = st.radio("Group", g_opts, index=g_opts.index("Severe") if "Severe" in g_opts else 0,
                             horizontal=True, key="burden_ichom_group")

            def val(src, c, h):
                t = bl if src == "life" else ch_b
                if c not in t.columns:
                    return np.nan
                if h == "all":
                    return sum(w_b[g] * t.loc[g, c] for g in groups)
                return t.loc[h, c]

            hsel = "all" if g_sel == "All CHD" else [h for h in groups if HS_LABELS[h] == g_sel][0]

            def one(v, u, signed=False):
                if pd.isna(v):
                    return "n/a"
                if u == "pct":
                    return ("{:+.0f} pp" if signed else "{:.0f}%").format(100 * v)
                return ("{:+,.1f}" if signed else "{:,.1f}").format(v)

            dcols = st.columns(len(ICHOM_HEADLINE))
            for (dom, what, items), dc in zip(ICHOM_HEADLINE, dcols):
                box = dc.container(border=True)
                box.markdown("**{}**<br><span style='font-size:0.8rem;color:#6b7280'>{}</span>".format(
                    dom, what), unsafe_allow_html=True)
                for src, c, lab, u, worse in items:
                    v0, v = val(src, c, 0), val(src, c, hsel)
                    ex = v - v0
                    bad = (ex > 0) == worse
                    colr = "#b3261e" if bad else "#2e7d5b"
                    if us_mode:
                        n = births_b[hsel] if hsel != "all" else sum(births_b[g] for g in groups)
                        if src == "life" and u == "pct":
                            # a share of adults alive at 25: count it among births
                            ex = (v * val("life", "share_reach_25", hsel) -
                                  v0 * val("life", "share_reach_25", 0))
                        lab = {"LE": "Life-years", "FIRST_EDUC3": "College graduates by 25"}.get(c, lab)
                        main = agg_fmt(ex * n, "n", signed=True) + (" people" if u == "pct" else
                                                                    " years" if u == "yr" else " QALYs")
                        sub = "excess, one year of U.S. births"
                    else:
                        main = one(v, u)
                        sub = "<span style='color:{}'>{}</span> vs. no CHD ({})".format(
                            colr, one(ex, u, signed=True), one(v0, u))
                    box.markdown(
                        "<div style='font-size:0.8rem;color:#374151;margin-top:0.4rem'>{}</div>"
                        "<div style='font-size:1.35rem;font-weight:600;line-height:1.3'>{}</div>"
                        "<div style='font-size:0.8rem;color:#6b7280'>{}</div>".format(lab, main, sub),
                        unsafe_allow_html=True)
            st.caption("Per child born. Childhood years are counted from birth to 17, including "
                       "children who die young. \"All CHD\" averages the three severities by their "
                       "share of births. ICHOM outcomes the model does not cover: growth, heart "
                       "failure, arrhythmias, and life satisfaction.")

            # ---- Every number, collapsed
            def burden_table(spec, src="life"):
                t = bl if src == "life" else ch_b
                rows_b = []
                for dom, c, lab, u in spec:
                    if c not in t.columns:
                        continue
                    if us_mode:
                        uu = "n" if u == "pct" else u
                        lab2 = lab.replace(" (%)", " (children)") if u == "pct" else lab
                        if c == "LE":
                            lab2 = "Life-years"
                        row = {"Domain": dom, "Outcome": lab2}
                        for h in groups:
                            row[HS_LABELS[h] + ": excess"] = agg_fmt(
                                (t.loc[h, c] - t.loc[0, c]) * births_b[h], uu, signed=True)
                        row["All CHD: excess"] = agg_fmt(sum(
                            (t.loc[h, c] - t.loc[0, c]) * births_b[h] for h in groups), uu, signed=True)
                    else:
                        row = {"Domain": dom, "Outcome": lab, "No CHD": life_fmt(t.loc[0, c], u)}
                        for h in groups:
                            row[HS_LABELS[h]] = life_fmt(t.loc[h, c], u)
                            row[HS_LABELS[h] + ": excess"] = life_fmt(t.loc[h, c] - t.loc[0, c], u,
                                                                      signed=True)
                    rows_b.append(row)
                return pd.DataFrame(rows_b)

            def show(tbl):
                st.dataframe(tbl, width="stretch", hide_index=True, height=35 * (len(tbl) + 1) + 3)

            st.markdown("#### All outcomes")
            child_spec = [(dom, c, cumulative_label(c), "pct" if c == "died_before_18" else "yr")
                          for dom, cs in ICHOM_CHILD for c in cs]
            # died_before_18 is a share; life_fmt shows "pct" units as percentages
            with st.expander("Childhood, birth to 17, by ICHOM domain"):
                show(burden_table(child_spec, "child"))
                st.caption("Years with each condition from birth to 17, per child born. Excess shares "
                           "are in percentage points.")
            per = "U.S. births per year, excess over no CHD" if us_mode else "per child born"
            with st.expander("Lifetime, " + per):
                show(burden_table([(LIFE_DOMAIN.get(c, "Costs"), c, l, u) for c, l, u in LIFE_PER_BIRTH]))
            if not us_mode:
                with st.expander("Lifetime, per adult alive at 25"):
                    show(burden_table([(LIFE_DOMAIN.get(c, "Costs"), c, l, u)
                                       for c, l, u in LIFE_PER_ADULT]))
            st.caption(
                "Dollars and discounted QALYs are discounted at 3% to birth, except medical spending "
                "per adult, which is discounted to age 25. Childhood insurer payments come from claims "
                "and adult medical spending from FAM, so they are different measures. \"Costs\" are not "
                "an ICHOM domain. Children without CHD continue after 25 as PSID adults with no "
                "childhood heart problem. Point estimates; the What if? tab shows intervals for "
                "changes.")

            st.markdown("#### Survival from birth")
            s_cols_b = sorted([c for c in lv_b.columns if c.startswith("S_") and not c.endswith("sd")])
            if s_cols_b:
                sb_b = life_at(lv_b, [("baseline", 1.0)], [("chd_base", 1.0)], s_cols_b)
                ages_b = [int(c[2:]) for c in s_cols_b]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ages_b, y=np.round(100 * sb_b.loc[0].values, 1),
                                         mode="lines", name="No CHD",
                                         line=dict(color="#222222", dash="dot", width=1.5),
                                         hovertemplate="No CHD, age %{x}: %{y:.1f}%<extra></extra>"))
                for h in groups:
                    fig.add_trace(go.Scatter(
                        x=ages_b, y=np.round(100 * sb_b.loc[h].values, 1), mode="lines",
                        name=HS_LABELS[h], line=dict(color=HS_COLORS[h]),
                        hovertemplate=HS_LABELS[h] + ", age %{x}: %{y:.1f}%<extra></extra>"))
                fig.update_layout(xaxis_title="Age", yaxis_title="Alive at age (%)",
                                  yaxis_range=[0, 101], xaxis_range=[0, 100],
                                  template="plotly_white", height=450, margin=dict(t=30))
                st.plotly_chart(fig, width="stretch")
                st.caption("Share of each birth cohort alive at each age. Birth to 24: childhood "
                           "model; 25 on: FAM.")

# ---------------------------------------------------------------------
# Lifetime
# ---------------------------------------------------------------------
if tab_life is not None:
    with tab_life:
        lv_all, df_all = LIFE
        nsim = meta.get("nsim", 1)
        has_boot = any(c.endswith("_bootsd") for c in df_all.columns)
        st.subheader("What if? Changing the risks that drive the burden")
        st.markdown("Start with the bounds: how much would change if severe CHD carried the risks of a "
                    "milder form. Then try realistic reductions in single childhood or adult risks, "
                    "alone or combined. Every result is compared with the same children's baseline.")

        child_ivs = [iv for iv in lv_all.loc[lv_all["child_kind"] == "intervention", "child_iv"].unique()
                     if not zero_effect(df_all, iv)]
        child_lab = {INTERVENTION_LABELS.get(iv, iv): iv for iv in child_ivs}
        fam_ivs = [iv for iv in lv_all["fam_iv"].unique() if iv]
        fam_lab = {FAM_IV_LABELS.get(iv, iv): iv for iv in fam_ivs}
        cfs = [k for k in CF_LABELS if k in set(lv_all.loc[lv_all["child_kind"] == "counterfactual",
                                                           "child_iv"])]

        l1, l2 = st.columns(2)
        with l1:
            st.markdown("**Childhood (birth to 24)**")
            kind = st.radio("Type", ["Severity counterfactual", "Intervention"], horizontal=True,
                            key="life_kind",
                            help="A severity counterfactual simulates children born with severe CHD "
                                 "as if they had a milder form (or only its mortality), and applies "
                                 "to the severe group only.") if cfs else "Intervention"
            if kind == "Intervention":
                c_opts = ["(none)"] + list(child_lab)
                c_label = st.selectbox("Intervention", c_opts, key="life_civ",
                                       index=c_opts.index("Mortality") if "Mortality" in c_opts else 0)
                if c_label == "(none)":
                    child_w, c_desc = [("baseline", 1.0)], "no childhood intervention"
                else:
                    civ = child_lab[c_label]
                    cg = child_grid(lv_all, civ)
                    cure = cure_scenario(lv_all, civ)
                    use_cure = cure is not None and st.checkbox(
                        "Eliminate entirely (100%, a bound)", key="life_cure",
                        help="Sets the risk to zero at every age. An upper bound, not a realistic "
                             "intervention; not interpolated with the grid below.")
                    if use_cure:
                        child_w, c_desc = [(cure, 1.0)], "{} eliminated".format(c_label.lower())
                    else:
                        cx = intensity_input(cg, key="life_cx", default=30.0)
                        child_w = weights(cg, cx)
                        c_desc = "{} risk reduced {:g}%".format(c_label.lower(), cx)
            else:
                cf = st.selectbox("Counterfactual", cfs, format_func=lambda k: CF_LABELS[k],
                                  key="life_cf", index=cfs.index("as_none") if "as_none" in cfs else 0)
                child_w = [("sev_{}_baseline".format(cf), 1.0)]
                c_desc = CF_LABELS[cf].lower()
        with l2:
            st.markdown("**Adulthood (25 and older)**")
            f_label = st.selectbox("Intervention", ["(none)"] + list(fam_lab), key="life_fiv",
                                   help="Applied to adults with a childhood heart condition. Adults "
                                        "without CHD are never treated.")
            if f_label == "(none)":
                fam_w, f_desc = [("chd_base", 1.0)], "no adult intervention"
            else:
                fg = fam_grid(lv_all, fam_lab[f_label])
                fx = intensity_input(fg, key="life_fx")
                fam_w = weights(fg, fx)
                f_desc = "{} risk reduced {:g}%".format(f_label.lower(), fx)

        # Counterfactuals change the severe group only, so show it alone by default.
        hs_l = (group_picker("life_groups", CHD_STATES) if kind == "Intervention"
                else group_picker("life_groups_cf", [4]))
        st.info("**Scenario:** {}; {}.".format(c_desc, f_desc))

        outcomes = [(c, l, u, "birth") for c, l, u in LIFE_PER_BIRTH if c in df_all.columns] + \
                   [(c, l, u, "adult") for c, l, u in LIFE_PER_ADULT if c in df_all.columns]
        cols = [c for c, _, _, _ in outcomes]
        base_lv = life_at(lv_all, [("baseline", 1.0)], [("chd_base", 1.0)], cols)
        scen_lv = life_at(lv_all, child_w, fam_w, cols)
        dmean = life_at(df_all, child_w, fam_w, cols)
        dsd = life_at(df_all, child_w, fam_w, [c + "_sd" for c in cols])
        dbs = life_at(df_all, child_w, fam_w, [c + "_bootsd" for c in cols]) if has_boot else None

        # Basis: per child born (average effect), per case prevented (effect on
        # the children the intervention reaches) or U.S. births per year.
        civ_sel = civ if (kind == "Intervention" and c_label != "(none)") else None
        cases = cases_prevented(civ_sel, child_w, dmean) if f_label == "(none)" else None
        births = us_births(meta)
        bases = ["Per child born"]
        if cases is not None:
            bases.append("Per death averted before 25" if civ_sel == "mort" else "Per case prevented")
        if births is not None:
            bases.append("U.S. births per year")
        b1, b2 = st.columns([3, 2])
        basis = b1.radio("Show results", bases, horizontal=True, key="life_basis",
                         help="Per child born averages over every child in the group, including those "
                              "the intervention never reaches. Per case prevented divides that by "
                              "the share of children whose outcome the intervention changes. U.S. "
                              "births per year multiplies it by the number born in the group each year.")
        wtp = b2.select_slider("Value of one QALY", options=[50000, 100000, 150000, 200000],
                               value=100000, format_func=lambda v: "${:,.0f}".format(v),
                               key="life_wtp")
        mode = "birth" if basis == "Per child born" else ("us" if basis.startswith("U.S.") else "case")

        def factor(h):
            if mode == "case":
                d = cases["dp"].get(h, np.nan)
                return 1.0 / d if d and d > 1e-9 else np.nan
            if mode == "us":
                return births.get(h, np.nan)
            return 1.0

        if mode == "case":
            what = "deaths before 25 averted" if civ_sel == "mort" else "cases of {} prevented".format(
                cases["what"])
            st.markdown("**{} per 100 children born:** ".format(what[:1].upper() + what[1:]) +
                        " · ".join("{} {:.1f}".format(HS_LABELS[h].lower(), 100 * cases["dp"][h])
                                   for h in hs_l if cases["dp"].get(h, 0) > 1e-9) +
                        ". Results below are the change per child born divided by these shares.")
        elif mode == "us":
            st.markdown("**U.S. births per year** (2025): " + " · ".join(
                "{} {:,.0f}".format(HS_LABELS[h].lower(), births[h]) for h in hs_l if h in births) +
                ", from {:,.0f} total births and the model's NSCH-weighted severity shares.".format(
                    US_BIRTHS))

        fm_ = agg_fmt if mode == "us" else life_fmt
        health, costs_l = "WVAS_LIFETIME_DIS", ["TOTPAY_0_24_DIS", "TOTMD_DIS_PB"]
        has_nmb = health in dmean.columns and all(c in dmean.columns for c in costs_l)

        def nmb_of(h):
            f = factor(h)
            dq, dc = dmean.loc[h, health] * f, sum(dmean.loc[h, c] for c in costs_l) * f
            return dq, dc, wtp * dq - dc

        # Headline tiles, one row per group
        tiles = [(c, lab, u) for c, lab, u in [
            ("LE", "Life expectancy (years)", "yr"),
            ("WVAS_LIFETIME", "Lifetime QALYs (GRACE)", "q"),
            ("TOTPAY_0_24", "Childhood insurer payments", "usd")] if c in dmean.columns]
        for h in hs_l:
            if h not in dmean.index:
                continue
            f = factor(h)
            tc = st.columns([1.2] + [2] * (len(tiles) + int(has_nmb)))
            tc[0].markdown("<div style='padding-top:1.6rem;font-weight:600;color:{}'>{}</div>".format(
                HS_COLORS[h], HS_LABELS[h]), unsafe_allow_html=True)
            for i, (c, lab, u) in enumerate(tiles):
                inv = "inverse" if u == "usd" else "normal"
                if mode == "birth":
                    tc[i + 1].metric(lab, life_fmt(scen_lv.loc[h, c], u),
                                     life_fmt(dmean.loc[h, c], u, signed=True), delta_color=inv)
                else:
                    lab2 = "Life-years" if c == "LE" else lab
                    tc[i + 1].metric(lab2 + (" per case" if mode == "case" else ", U.S. total"),
                                     fm_(dmean.loc[h, c] * f, u, signed=True))
            if has_nmb:
                tc[-1].metric("Net monetary benefit", fm_(nmb_of(h)[2], "usd", signed=True),
                              help="Value of the change in discounted lifetime QALYs at the QALY "
                                   "value above, minus the change in discounted medical spending.")

        # Full results: one row per outcome, columns by group
        def table(per):
            rows_t = []
            for c, lab, u, p_ in outcomes:
                if p_ != per or (mode == "case" and c == "share_reach_25"):
                    continue
                uu = "n" if (mode == "us" and u == "pct") else u
                lab2 = lab.replace(" (%)", " (children)") if (mode == "us" and u == "pct") else lab
                if mode != "birth" and c == "LE":
                    lab2 = "Life-years"
                row = {"Outcome": lab2}
                for h in hs_l:
                    if h not in dmean.index:
                        continue
                    f, g = factor(h), HS_LABELS[h]
                    hw = life_halfwidth(dsd.loc[h, c + "_sd"],
                                        None if dbs is None else dbs.loc[h, c + "_bootsd"], nsim)
                    if mode != "case":
                        row[g + ": baseline"] = fm_(base_lv.loc[h, c] * f, uu)
                        row[g + ": scenario"] = fm_(scen_lv.loc[h, c] * f, uu)
                    row[g + ": change"] = "{} ± {}".format(fm_(dmean.loc[h, c] * f, uu, signed=True),
                                                           fm_(hw * abs(f), uu))
                rows_t.append(row)
            return pd.DataFrame(rows_t)

        st.markdown("#### All outcomes")
        st.markdown("**Per child born**" if mode == "birth" else
                    ("**Per case prevented**" if mode == "case" else "**U.S. births per year**"))
        t_b = table("birth")
        st.dataframe(t_b, width="stretch", hide_index=True, height=35 * (len(t_b) + 1) + 3)
        if mode == "birth":
            st.markdown("**Per adult alive at 25**")
            t_a = table("adult")
            st.dataframe(t_a, width="stretch", hide_index=True, height=35 * (len(t_a) + 1) + 3)
        basis_note = {
            "birth": "Per child born counts every birth, including children who die before 25. Per "
                     "adult is the average over those alive at 25, so it can move in either direction "
                     "when an intervention changes who survives. ",
            "case": "Per case prevented divides the change per child born by the share of births "
                    "whose outcome the intervention changes. Children it never reaches are identical "
                    "in both runs (they share every random draw), so this is the average effect on "
                    "the children it does reach. Effects after 25 run only through childhood "
                    "self-rated health, so for conditions they are lower bounds. ",
            "us": "Totals for one year of U.S. births ({:,.0f} in 2025, CDC/NCHS provisional), split "
                  "by severity using the NSCH-weighted shares in the model. ".format(US_BIRTHS)}[mode]
        st.caption(
            basis_note + "Changes are shown ± a 95% interval. "
            "Dollars and discounted QALYs are discounted at 3% to birth, except medical spending per "
            "adult, which is discounted to age 25. Childhood insurer payments come from claims and "
            "adult medical spending from FAM, so they are different measures. " +
            ("Intervals combine Monte Carlo error and uncertainty from the small pool of PSID adults "
             "with childhood heart conditions (bootstrap)." if has_boot else
             "Intervals are Monte Carlo error only; they omit uncertainty from the small pool of "
             "PSID adults with childhood heart conditions, which is larger.") +
            " These intervals do not include uncertainty in the estimated model parameters. "
            "Childhood interventions were simulated at 10, 20 and 30% (plus elimination); adult "
            "interventions at 10 to 70%.")

        # Net monetary benefit
        if has_nmb:
            st.markdown("#### Net monetary benefit")
            nmb = []
            for h in hs_l:
                if h not in dmean.index:
                    continue
                dq, dc, nb = nmb_of(h)
                nmb.append({"Group": HS_LABELS[h], "Change in discounted QALYs": fm_(dq, "q", True),
                            "Change in discounted spending": fm_(dc, "usd", True),
                            "Net monetary benefit": fm_(nb, "usd", True)})
            st.dataframe(pd.DataFrame(nmb), width="stretch", hide_index=True)
            st.caption("{}: the value of the QALY change at ${:,.0f} per QALY, minus the change in "
                       "medical spending (childhood insurer payments plus adult medical spending), "
                       "both discounted at 3% to birth. Program costs are not included, so this is "
                       "what an intervention could cost, on that basis, and still break even. Point "
                       "estimates only.".format(basis, wtp))

        # Survival
        st.markdown("#### Survival from birth")
        s_cols = sorted([c for c in lv_all.columns if c.startswith("S_") and not c.endswith("sd")])
        if s_cols:
            sb = life_at(lv_all, [("baseline", 1.0)], [("chd_base", 1.0)], s_cols)
            ss = life_at(lv_all, child_w, fam_w, s_cols)
            ages = [int(c[2:]) for c in s_cols]
            fig = go.Figure()
            if 0 in sb.index:
                fig.add_trace(go.Scatter(x=ages, y=np.round(100 * sb.loc[0].values, 1), mode="lines",
                                         name="No CHD (reference)",
                                         line=dict(color="#222222", dash="dot", width=1.5),
                                         hovertemplate="No CHD, age %{x}: %{y:.1f}%<extra></extra>"))
            for h in hs_l:
                if h not in sb.index or h == 0:
                    continue
                fig.add_trace(go.Scatter(x=ages, y=np.round(100 * sb.loc[h].values, 1), mode="lines",
                                         name=HS_LABELS[h], line=dict(color=HS_COLORS[h]),
                                         hovertemplate=HS_LABELS[h] + ", age %{x}: %{y:.1f}%<extra></extra>"))
                if np.abs(sb.loc[h].values - ss.loc[h].values).max() > 0.002:
                    fig.add_trace(go.Scatter(
                        x=ages, y=np.round(100 * ss.loc[h].values, 1), mode="lines",
                        name=HS_LABELS[h] + " (scenario)", line=dict(color=HS_COLORS[h], dash="dash"),
                        hovertemplate=HS_LABELS[h] + " scenario, age %{x}: %{y:.1f}%<extra></extra>"))
            fig.update_layout(xaxis_title="Age", yaxis_title="Alive at age (%)", yaxis_range=[0, 101],
                              xaxis_range=[0, 100], template="plotly_white", height=450,
                              margin=dict(t=30))
            st.plotly_chart(fig, width="stretch")

        # Combination surface
        if kind == "Intervention" and c_label != "(none)" and f_label != "(none)" and not use_cure:
            st.markdown("#### Every simulated combination")
            k1, k2 = st.columns(2)
            surf_out = k1.selectbox("Outcome", [lab for _, lab, _, _ in outcomes], key="life_surf_out")
            sc_col = [c for c, lab, _, _ in outcomes if lab == surf_out][0]
            sc_u = life_unit(sc_col)
            sh = k2.selectbox("Group", [HS_LABELS[h] for h in hs_l], key="life_surf_g")
            sh = [h for h in hs_l if HS_LABELS[h] == sh][0]
            z = [[life_at(df_all, [(cs_, 1.0)], [(fs_, 1.0)], [sc_col]).loc[sh, sc_col]
                  * (100 if sc_u == "pct" else 1) for cs_ in [s_ for _, s_ in cg]]
                 for fs_ in [s_ for _, s_ in fg]]
            fig = go.Figure(go.Heatmap(
                z=z, x=["{:g}%".format(p) for p, _ in cg], y=["{:g}%".format(p) for p, _ in fg],
                colorscale="Blues", colorbar=dict(title="Change"),
                hovertemplate="childhood %{x}, adult %{y}: %{z:,.2f}<extra></extra>"))
            fig.update_layout(xaxis_title="Childhood: reduction in {} risk".format(c_label.lower()),
                              yaxis_title="Adult: reduction in {} risk".format(f_label.lower()),
                              template="plotly_white", height=420, margin=dict(t=30))
            st.plotly_chart(fig, width="stretch")
            st.caption("Change from baseline in {} for {} CHD, at each simulated pair of "
                       "intensities.".format(surf_out.lower(), HS_LABELS[sh].lower()))

# ---------------------------------------------------------------------
# Health care use: savings from averted visits (0-24)
# ---------------------------------------------------------------------
VISIT_LABELS = {"n_inpatient_visits": ("Hospitalizations", "hospitalization"),
                "n_ed_visits": ("ED visits", "ED visit"),
                "n_op_clinic_visits": ("Outpatient clinic visits", "clinic visit")}

if tab_use is not None:
    with tab_use:
        st.subheader("What averting hospital, ED or clinic visits would save, birth to 24")
        st.markdown(
            "Each year, the model predicts a child's hospitalizations, ED visits and outpatient clinic "
            "visits, then insurer payments given that use. Payments rise by a fixed amount per visit "
            "(the claims models are linear), so the savings from fewer visits follow directly: no "
            "extra simulation is needed.")
        u1, u2 = st.columns(2)
        v_lab = u1.selectbox("Type of visit", [VISIT_LABELS[v][0] for v in VISIT_LABELS], key="use_v")
        vcol = [v for v in VISIT_LABELS if VISIT_LABELS[v][0] == v_lab][0]
        red = u2.slider("Reduction in visits (%)", 0, 50, 10, step=1, format="%d%%", key="use_red")
        hs_u = group_picker("use_groups", CHD_STATES)

        base_c = COSTS_RAW[COSTS_RAW["scenario"] == "baseline"]
        sv = surv[surv["scenario"] == "baseline"].groupby(["heart_state", "age"])["survival"].mean()
        price = USE_PRICES.set_index(["payer", "visit"])["price"]
        b_comm, b_mcaid = price[("comm", vcol)], price[("mcaid", vcol)]
        rows_u, frac = [], red / 100.0
        for h in hs_u:
            d = base_c[base_c["heart_state"] == h].set_index("age").sort_index()
            d = d[d.index <= 24]
            if d.empty:
                continue
            # payer weight at each age, recovered from the blended and payer-specific totals
            p = ((d["tot_pay_comm"] - d["tot_pay"]) /
                 (d["tot_pay_comm"] - d["tot_pay_mcaid"])).clip(0, 1).fillna(0)
            s_a = sv.loc[h].reindex(d.index).ffill().to_numpy()
            disc = (1.03) ** -d.index.to_numpy(dtype=float)
            v_c, v_m = d[vcol + "_comm"].to_numpy(), d[vcol + "_mcaid"].to_numpy()
            visits = s_a * ((1 - p) * v_c + p * v_m)
            sav = frac * s_a * ((1 - p) * v_c * b_comm + p * v_m * b_mcaid)
            n_vis, n_sav, n_sav_d = visits.sum(), sav.sum(), (sav * disc).sum()
            per_visit = n_sav / (frac * n_vis) if frac > 0 and n_vis > 0 else np.nan
            births_h = (us_births(meta) or {}).get(h, np.nan)
            rows_u.append({"Group": HS_LABELS[h],
                           "{} per child, 0-24".format(v_lab): "{:,.1f}".format(n_vis),
                           "Averted per child": "{:,.2f}".format(frac * n_vis),
                           "Savings per averted {}".format(VISIT_LABELS[vcol][1]): life_fmt(per_visit, "usd"),
                           "Savings per child born": life_fmt(n_sav, "usd"),
                           "Discounted": life_fmt(n_sav_d, "usd"),
                           "Per year of U.S. births": agg_fmt(n_sav * births_h, "usd")})
        if rows_u:
            st.dataframe(pd.DataFrame(rows_u), width="stretch", hide_index=True)
        st.markdown("**Insurer payment per {} in the claims models**".format(VISIT_LABELS[vcol][1]))
        pc1, pc2, _ = st.columns([1, 1, 2])
        pc1.metric("Commercial", life_fmt(b_comm, "usd"))
        pc2.metric("Medicaid", life_fmt(b_mcaid, "usd"))
        st.caption(
            "Savings = the reduction × the visits each child would have had at each age (counting only "
            "children still alive) × the insurer payment per visit, weighted by the chance the child is "
            "covered by Medicaid or commercial insurance at that age. Discounted at 3% to birth. "
            "The payment per visit is the average for children in the claims data, including "
            "surgical and other planned admissions; the visits a program could realistically prevent "
            "may cost less, so these are upper estimates for avoidable-visit programs. In the "
            "Medicaid model, an ED visit adds essentially nothing to total payments once "
            "hospitalizations and clinic visits are accounted for. Visits do not feed back into "
            "health in the model, so this changes costs only, through age 24.")

st.divider()
st.caption("Data generated {}, revision {}.".format(
    meta.get("generated", "?")[:10], meta.get("revision", "?")))
