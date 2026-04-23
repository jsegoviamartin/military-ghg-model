import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import zipfile
import io
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ============================================================
# FILES
# ============================================================
sim_file = "generated_data_2025-2050.csv"
ar6_zip_file = "1668008312256-AR6_Scenarios_Database_World_v1.1.csv.zip"

# ============================================================
# SETTINGS
# ============================================================
category_alpha = 0.50

# Choose which AR6 climate categories to display
SELECTED_CATEGORIES = ["C1", "C4", "C6", "C8"]

# Single combined legend
LEGEND_NCOLS = 2
LEGEND_X = 0.01
LEGEND_Y = 0.99

# ============================================================
# LOAD YOUR SIMULATED SCENARIOS
# ============================================================
df = pd.read_csv(sim_file)

required_cols = {
    "year",
    "scenario",
    "E_world_Gt",
    "g_world",
    "d_mil",
    "d_rest",
    "epsilon",
    "s_mil_2025_assumed",
}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(
        f"This script needs the full time-series simulation file. Missing columns: {missing}"
    )

# Clean scenario names
df["scenario_short"] = df["scenario"].astype(str).str.replace(
    r" \(baseline=.*\)", "", regex=True
)

scenario_code_map = {
    "Baseline": "S0",
    "NATO→3.5%, nonNATO holds": "S1",
    "NATO→3.5%, nonNATO→3.5%": "S2",
    "NATO→5%, nonNATO→3.5%": "S3",
}
df["S_code"] = df["scenario_short"].map(scenario_code_map)

if df["S_code"].isna().any():
    bad = sorted(df.loc[df["S_code"].isna(), "scenario_short"].dropna().unique())
    raise ValueError(f"Unrecognised scenario names in simulation file: {bad}")

# Envelope of all simulated pathways
envelope = (
    df.groupby("year", as_index=False)
      .agg(ymin=("E_world_Gt", "min"),
           ymax=("E_world_Gt", "max"))
)

# ============================================================
# REPRESENTATIVE SIMULATED PATHWAYS
# ============================================================
pathways = [

    {
        "scenario_short": "NATO→5%, nonNATO→3.5%",
        "baseline": 0.070,
        "g_world": 0.04,
        "d_mil": 0.00,
        "d_rest": -0.01,
        "epsilon": 0.02,
        "lw": 3.0,
        "ls": "-"
    },

    {
        "scenario_short": "NATO→5%, nonNATO→3.5%",
        "baseline": 0.033,
        "g_world": 0.04,
        "d_mil": -0.01,
        "d_rest": -0.01,
        "epsilon": 0.02,
        "lw": 3.0,
        "ls": "-"
    },

    {
        "scenario_short": "Baseline",
        "baseline": 0.055,
        "g_world": 0.04,
        "d_mil": -0.01,
        "d_rest": -0.01,
        "epsilon": 0.000,
        "lw": 2.9,
        "ls": "-"
    },

    {
        "scenario_short": "Baseline",
        "baseline": 0.055,
        "g_world": 0.03,
        "d_mil": -0.01,
        "d_rest": -0.01,
        "epsilon": 0.015,
        "lw": 2.8,
        "ls": "-"
    },
    {
        "scenario_short": "NATO→3.5%, nonNATO holds",
        "baseline": 0.055,
        "g_world": 0.03,
        "d_mil": -0.01,
        "d_rest": -0.01,
        "epsilon": 0.015,
        "lw": 2.8,
        "ls": "-"
    },
    {
        "scenario_short": "NATO→3.5%, nonNATO→3.5%",
        "baseline": 0.055,
        "g_world": 0.03,
        "d_mil": -0.01,
        "d_rest": -0.01,
        "epsilon": 0.015,
        "lw": 2.8,
        "ls": "-"
    },
    {
        "scenario_short": "NATO→5%, nonNATO→3.5%",
        "baseline": 0.055,
        "g_world": 0.03,
        "d_mil": -0.01,
        "d_rest": -0.01,
        "epsilon": 0.015,
        "lw": 2.8,
        "ls": "-"
    },

    {
        "scenario_short": "Baseline",
        "baseline": 0.055,
        "g_world": 0.02,
        "d_mil": -0.01,
        "d_rest": -0.01,
        "epsilon": 0.015,
        "lw": 2.8,
        "ls": "-"
    },
    {
        "scenario_short": "NATO→5%, nonNATO→3.5%",
        "baseline": 0.055,
        "g_world": 0.02,
        "d_mil": -0.01,
        "d_rest": -0.01,
        "epsilon": 0.015,
        "lw": 2.8,
        "ls": "-"
    },

    {
        "scenario_short": "Baseline",
        "baseline": 0.033,
        "g_world": 0.03,
        "d_mil": -0.03,
        "d_rest": -0.05,
        "epsilon": 0.00,
        "lw": 2.9,
        "ls": "-"
    },
    {
        "scenario_short": "Baseline",
        "baseline": 0.055,
        "g_world": 0.02,
        "d_mil": -0.03,
        "d_rest": -0.05,
        "epsilon": 0.00,
        "lw": 2.9,
        "ls": "-"
    },
    {
        "scenario_short": "Baseline",
        "baseline": 0.033,
        "g_world": 0.02,
        "d_mil": -0.05,
        "d_rest": -0.05,
        "epsilon": 0.00,
        "lw": 2.9,
        "ls": "-"
    },
    {
        "scenario_short": "Baseline",
        "baseline": 0.055,
        "g_world": 0.02,
        "d_mil": -0.07,
        "d_rest": -0.07,
        "epsilon": 0.00,
        "lw": 2.9,
        "ls": "-"
    },
    {
        "scenario_short": "Baseline",
        "baseline": 0.033,
        "g_world": 0.01,
        "d_mil": -0.07,
        "d_rest": -0.07,
        "epsilon": 0.00,
        "lw": 2.9,
        "ls": "-"
    },

    {
        "scenario_short": "NATO→5%, nonNATO→3.5%",
        "baseline": 0.055,
        "g_world": 0.02,
        "d_mil": -0.01,
        "d_rest": -0.05,
        "epsilon": 0.02,
        "lw": 2.9,
        "ls": "-"
    },

    {
        "scenario_short": "Baseline",
        "baseline": 0.055,
        "g_world": 0.03,
        "d_mil": -0.03,
        "d_rest": -0.03,
        "epsilon": 0.015,
        "lw": 2.8,
        "ls": "-"
    },
    {
        "scenario_short": "NATO→5%, nonNATO→3.5%",
        "baseline": 0.055,
        "g_world": 0.03,
        "d_mil": -0.03,
        "d_rest": -0.03,
        "epsilon": 0.015,
        "lw": 2.8,
        "ls": "-"
    },
    {
        "scenario_short": "Baseline",
        "baseline": 0.055,
        "g_world": 0.02,
        "d_mil": -0.03,
        "d_rest": -0.03,
        "epsilon": 0.015,
        "lw": 2.8,
        "ls": "-"
    },
    {
        "scenario_short": "NATO→5%, nonNATO→3.5%",
        "baseline": 0.055,
        "g_world": 0.02,
        "d_mil": -0.03,
        "d_rest": -0.03,
        "epsilon": 0.015,
        "lw": 2.8,
        "ls": "-"
    },

    {
        "scenario_short": "NATO→5%, nonNATO→3.5%",
        "baseline": 0.055,
        "g_world": 0.03,
        "d_mil": -0.03,
        "d_rest": -0.03,
        "epsilon": 0.015,
        "lw": 2.8,
        "ls": "-"
    },
    {
        "scenario_short": "Baseline",
        "baseline": 0.055,
        "g_world": 0.03,
        "d_mil": -0.01,
        "d_rest": -0.03,
        "epsilon": 0.015,
        "lw": 2.8,
        "ls": "-"
    },
]


def pct_from_fraction_1dec(x):
    return f"{100*x:.1f}%"


def pct_from_rate_int(x):
    return f"{100*x:.0f}%"


selected = []

for p in pathways:
    sub = df[
        (df["scenario_short"] == p["scenario_short"]) &
        (np.isclose(df["s_mil_2025_assumed"], p["baseline"])) &
        (np.isclose(df["g_world"], p["g_world"])) &
        (np.isclose(df["d_mil"], p["d_mil"])) &
        (np.isclose(df["d_rest"], p["d_rest"])) &
        (np.isclose(df["epsilon"], p["epsilon"]))
    ].copy()

    if sub.empty:
        print("WARNING: pathway not found:", p)
        continue

    s_code = sub["S_code"].iloc[0]
    label = (
        f"{s_code} "
        f"(b={pct_from_fraction_1dec(p['baseline'])}, "
        f"g={pct_from_rate_int(p['g_world'])}, "
        f"d_m={pct_from_rate_int(p['d_mil'])}, "
        f"d_rest={pct_from_rate_int(p['d_rest'])}, "
        f"ε={pct_from_fraction_1dec(p['epsilon'])})"
    )

    sub["label"] = label
    sub["lw"] = p["lw"]
    sub["ls"] = p["ls"]
    selected.append(sub)

if not selected:
    raise ValueError("None of the requested pathways were found in the simulation file.")

plot_df = pd.concat(selected, ignore_index=True)

# ============================================================
# LOAD AR6 SSP REFERENCE PATHWAYS + CLIMATE CATEGORIES
# ============================================================
variable_ar6 = "AR6 climate diagnostics|Infilled|Emissions|Kyoto Gases (AR6-GWP100)"
ssp_scenarios = [
    "SSP1-19",
    "SSP1-26",
    "SSP2-45",
    "SSP3-7.0_zeromig",
    "SSP5-8.5_zeromig",
]
year_cols = [str(y) for y in range(2025, 2051)]

with zipfile.ZipFile(ar6_zip_file) as z:
    with z.open("AR6_Scenarios_Database_World_v1.1.csv") as f:
        ar6_raw = pd.read_csv(
            f,
            usecols=["Model", "Scenario", "Region", "Variable", "Unit"] + year_cols,
            low_memory=False
        )

    with z.open("AR6_Scenarios_Database_metadata_indicators_v1.1.xlsx") as f:
        ar6_meta = pd.read_excel(
            io.BytesIO(f.read()),
            sheet_name="meta_Ch3vetted_withclimate"
        )

ar6_ghg = ar6_raw[
    (ar6_raw["Region"] == "World") &
    (ar6_raw["Variable"] == variable_ar6)
].copy()

meta_cols = ["Model", "Scenario", "Category", "Category_name", "Category_color_hex"]
ar6_ghg = ar6_ghg.merge(
    ar6_meta[meta_cols].drop_duplicates(),
    on=["Model", "Scenario"],
    how="left"
)

ar6_ghg_long = ar6_ghg.melt(
    id_vars=["Model", "Scenario", "Region", "Variable", "Unit", "Category", "Category_name", "Category_color_hex"],
    value_vars=year_cols,
    var_name="year",
    value_name="value_mt"
)
ar6_ghg_long["year"] = ar6_ghg_long["year"].astype(int)
ar6_ghg_long["value_gt"] = ar6_ghg_long["value_mt"] / 1000.0

all_cat_order = [f"C{i}" for i in range(1, 9)]

category_bands_all = (
    ar6_ghg_long[
        ar6_ghg_long["Category"].isin(all_cat_order)
    ]
    .groupby(["Category", "year"], as_index=False)
    .agg(
        ymin=("value_gt", "min"),
        ymax=("value_gt", "max")
    )
)

cat_order = [c for c in all_cat_order if c in SELECTED_CATEGORIES]

category_bands = category_bands_all[
    category_bands_all["Category"].isin(cat_order)
].copy()


def normalize_hex_color(x):
    if pd.isna(x):
        return None
    x = str(x).strip()
    if not x:
        return None
    return x if x.startswith("#") else f"#{x}"


cat_color_map = (
    ar6_meta[["Category", "Category_color_hex"]]
    .dropna()
    .drop_duplicates()
    .assign(color=lambda d: d["Category_color_hex"].map(normalize_hex_color))
    .set_index("Category")["color"]
    .to_dict()
)

ar6_ssp = ar6_ghg[ar6_ghg["Scenario"].isin(ssp_scenarios)].copy()

ar6_long = ar6_ssp.melt(
    id_vars=["Model", "Scenario", "Region", "Variable", "Unit", "Category", "Category_name", "Category_color_hex"],
    value_vars=year_cols,
    var_name="year",
    value_name="value_mt"
)
ar6_long["year"] = ar6_long["year"].astype(int)
ar6_long["value_gt"] = ar6_long["value_mt"] / 1000.0

ar6_plot = (
    ar6_long.groupby(["Scenario", "year"], as_index=False)["value_gt"]
    .median()
)

ssp_label_map = {
    "SSP1-19": "SSP1-1.9",
    "SSP1-26": "SSP1-2.6",
    "SSP2-45": "SSP2-4.5",
    "SSP3-7.0_zeromig": "SSP3-7.0",
    "SSP5-8.5_zeromig": "SSP5-8.5",
}
ar6_plot["label"] = ar6_plot["Scenario"].map(ssp_label_map)

# ============================================================
# HELPERS
# ============================================================
def adjusted_label_positions(points, min_sep):
    pts = sorted(points, key=lambda d: d["y"])
    adjusted = []
    for p in pts:
        y_text = p["y"]
        if adjusted:
            y_text = max(y_text, adjusted[-1]["y_text"] + min_sep)
        q = p.copy()
        q["y_text"] = y_text
        adjusted.append(q)
    return adjusted


def darken_color(color, factor=0.75):
    rgb = np.array(mcolors.to_rgb(color))
    return tuple(np.clip(rgb * factor, 0, 1))


def closest_category_for_y(y, cat_table):
    inside = cat_table[(cat_table["ymin"] <= y) & (cat_table["ymax"] >= y)]
    if not inside.empty:
        idx = (inside["mid"] - y).abs().idxmin()
        return inside.loc[idx, "Category"]
    idx = (cat_table["mid"] - y).abs().idxmin()
    return cat_table.loc[idx, "Category"]

# ============================================================
# MATCH SIMULATED PATHWAY COLOURS TO CLOSEST AR6 CATEGORY BAND
# ============================================================
cat_2050_all = category_bands_all[category_bands_all["year"] == 2050].copy()
cat_2050_all["mid"] = 0.5 * (cat_2050_all["ymin"] + cat_2050_all["ymax"])

c8_rows = cat_2050_all.loc[cat_2050_all["Category"] == "C8", "ymax"]
if c8_rows.empty:
    raise ValueError("Could not find C8 band for year 2050 in AR6 category table.")
c8_2050_max = c8_rows.iloc[0]

sim_labels = plot_df["label"].unique()
sim_colors = {}

for lab in sim_labels:
    sub = plot_df[plot_df["label"] == lab].sort_values("year")
    y2050_series = sub.loc[sub["year"] == 2050, "E_world_Gt"]
    if y2050_series.empty:
        raise ValueError(f"No 2050 point found for simulated pathway: {lab}")
    y2050 = y2050_series.iloc[0]

    if y2050 > c8_2050_max:
        sim_colors[lab] = "grey"
    else:
        cat = closest_category_for_y(y2050, cat_2050_all)
        sim_colors[lab] = darken_color(cat_color_map.get(cat, "#000000"), factor=0.75)

# ============================================================
# PLOT
# ============================================================
fig, ax = plt.subplots(figsize=(14, 10))

for cat in cat_order:
    sub = category_bands[category_bands["Category"] == cat].sort_values("year")
    if sub.empty:
        continue

    ax.fill_between(
        sub["year"],
        sub["ymin"],
        sub["ymax"],
        color=cat_color_map.get(cat, "lightgrey"),
        alpha=category_alpha,
        zorder=0
    )

ax.fill_between(
    envelope["year"],
    envelope["ymin"],
    envelope["ymax"],
    color="lightgrey",
    alpha=0.35,
    zorder=1
)

ssp_order = ["SSP1-1.9", "SSP1-2.6", "SSP2-4.5", "SSP3-7.0", "SSP5-8.5"]
ssp_colors = {
    "SSP1-1.9": "#1b9e77",
    "SSP1-2.6": "#66a61e",
    "SSP2-4.5": "#7570b3",
    "SSP3-7.0": "#e6ab02",
    "SSP5-8.5": "#d95f02",
}

ssp_points = []
for lab in ssp_order:
    sub = ar6_plot[ar6_plot["label"] == lab].sort_values("year")
    if sub.empty:
        print(f"WARNING: SSP line not found for {lab}")
        continue

    ax.plot(
        sub["year"],
        sub["value_gt"],
        color=ssp_colors[lab],
        lw=2.2,
        ls=":",
        alpha=1.0,
        zorder=3
    )

    y_end_series = sub.loc[sub["year"] == 2050, "value_gt"]
    if not y_end_series.empty:
        ssp_points.append({"label": lab, "y": y_end_series.iloc[0], "color": ssp_colors[lab]})

sim_points = []
for lab in sim_labels:
    sub = plot_df[plot_df["label"] == lab].sort_values("year")

    ax.plot(
        sub["year"],
        sub["E_world_Gt"],
        color=sim_colors[lab],
        lw=sub["lw"].iloc[0],
        ls=sub["ls"].iloc[0],
        zorder=4
    )

    y_end_series = sub.loc[sub["year"] == 2050, "E_world_Gt"]
    if not y_end_series.empty:
        sim_points.append({"label": lab, "y": y_end_series.iloc[0], "color": sim_colors[lab]})

sim_points_adj = adjusted_label_positions(sim_points, min_sep=4)
ssp_points_adj = adjusted_label_positions(ssp_points, min_sep=4)

manual_shift = {
    "S0 (b=5.5%, g=3%, d_m=-1%, d_rest=-1%, ε=1.5%)": -3,
    "S1 (b=5.5%, g=3%, d_m=-1%, d_rest=-1%, ε=1.5%)": -3,
    "S2 (b=5.5%, g=3%, d_m=-1%, d_rest=-1%, ε=1.5%)": -3,
    "S3 (b=5.5%, g=3%, d_m=-1%, d_rest=-1%, ε=1.5%)": -3,

    "S0 (b=5.5%, g=3%, d_m=-3%, d_rest=-3%, ε=1.5%)": 2.0,
    "S3 (b=5.5%, g=3%, d_m=-3%, d_rest=-3%, ε=1.5%)": 2.0,

    "S0 (b=3.3%, g=2%, d_m=-5%, d_rest=-5%, ε=0.0%)": -2.0,
    "S0 (b=5.5%, g=3%, d_m=-1%, d_rest=-3%, ε=1.5%)": 2.0,
    "S0 (b=3.3%, g=1%, d_m=-7%, d_rest=-7%, ε=0.0%)": 0.5,
}

for p in sim_points_adj:
    if p["label"] in manual_shift:
        p["y_text"] += manual_shift[p["label"]]

x_end = 2050
x_sim = 2053.0
x_ssp = 2060.0

for p in sim_points_adj:
    ax.plot([x_end, x_sim - 0.25], [p["y"], p["y_text"]], color=p["color"], lw=1.0)
    ax.text(
        x_sim,
        p["y_text"],
        p["label"],
        color=p["color"],
        fontsize=10.5,
        va="center"
    )

for p in ssp_points_adj:
    ax.plot([x_end, x_ssp - 0.25], [p["y"], p["y_text"]], color=p["color"], lw=0.9)
    ax.text(
        x_ssp,
        p["y_text"],
        p["label"],
        color=p["color"],
        fontsize=10,
        va="center"
    )

# ============================================================
# LEGEND
# ============================================================
category_handles = [
    Patch(
        facecolor=cat_color_map.get(cat, "lightgrey"),
        edgecolor="none",
        alpha=category_alpha,
        label=cat
    )
    for cat in cat_order
]

ssp_handles = [
    Line2D([0], [0], color=ssp_colors[lab], lw=2.2, ls=":", label=lab)
    for lab in ssp_order
]

other_handles = [
    Patch(facecolor="lightgrey", edgecolor="none", alpha=0.35, label="Envelope of simulated pathways"),
]

combined_handles = category_handles + ssp_handles + other_handles

ax.legend(
    handles=combined_handles,
    title="AR6 climate categories and reference pathways",
    loc="upper left",
    bbox_to_anchor=(LEGEND_X, LEGEND_Y),
    frameon=True,
    fontsize=10,
    title_fontsize=11,
    ncol=LEGEND_NCOLS
)

# ============================================================
# STYLING
# ============================================================
ax.axvline(2035, color="black", ls="--", lw=1.0, alpha=0.6)
ax.text(2035 + 0.2, envelope["ymax"].max() - 1.5, "2035", fontsize=10)

ax.set_title("Global GHG emissions pathways to 2050", fontsize=20, weight="bold")
ax.set_xlabel("Year", fontsize=14)
ax.set_ylabel("Total global GHG emissions (GtCO$_2$-eq/year)", fontsize=14)

ax.set_xlim(2025, 2068)

y_min = min(envelope["ymin"].min(), ar6_ghg_long["value_gt"].min()) - 2
label_y_values = [p["y_text"] for p in sim_points_adj] + [p["y_text"] for p in ssp_points_adj]
y_max = max(
    envelope["ymax"].max(),
    max(label_y_values) if label_y_values else envelope["ymax"].max()
) + 2
ax.set_ylim(y_min, y_max)

ax.grid(axis="y", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

plt.savefig("Fig_6.pdf", bbox_inches="tight")
plt.savefig("Fig_6.png", dpi=300, bbox_inches="tight")

plt.show()