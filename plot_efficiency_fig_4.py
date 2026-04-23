import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ============================================================
# FILE
# ============================================================

csv_file = "generated_data_budgets_2050.csv"

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(csv_file)

# ============================================================
# NORMALISE / CREATE REQUIRED COLUMNS
# ============================================================

# 1) Ensure cumulative emissions column exists as "cum_E_mil"
if "cum_E_mil" not in df.columns:
    if "cum_E_mil_Gt" in df.columns:
        df = df.rename(columns={"cum_E_mil_Gt": "cum_E_mil"})
    elif "E_mil_Gt" in df.columns:
        # Compute cumulative emissions from yearly time series
        if "year" not in df.columns:
            raise ValueError(
                "Found 'E_mil_Gt' but no 'year' column, so cumulative emissions cannot be computed."
            )

        group_keys_for_cumsum = [
            col for col in [
                "scenario",
                "scenario_code",
                "g_world",
                "d_mil",
                "d_rest",
                "epsilon",
                "s_mil_2025_assumed",
            ]
            if col in df.columns
        ]

        df = df.sort_values(group_keys_for_cumsum + ["year"]).copy()
        df["cum_E_mil"] = df.groupby(group_keys_for_cumsum)["E_mil_Gt"].cumsum()
    else:
        raise ValueError(
            "Missing cumulative emissions column. I could not find either "
            "'cum_E_mil', 'cum_E_mil_Gt', or yearly 'E_mil_Gt' to build it."
        )

# 2) Keep only year 2050 if a year column exists
if "year" in df.columns:
    df = df[df["year"] == 2050].copy()

# 3) Ensure scenario_code exists
if "scenario_code" not in df.columns:
    if "scenario" not in df.columns:
        raise ValueError("Missing both 'scenario_code' and 'scenario' columns.")

    scenario_str = df["scenario"].astype(str)

    df["scenario_code"] = np.select(
        [
            scenario_str.str.startswith("Baseline"),
            scenario_str.str.contains("NATO→3.5%, nonNATO holds", regex=False),
            scenario_str.str.contains("NATO→3.5%, nonNATO→3.5%", regex=False),
            scenario_str.str.contains("NATO→5%, nonNATO→3.5%", regex=False),
        ],
        ["S0", "S1", "S2", "S3"],
        default=np.nan,
    )

    if df["scenario_code"].isna().any():
        bad = sorted(df.loc[df["scenario_code"].isna(), "scenario"].dropna().unique())
        raise ValueError(
            "Some scenario names could not be mapped to S0/S1/S2/S3.\n"
            f"Unrecognised values: {bad}"
        )

# 4) If "scenario" is missing, create a readable version from scenario_code
if "scenario" not in df.columns:
    scenario_name_map = {
        "S0": "Baseline",
        "S1": "NATO→3.5%, nonNATO holds",
        "S2": "NATO→3.5%, nonNATO→3.5%",
        "S3": "NATO→5%, nonNATO→3.5%",
    }
    df["scenario"] = df["scenario_code"].map(scenario_name_map)

# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================
required_cols = {
    "scenario",
    "scenario_code",
    "g_world",
    "d_mil",
    "d_rest",
    "epsilon",
    "s_mil_2025_assumed",
    "cum_E_mil",
}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns after normalisation: {missing}")

# ============================================================
# SCENARIO DEFINITIONS
# ============================================================
SCENARIO_BASE = "S0"
SCENARIO_S1 = "S1"
SCENARIO_S2 = "S2"
SCENARIO_S3 = "S3"

scenario_order = [SCENARIO_S1, SCENARIO_S2, SCENARIO_S3]
scenario_labels = {
    SCENARIO_S1: "S1",
    SCENARIO_S2: "S2",
    SCENARIO_S3: "S3",
}
scenario_titles = {
    SCENARIO_S1: "NATO→3.5%, nonNATO holds",
    SCENARIO_S2: "NATO→3.5%, nonNATO→3.5%",
    SCENARIO_S3: "NATO→5%, nonNATO→3.5%",
}

scenario_colors = {
    SCENARIO_S1: "#3B6EA5",  # muted blue
    SCENARIO_S2: "#C9A227",  # muted ochre
    SCENARIO_S3: "#B24A4A",  # muted red
}

# ============================================================
# ORDER OF d_mil VALUES
# ============================================================
d_mil_order = sorted(df["d_mil"].dropna().unique(), reverse=True)
x_positions = np.arange(len(d_mil_order))
x_labels = [f"{int(round(v * 100))}%/yr" for v in d_mil_order]

print("d_mil values found:", d_mil_order)

# ============================================================
# BASELINE REFERENCE FOR COMPARISON
# Compare each escalation scenario to Baseline (S0) with d_mil = -0.01
# while keeping all other assumptions fixed
# ============================================================
merge_keys = ["g_world", "d_rest", "epsilon", "s_mil_2025_assumed"]

baseline_ref = df[
    (df["scenario_code"] == SCENARIO_BASE) &
    (np.isclose(df["d_mil"], -0.01))
].copy()

if baseline_ref.empty:
    raise ValueError("No Baseline (S0) rows with d_mil = -0.01 were found.")

baseline_ref = baseline_ref[merge_keys + ["cum_E_mil"]].rename(
    columns={"cum_E_mil": "cum_E_mil_baseline_ref"}
)

# ============================================================
# MERGE ESCALATION SCENARIOS WITH MATCHED BASELINE REFERENCE
# ============================================================
esc = df[df["scenario_code"].isin(scenario_order)].copy()

esc = esc.merge(
    baseline_ref,
    on=merge_keys,
    how="left",
    validate="many_to_one"
)

if esc["cum_E_mil_baseline_ref"].isna().any():
    bad_rows = esc[esc["cum_E_mil_baseline_ref"].isna()][
        ["scenario_code"] + merge_keys + ["d_mil"]
    ]
    print("\nRows that failed to match baseline reference:\n")
    print(bad_rows.to_string(index=False))
    raise ValueError(
        "Some escalation rows could not be matched to Baseline (S0) with d_mil = -0.01."
    )

# Difference from Baseline (S0) with d_mil = -1%/yr
esc["delta_cum_E_mil"] = esc["cum_E_mil"] - esc["cum_E_mil_baseline_ref"]

# ============================================================
# PRINT ALL VALUES USED IN THE PLOT
# ============================================================
plot_values = (
    esc[
        [
            "scenario_code", "scenario", "d_mil", "g_world", "d_rest", "epsilon",
            "s_mil_2025_assumed", "cum_E_mil",
            "cum_E_mil_baseline_ref", "delta_cum_E_mil"
        ]
    ]
    .copy()
    .sort_values(
        ["scenario_code", "d_mil", "s_mil_2025_assumed", "g_world", "d_rest", "epsilon"],
        ascending=[True, False, True, True, True, True]
    )
)

print("\n" + "=" * 100)
print("ALL UNDERLYING VALUES USED IN THE BREAK-EVEN PLOT")
print("=" * 100)
print(
    plot_values.to_string(
        index=False,
        formatters={
            "d_mil": "{:.2%}".format,
            "g_world": "{:.0%}".format,
            "d_rest": "{:.2%}".format,
            "epsilon": "{:.3f}".format,
            "s_mil_2025_assumed": "{:.3f}".format,
            "cum_E_mil": "{:.3f}".format,
            "cum_E_mil_baseline_ref": "{:.3f}".format,
            "delta_cum_E_mil": "{:.3f}".format,
        }
    )
)

plot_values.to_csv("break_even_plot_all_values.csv", index=False)

# ============================================================
# SUMMARY OF PLOTTED VALUES
# ============================================================
plot_summary = (
    esc.groupby(["scenario_code", "d_mil"])["delta_cum_E_mil"]
    .agg(min="min", median="median", max="max")
    .reset_index()
    .sort_values(["scenario_code", "d_mil"], ascending=[True, False])
)

print("\n" + "=" * 100)
print("VALUES ACTUALLY PLOTTED IN THE FIGURE")
print("median = line, min/max = ribbon bounds")
print("=" * 100)
print(
    plot_summary.to_string(
        index=False,
        formatters={
            "d_mil": "{:.2%}".format,
            "min": "{:.3f}".format,
            "median": "{:.3f}".format,
            "max": "{:.3f}".format,
        }
    )
)

plot_summary.to_csv("break_even_plot_summary.csv", index=False)

for scenario_code in scenario_order:
    sub = plot_summary[plot_summary["scenario_code"] == scenario_code].copy()

    print("\n" + "-" * 100)
    print(f"{scenario_labels[scenario_code]} — {scenario_titles[scenario_code]}")
    print("-" * 100)
    print(
        sub.to_string(
            index=False,
            columns=["d_mil", "min", "median", "max"],
            formatters={
                "d_mil": "{:.2%}".format,
                "min": "{:.3f}".format,
                "median": "{:.3f}".format,
                "max": "{:.3f}".format,
            }
        )
    )

# ============================================================
# SUMMARY TABLE FOR PLOTTING
# ============================================================
summary = (
    esc.groupby(["scenario_code", "d_mil"])["delta_cum_E_mil"]
    .agg(min="min", median="median", max="max")
    .reset_index()
)

summary["d_mil"] = pd.Categorical(summary["d_mil"], categories=d_mil_order, ordered=True)
summary = summary.sort_values(["scenario_code", "d_mil"])

print("\nSummary of Δ cumulative military emissions relative to Baseline with d_mil = -1%/yr:\n")
print(summary.to_string(index=False))

# ============================================================
# PLOT
# ============================================================
fig, ax = plt.subplots(figsize=(10.5, 6.5))

line_handles = []

for scenario_code in scenario_order:
    sub = summary[summary["scenario_code"] == scenario_code].copy()
    sub = sub.set_index("d_mil").reindex(d_mil_order).reset_index()

    y_min = sub["min"].to_numpy(dtype=float)
    y_med = sub["median"].to_numpy(dtype=float)
    y_max = sub["max"].to_numpy(dtype=float)

    color = scenario_colors[scenario_code]

    line, = ax.plot(
        x_positions,
        y_med,
        marker="o",
        linewidth=2.6,
        markersize=6.5,
        color=color,
        label=scenario_labels[scenario_code]
    )

    ax.fill_between(
        x_positions,
        y_min,
        y_max,
        alpha=0.18,
        color=color
    )

    line_handles.append(line)

zero_line = ax.axhline(
    0,
    linestyle="--",
    linewidth=1.3,
    color="black"
)

# ============================================================
# LABELS AND STYLING
# ============================================================
ax.set_xticks(x_positions)
ax.set_xticklabels(x_labels)
ax.set_xlabel("Military decarbonisation rate")
ax.set_ylabel(r"$\Delta$ cumulative military emissions, 2025--2050 (GtCO$_2$e)")
ax.set_title(
    "Can military-side decarbonisation offset burden escalation?\n"
    "Difference relative to Baseline (S0) with $d_{mil}=-1\\%$/yr"
)

ax.text(
    0.02, 0.98,
    "Above zero: escalation still exceeds baseline\nBelow zero: decarbonisation fully offsets escalation",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=10
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.25, linestyle=":")

# ============================================================
# LEGEND
# ============================================================
ribbon_patch = Patch(
    facecolor="grey",
    edgecolor="none",
    alpha=0.18,
    label=r"Ribbon: min–max across $g$, $d_{\mathrm{rest}}$, $\epsilon$, and $s^{\mathrm{mil}}_{2025}$"
)

zero_handle = Line2D(
    [0], [0],
    color="black",
    linestyle="--",
    linewidth=1.3,
    label="Break-even line"
)

legend_handles = line_handles + [ribbon_patch, zero_handle]
ax.legend(handles=legend_handles, frameon=False, loc="upper right")

plt.tight_layout()

# ============================================================
# SAVE
# ============================================================
plt.savefig("Fig_4.pdf", bbox_inches="tight")
plt.savefig("Fig_4.png", dpi=300, bbox_inches="tight")
plt.show()