"""Generate Figures 5--7 for the cumulative military-emissions results."""

import argparse
from pathlib import Path
import importlib.util
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tmp" / "python_deps"))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


DATA_PATH = PROJECT_ROOT / "data" / "generated" / "generated_data_2025-2050.csv"
SA_BUDGET_PATH = (
    PROJECT_ROOT
    / "output"
    / "three_sensitivity_analyses"
    / "Combined"
    / "combined_actual_projected_budget_results_2026-2050.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "output" / "cumulative_military_emissions"
MANUSCRIPT_DIR = PROJECT_ROOT / "manuscript"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ABSOLUTE_PLOT_PATH = PROJECT_ROOT / "model" / "plot_absolute_military_emissions.py"
spec = importlib.util.spec_from_file_location("absolute_plot_style", ABSOLUTE_PLOT_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load {ABSOLUTE_PLOT_PATH}")
absolute_plot = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = absolute_plot
spec.loader.exec_module(absolute_plot)


RC_PARAMS = dict(absolute_plot.HARMONIZED_RC_PARAMS)
BASELINES = [0.033, 0.055, 0.070]
BASELINE_LABELS = {
    0.033: "Baseline = 3.3%",
    0.055: "Baseline = 5.5%",
    0.070: "Baseline = 7.0%",
}
SCENARIOS = ["S0", "S1", "S2", "S3"]
COLUMN_TITLES = absolute_plot.HARMONIZED_SCENARIO_TITLES
DM_FIG5 = [0.07, 0.05, 0.03, 0.01, 0.00]
DM_FIG6 = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
DM_COLORS = {
    value: absolute_plot.HARMONIZED_COLOR_PALETTE[index]
    for index, value in enumerate(DM_FIG5)
}
SCENARIO_COLORS = {
    "S1": "#0072B2",
    "S2": "#E69F00",
    "S3": "#D55E00",
}
BUDGET_1P5 = 130.0
BUDGET_2C = 1050.0
CO2_SHARE = 0.74
FIG7_BASELINE = 0.055
FIG7_D_REST = 0.01
FIG7_EPSILON = 0.0
FIG7_GROWTH_VALUES = [0.01, 0.02, 0.03, 0.04]
FIG7_D_MIL_VALUES = [0.07, 0.05, 0.03, 0.01, 0.0]
FIG7_USED_COLOR = "#4C72B0"
FIG7_REMAINING_COLOR = "#CBD5E5"
FIG7_TWO_C_COLOR = "#A5160A"
FIG7_GRID_COLOR = "#B8B8B8"
AR6_PATHWAYS = ["SSP1-1.9", "SSP1-2.6", "SSP2-4.5", "SSP3-7.0", "SSP5-8.5"]
PATHWAY_COLORS = {
    "SSP1-1.9": "#1b4f72",
    "SSP1-2.6": "#2874a6",
    "SSP2-4.5": "#7d8f69",
    "SSP3-7.0": "#c27c0e",
    "SSP5-8.5": "#922b21",
}
PPP_COLOR = "#7B2CBF"


def load_cumulative_data() -> pd.DataFrame:
    raw = pd.read_csv(DATA_PATH)
    raw = raw[raw["year"].between(2025, 2050)].copy()

    # E_mil is invariant to d_rest and epsilon. Remove those duplicate rows
    # before accumulating annual military emissions.
    unique = raw.drop_duplicates(
        subset=[
            "year",
            "scenario_code",
            "s_mil_2025_assumed",
            "g_world",
            "d_mil",
            "E_mil_Gt",
        ]
    ).copy()
    unique = unique.sort_values(
        ["scenario_code", "s_mil_2025_assumed", "g_world", "d_mil", "year"]
    )
    unique["cum_E_mil_Gt"] = unique.groupby(
        ["scenario_code", "s_mil_2025_assumed", "g_world", "d_mil"]
    )["E_mil_Gt"].cumsum()
    return unique


def style_grid_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}"))
    ax.grid(True, alpha=0.20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=10, width=0.5)


@mpl.rc_context(RC_PARAMS)
def make_fig_5(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        3, 4, figsize=(11, 10.4), sharex=True, sharey=True
    )
    range_offsets = np.linspace(0.18, 0.82, len(DM_FIG5))
    panel_counter = 0

    for row_index, baseline in enumerate(BASELINES):
        for column_index, scenario_code in enumerate(SCENARIOS):
            ax = axes[row_index, column_index]
            scenario_data = data[
                np.isclose(data["s_mil_2025_assumed"], baseline)
                & (data["scenario_code"] == scenario_code)
            ].copy()
            ax.axvline(2050.07, color="#BFBFBF", linewidth=0.5, zorder=0)

            for offset, d_m in zip(range_offsets, DM_FIG5):
                subset = scenario_data[np.isclose(scenario_data["d_mil"], d_m)]
                envelope = (
                    subset.groupby("year", as_index=False)["cum_E_mil_Gt"]
                    .agg(minimum="min", maximum="max")
                    .sort_values("year")
                )
                deterministic = (
                    subset[np.isclose(subset["g_world"], 0.03)]
                    .groupby("year", as_index=False)["cum_E_mil_Gt"]
                    .mean()
                    .sort_values("year")
                )

                ax.fill_between(
                    envelope["year"],
                    envelope["minimum"],
                    envelope["maximum"],
                    color=DM_COLORS[d_m],
                    alpha=0.085,
                    linewidth=0,
                    zorder=1,
                )
                ax.plot(
                    deterministic["year"],
                    deterministic["cum_E_mil_Gt"],
                    color=DM_COLORS[d_m],
                    linewidth=1.55,
                    zorder=3,
                )

                values_2050 = subset.loc[subset["year"] == 2050, "cum_E_mil_Gt"]
                centre = float(
                    deterministic.loc[
                        deterministic["year"] == 2050, "cum_E_mil_Gt"
                    ].iloc[0]
                )
                lower = float(values_2050.min())
                upper = float(values_2050.max())
                ax.errorbar(
                    2050 + offset,
                    centre,
                    yerr=np.array([[centre - lower], [upper - centre]]),
                    fmt="o",
                    color=DM_COLORS[d_m],
                    ecolor=DM_COLORS[d_m],
                    elinewidth=0.9,
                    capsize=2.1,
                    capthick=0.9,
                    markersize=2.8,
                    markeredgewidth=0,
                    zorder=5,
                )

            if np.isclose(baseline, 0.055):
                reference = scenario_data[
                    np.isclose(scenario_data["g_world"], 0.03)
                    & np.isclose(scenario_data["d_mil"], 0.01)
                ].sort_values("year")
                ax.plot(
                    reference["year"],
                    reference["cum_E_mil_Gt"],
                    color="#111111",
                    linewidth=1.8,
                    zorder=10,
                )

            style_grid_axis(ax)
            ax.set_xlim(2025, 2051.05)
            ax.set_ylim(0, 285)
            ax.set_xticks([2025, 2035, 2050])

            if column_index == 0:
                ax.set_ylabel(
                    f"{BASELINE_LABELS[baseline]}\n"
                    "Cumulative emissions (GtCO$_2$e)",
                    fontsize=10,
                )
            if row_index == 0:
                ax.set_title(
                    COLUMN_TITLES[scenario_code],
                    fontsize=11,
                    pad=5,
                    linespacing=1.0,
                )
            ax.text(
                0.02,
                0.95,
                f"({chr(97 + panel_counter)})",
                transform=ax.transAxes,
                fontsize=11,
                fontweight="bold",
                va="top",
            )
            panel_counter += 1

    for ax in axes[-1, :]:
        ax.set_xlabel("Year", fontsize=10)

    line_handles = [
        Line2D(
            [0],
            [0],
            color=DM_COLORS[d_m],
            lw=2.2,
            label=f"$d_m$ = {d_m * 100:.0f}%/yr",
        )
        for d_m in DM_FIG5
    ]
    context_handles = [
        Patch(
            facecolor="#777777",
            alpha=0.16,
            edgecolor="none",
            label="Ribbons: full tested parameter range across g = 1-4%/yr",
        ),
        Line2D(
            [0],
            [0],
            color="#555555",
            marker="o",
            markersize=3.2,
            lw=0.9,
            label="2050 whiskers: full tested range across g = 1-4%/yr",
        ),
        Line2D(
            [0],
            [0],
            color="#111111",
            lw=2.5,
            label=(
                "Main model reference case: baseline = 5.5%, g = 3%, "
                "$d_m$ = 1%/yr"
            ),
        ),
    ]
    fig.legend(
        handles=line_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.112),
        ncol=5,
        frameon=False,
        fontsize=10,
        columnspacing=1.35,
        handletextpad=0.5,
        borderaxespad=0.2,
    )
    fig.legend(
        handles=context_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.047),
        ncol=2,
        frameon=False,
        fontsize=9.7,
        columnspacing=1.20,
        handletextpad=0.5,
        labelspacing=0.35,
        borderaxespad=0.2,
    )
    fig.suptitle(
        "Cumulative military GHG emissions (2025-2050)", fontsize=14, y=0.98
    )
    fig.tight_layout(rect=[0.035, 0.145, 0.995, 0.955], h_pad=1.2, w_pad=0.8)
    fig.savefig(OUTPUT_DIR / "Fig_5.pdf", format="pdf", dpi=300)
    fig.savefig(OUTPUT_DIR / "Fig_5.png", format="png", dpi=180)
    plt.close(fig)


@mpl.rc_context(RC_PARAMS)
def make_fig_6(data: pd.DataFrame) -> None:
    cumulative_2050 = data[data["year"] == 2050].copy()
    s0_comparator = cumulative_2050[
        (cumulative_2050["scenario_code"] == "S0")
        & np.isclose(cumulative_2050["d_mil"], 0.01)
    ][["s_mil_2025_assumed", "g_world", "cum_E_mil_Gt"]].rename(
        columns={"cum_E_mil_Gt": "s0_cumulative"}
    )
    differences = cumulative_2050[
        cumulative_2050["scenario_code"].isin(["S1", "S2", "S3"])
    ].merge(
        s0_comparator,
        on=["s_mil_2025_assumed", "g_world"],
        validate="many_to_one",
    )
    differences["difference"] = (
        differences["cum_E_mil_Gt"] - differences["s0_cumulative"]
    )
    summary = (
        differences.groupby(["scenario_code", "d_mil"], as_index=False)[
            "difference"
        ]
        .agg(minimum="min", median="median", maximum="max")
    )

    x_positions = np.arange(len(DM_FIG6))
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    line_handles = []

    for scenario_code in ["S1", "S2", "S3"]:
        scenario_summary = (
            summary[summary["scenario_code"] == scenario_code]
            .set_index("d_mil")
            .reindex(DM_FIG6)
        )
        color = SCENARIO_COLORS[scenario_code]
        ax.fill_between(
            x_positions,
            scenario_summary["minimum"].to_numpy(dtype=float),
            scenario_summary["maximum"].to_numpy(dtype=float),
            color=color,
            alpha=0.18,
            linewidth=0,
            zorder=1,
        )
        line, = ax.plot(
            x_positions,
            scenario_summary["median"].to_numpy(dtype=float),
            marker="o",
            linewidth=1.55,
            markersize=6.5,
            color=color,
            label=scenario_code,
            zorder=3,
        )
        line_handles.append(line)

    zero_line = ax.axhline(
        0, color="#111111", linestyle="--", linewidth=1.2, zorder=2
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{value * 100:.0f}%" for value in DM_FIG6])
    ax.set_xlabel("Military decarbonisation rate, $d_m$")
    ax.set_ylabel(
        "$\\Delta$ cumulative military emissions, 2025--2050 (GtCO$_2$e)"
    )
    ax.set_title(
        "Can military-side decarbonisation offset burden escalation?\n"
        "Difference relative to Baseline (S0) with $d_m=1\\%$/yr"
    )
    ax.set_xlim(-0.2, len(DM_FIG6) - 0.8)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.02,
        0.98,
        "Above zero: escalation still exceeds S0\n"
        "Below zero: decarbonisation fully offsets escalation",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
    )

    ribbon_patch = Patch(
        facecolor="grey",
        edgecolor="none",
        alpha=0.18,
        label=(
            "Ribbon: min-max across baseline = 3.3-7.0% "
            "and $g$ = 1-4%/yr"
        ),
    )
    zero_handle = Line2D(
        [0],
        [0],
        color="#111111",
        linestyle="--",
        linewidth=1.3,
        label="Break-even line",
    )
    ax.legend(
        handles=line_handles + [ribbon_patch, zero_handle],
        frameon=False,
        loc="upper right",
        title="Lines: median of deterministic parameter grid",
    )
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "Fig_6.pdf", format="pdf", dpi=300,
        bbox_inches="tight"
    )
    fig.savefig(
        OUTPUT_DIR / "Fig_6.png", format="png", dpi=180,
        bbox_inches="tight"
    )
    plt.close(fig)


def load_budget_comparison_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the main-model grid and SA1--SA3 carbon-budget comparisons."""
    main = pd.read_csv(DATA_PATH)
    main = main[
        main["scenario_code"].isin(SCENARIOS)
        & np.isclose(main["s_mil_2025_assumed"], FIG7_BASELINE)
        & main["g_world"].round(4).isin(FIG7_GROWTH_VALUES)
        & main["d_mil"].round(4).isin(FIG7_D_MIL_VALUES)
        & np.isclose(main["d_rest"], FIG7_D_REST)
        & np.isclose(main["epsilon"], FIG7_EPSILON)
        & main["year"].between(2026, 2050)
    ].copy()

    group_cols = ["scenario_code", "g_world", "d_mil"]
    main = main.groupby(group_cols, as_index=False)["E_mil_Gt"].sum()
    main["cum_E_mil_CO2_Gt"] = main["E_mil_Gt"] * CO2_SHARE
    main["pct_1p5"] = 100 * main["cum_E_mil_CO2_Gt"] / BUDGET_1P5
    main["pct_2c"] = 100 * main["cum_E_mil_CO2_Gt"] / BUDGET_2C
    main["label"] = main.apply(
        lambda row: (
            f"g={int(round(row['g_world'] * 100))}%, "
            f"dₘ={int(round(row['d_mil'] * 100))}%"
        ),
        axis=1,
    )
    main["is_reference"] = (
        np.isclose(main["g_world"], 0.03)
        & np.isclose(main["d_mil"], 0.01)
    )

    sensitivity = pd.read_csv(SA_BUDGET_PATH)
    sensitivity = sensitivity[
        sensitivity["combined_model"].isin(["SA1", "SA2", "SA3"])
    ].copy()
    return main, sensitivity


def draw_sensitivity_bars(
    ax: plt.Axes,
    subset: pd.DataFrame,
    positions: np.ndarray,
    colors: list[str],
) -> None:
    used = subset["pct_1p5_budget_used"].to_numpy(dtype=float)
    remaining = np.clip(100 - used, 0, None)
    ax.bar(positions, used, width=0.29, color=colors, zorder=2)
    ax.bar(
        positions,
        remaining,
        width=0.29,
        bottom=used,
        color=FIG7_REMAINING_COLOR,
        zorder=1,
    )
    ax.scatter(
        positions,
        subset["pct_2C_budget_used"],
        marker="D",
        s=25,
        color=FIG7_TWO_C_COLOR,
        edgecolor="white",
        linewidth=0.35,
        zorder=4,
    )


@mpl.rc_context(RC_PARAMS)
def make_fig_7(main: pd.DataFrame, sensitivity: pd.DataFrame) -> None:
    """Plot main-model parameter combinations alongside SA1--SA3."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12.5,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    axes = axes.flatten()
    sa1_positions = 20.85 + np.arange(5) * 0.36
    sa2_positions = 22.85 + np.arange(5) * 0.36
    sa3_positions = np.array([25.05])
    sensitivity_groups = {
        "SA1": (sa1_positions, sa1_positions.mean()),
        "SA2": (sa2_positions, sa2_positions.mean()),
        "SA3": (sa3_positions, sa3_positions.mean()),
    }

    for ax, scenario_code in zip(axes, SCENARIOS):
        panel = (
            main[main["scenario_code"] == scenario_code]
            .sort_values("pct_1p5")
            .reset_index(drop=True)
        )
        x = np.arange(len(panel))
        used = panel["pct_1p5"].to_numpy()
        remaining = np.clip(100 - used, 0, None)

        bars = ax.bar(x, used, width=0.78, color=FIG7_USED_COLOR, zorder=2)
        ax.bar(
            x,
            remaining,
            width=0.78,
            bottom=used,
            color=FIG7_REMAINING_COLOR,
            zorder=1,
        )
        ax.scatter(
            x,
            panel["pct_2c"],
            marker="D",
            s=34,
            color=FIG7_TWO_C_COLOR,
            zorder=4,
        )

        for bar, is_reference in zip(bars, panel["is_reference"]):
            if is_reference:
                bar.set_edgecolor("black")
                bar.set_linewidth(1.5)
                bar.set_hatch("///")

        tick_positions = list(x)
        tick_labels = list(panel["label"])
        for analysis in ["SA1", "SA2", "SA3"]:
            subset = sensitivity[
                (sensitivity["scenario_code"] == scenario_code)
                & (sensitivity["combined_model"] == analysis)
            ].copy()
            positions, group_center = sensitivity_groups[analysis]
            if analysis in ["SA1", "SA2"]:
                subset["ssp"] = pd.Categorical(
                    subset["ssp"], categories=AR6_PATHWAYS, ordered=True
                )
                subset = subset.sort_values("ssp")
                colors = [PATHWAY_COLORS[pathway] for pathway in subset["ssp"]]
            else:
                colors = [PPP_COLOR]
            draw_sensitivity_bars(ax, subset, positions, colors)
            tick_positions.append(group_center)
            tick_labels.append(analysis)

        ax.axvline(20.3, color="#777777", lw=0.9, ls=(0, (3, 3)), zorder=0)
        ax.axhline(100, linestyle="--", linewidth=1, color="#777777", zorder=3)
        ax.set_title(COLUMN_TITLES[scenario_code], pad=8)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        for index, label in enumerate(ax.get_xticklabels()):
            if index < len(panel):
                label.set_rotation(52)
                label.set_ha("right")
            else:
                label.set_rotation(0)
                label.set_ha("center")

        ax.set_xlim(-1, 25.65)
        ax.set_ylim(0, 128)
        ax.grid(axis="y", color=FIG7_GRID_COLOR, linewidth=0.6, alpha=0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Share of remaining carbon budget (%)")
    axes[2].set_ylabel("Share of remaining carbon budget (%)")
    axes[2].set_xlabel("Parameter combination")
    axes[3].set_xlabel("Parameter combination")

    metric_handles = [
        Patch(facecolor=FIG7_USED_COLOR, label="Share of 1.5°C budget"),
        Patch(facecolor=FIG7_REMAINING_COLOR, label="1.5°C budget remaining"),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor=FIG7_TWO_C_COLOR,
            markeredgecolor=FIG7_TWO_C_COLOR,
            markersize=6,
            label="Share of 2°C budget",
        ),
        Patch(
            facecolor=FIG7_USED_COLOR,
            edgecolor="black",
            hatch="///",
            label="Main-model reference case",
        ),
    ]
    pathway_handles = [
        Patch(facecolor=PATHWAY_COLORS[pathway], label=pathway)
        for pathway in AR6_PATHWAYS
    ] + [Patch(facecolor=PPP_COLOR, label="PPP recalibration (SA3)")]
    fig.legend(
        handles=metric_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.038),
    )
    fig.legend(
        handles=pathway_handles,
        loc="lower center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, 0.006),
    )
    fig.suptitle(
        "Military CO$_2$ emissions relative to remaining carbon budgets "
        "(2026-2050)\nBaseline military share = 5.5%",
        fontsize=16,
        y=0.985,
    )
    fig.tight_layout(rect=[0.025, 0.105, 1, 0.94], h_pad=1.6, w_pad=1.8)
    fig.savefig(OUTPUT_DIR / "Fig_7.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "Fig_7.png", format="png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Fig_5, Fig_6 and Fig_7 for the cumulative military-emissions "
            "results."
        )
    )
    parser.add_argument(
        "--sync-manuscript",
        action="store_true",
        help="Copy the generated PDF figures into the manuscript directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_cumulative_data()
    make_fig_5(data)
    make_fig_6(data)
    main_budget, sensitivity_budget = load_budget_comparison_data()
    make_fig_7(main_budget, sensitivity_budget)
    for name in ("Fig_5.pdf", "Fig_6.pdf", "Fig_7.pdf"):
        if args.sync_manuscript:
            shutil.copy2(OUTPUT_DIR / name, MANUSCRIPT_DIR / name)
        print(OUTPUT_DIR / name)


if __name__ == "__main__":
    main()
