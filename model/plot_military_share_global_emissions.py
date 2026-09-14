from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import shutil
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


ABSOLUTE_PLOT_PATH = (
    PROJECT_ROOT / "model" / "plot_absolute_military_emissions.py"
)
OUTPUT_DIR = PROJECT_ROOT / "output" / "military_share_global_emissions"
MANUSCRIPT_DIR = PROJECT_ROOT / "manuscript"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


spec = importlib.util.spec_from_file_location(
    "absolute_plot_for_share_preview", ABSOLUTE_PLOT_PATH
)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load {ABSOLUTE_PLOT_PATH}")
absolute_plot = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = absolute_plot
spec.loader.exec_module(absolute_plot)


RC_PARAMS = dict(absolute_plot.HARMONIZED_RC_PARAMS)
RC_PARAMS["legend.fontsize"] = 10

BASELINES = [0.033, 0.055, 0.070]
SCENARIOS = ["S0", "S1", "S2", "S3"]
BASELINE_LABELS = {
    0.033: "Baseline = 3.3%",
    0.055: "Baseline = 5.5%",
    0.070: "Baseline = 7.0%",
}
COLUMN_TITLES = absolute_plot.HARMONIZED_SCENARIO_TITLES
DM_VALUES = [0.00, 0.01, 0.03, 0.05, 0.07]
DR_VALUES = [0.01, 0.03, 0.05, 0.07]
G_VALUES = [0.01, 0.02, 0.03, 0.04]
EPSILON_VALUES = [0.0, 0.009, 0.015, 0.02]
DM_COLORS = {
    value: absolute_plot.HARMONIZED_COLOR_PALETTE[index]
    for index, value in enumerate(sorted(DM_VALUES, reverse=True))
}
DR_STYLES = {
    value: style
    for value, style in zip(
        sorted(DR_VALUES, reverse=True), ["-", "--", "-.", ":"]
    )
}
MAIN_COLOR = "#111111"
PPP_COLOR = "#7B2CBF"


def style_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}"))
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=10, width=0.5)


def reference_path(df_scenario: pd.DataFrame) -> pd.DataFrame:
    selected = df_scenario[
        np.isclose(df_scenario["s_mil_2025_assumed"], 0.055)
        & np.isclose(df_scenario["g_world"], 0.03)
        & np.isclose(df_scenario["d_mil"], 0.01)
        & np.isclose(df_scenario["d_rest"], 0.01)
        & np.isclose(df_scenario["epsilon"], 0.0)
    ].copy()
    return (
        selected.groupby("year", as_index=False)["share_pct"]
        .mean()
        .sort_values("year")
    )


def configure_grid(
    fig: plt.Figure,
    axes: np.ndarray,
    row_labels: dict[float, str] | dict[str, str],
    row_order: list,
    ylabel: str,
    x_max: float,
) -> None:
    panel_counter = 0
    for row_index, row_key in enumerate(row_order):
        for column_index, scenario_code in enumerate(SCENARIOS):
            ax = axes[row_index, column_index]
            ax.set_xlim(2025, x_max)
            ax.set_xticks([2025, 2030, 2035])
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
        axes[row_index, 0].set_ylabel(
            f"{row_labels[row_key]}\n{ylabel}", fontsize=10
        )

    for column_index, scenario_code in enumerate(SCENARIOS):
        axes[0, column_index].set_title(
            COLUMN_TITLES[scenario_code], fontsize=11, pad=5, linespacing=1.0
        )
    for ax in axes[-1, :]:
        ax.set_xlabel("Year", fontsize=10)


@mpl.rc_context(RC_PARAMS)
def make_fig_3(main_data: pd.DataFrame) -> None:
    """Main-text parameter-envelope figure for military emissions shares."""
    fig, axes = plt.subplots(
        3, 4, figsize=(11, 10.4), sharex=True, sharey=True
    )
    range_offsets = np.linspace(0.18, 0.82, len(DM_VALUES))

    for row_index, baseline in enumerate(BASELINES):
        for column_index, scenario_code in enumerate(SCENARIOS):
            ax = axes[row_index, column_index]
            scenario_data = main_data[
                np.isclose(main_data["s_mil_2025_assumed"], baseline)
                & (main_data["scenario_code"] == scenario_code)
            ].copy()
            ax.axvline(2035.07, color="#BFBFBF", linewidth=0.5, zorder=0)

            for offset, d_m in zip(
                range_offsets, sorted(DM_VALUES, reverse=True)
            ):
                subset = scenario_data[np.isclose(scenario_data["d_mil"], d_m)]
                subset = subset.drop_duplicates(
                    subset=[
                        "g_world",
                        "d_mil",
                        "d_rest",
                        "epsilon",
                        "year",
                        "share_pct",
                    ]
                )
                envelope = (
                    subset.groupby("year", as_index=False)["share_pct"]
                    .agg(minimum="min", maximum="max")
                    .sort_values("year")
                )
                reference_conditions = (
                    np.isclose(subset["g_world"], 0.03)
                    & np.isclose(subset["d_rest"], 0.01)
                    & np.isclose(subset["epsilon"], 0.0)
                )
                deterministic = (
                    subset.loc[reference_conditions]
                    .groupby("year", as_index=False)["share_pct"]
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
                    deterministic["share_pct"],
                    color=DM_COLORS[d_m],
                    linewidth=1.55,
                    zorder=3,
                )

                values_2035 = subset.loc[subset["year"] == 2035, "share_pct"]
                centre = float(
                    deterministic.loc[
                        deterministic["year"] == 2035, "share_pct"
                    ].iloc[0]
                )
                lower = float(values_2035.min())
                upper = float(values_2035.max())
                ax.errorbar(
                    2035 + offset,
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
                reference = reference_path(scenario_data)
                ax.plot(
                    reference["year"],
                    reference["share_pct"],
                    color=MAIN_COLOR,
                    linewidth=1.8,
                    zorder=10,
                )

            ax.axhline(
                100.0 * baseline,
                linestyle="--",
                color=MAIN_COLOR,
                linewidth=1.0,
                alpha=0.6,
            )
            style_axis(ax)
            ax.set_ylim(1.0, 22.5)

    configure_grid(
        fig,
        axes,
        BASELINE_LABELS,
        BASELINES,
        "Military share of global GHG (%)",
        2036.05,
    )

    line_handles = [
        Line2D(
            [0],
            [0],
            color=DM_COLORS[d_m],
            lw=2.2,
            label=f"$d_m$ = {d_m * 100:.0f}%/yr",
        )
        for d_m in sorted(DM_VALUES, reverse=True)
    ]
    context_handles = [
        Patch(
            facecolor="#777777",
            alpha=0.16,
            edgecolor="none",
            label=(
                "Ribbons: full tested parameter range across "
                "$g$, $d_r$, and $\\varepsilon$"
            ),
        ),
        Line2D(
            [0],
            [0],
            color="#555555",
            marker="o",
            markersize=3.2,
            lw=0.9,
            label=(
                "2035 whiskers: full tested range across "
                "$g$, $d_r$, and $\\varepsilon$"
            ),
        ),
        Line2D(
            [0],
            [0],
            color=MAIN_COLOR,
            linestyle="--",
            lw=1.2,
            label="2025 baseline",
        ),
        Line2D(
            [0],
            [0],
            color=MAIN_COLOR,
            lw=2.5,
            label=(
                "Reference case: baseline = 5.5%, g = 3%, "
                "$d_m$ = $d_r$ = 1%/yr, $\\varepsilon$ = 0"
            ),
        ),
    ]
    fig.legend(
        handles=line_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.115),
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
        bbox_to_anchor=(0.5, 0.045),
        ncol=2,
        frameon=False,
        fontsize=9.6,
        columnspacing=1.15,
        handletextpad=0.5,
        labelspacing=0.35,
        borderaxespad=0.2,
    )
    fig.suptitle(
        "Military share of global GHG emissions (2025-2035)",
        fontsize=14,
        y=0.98,
    )
    fig.tight_layout(rect=[0.035, 0.145, 0.995, 0.955], h_pad=1.2, w_pad=0.8)
    fig.savefig(OUTPUT_DIR / "Fig_3.pdf", format="pdf", dpi=300)
    fig.savefig(OUTPUT_DIR / "Fig_3.png", format="png", dpi=300)
    plt.close(fig)


@mpl.rc_context(RC_PARAMS)
def make_fig_s3(main_data: pd.DataFrame) -> None:
    """Supplementary deterministic d_m by d_r line figure."""
    deterministic_data = main_data[
        np.isclose(main_data["g_world"], 0.03)
        & np.isclose(main_data["epsilon"], 0.0)
    ].copy()
    fig, axes = plt.subplots(
        3, 4, figsize=(11, 10.4), sharex=True, sharey=True
    )

    for row_index, baseline in enumerate(BASELINES):
        for column_index, scenario_code in enumerate(SCENARIOS):
            ax = axes[row_index, column_index]
            scenario_data = deterministic_data[
                np.isclose(deterministic_data["s_mil_2025_assumed"], baseline)
                & (deterministic_data["scenario_code"] == scenario_code)
            ].copy()

            for d_m in sorted(DM_VALUES, reverse=True):
                for d_r in sorted(DR_VALUES, reverse=True):
                    trajectory = scenario_data[
                        np.isclose(scenario_data["d_mil"], d_m)
                        & np.isclose(scenario_data["d_rest"], d_r)
                    ].sort_values("year")
                    ax.plot(
                        trajectory["year"],
                        trajectory["share_pct"],
                        color=DM_COLORS[d_m],
                        linestyle=DR_STYLES[d_r],
                        linewidth=1.0,
                        alpha=0.9,
                    )

            if np.isclose(baseline, 0.055):
                reference = reference_path(scenario_data)
                ax.plot(
                    reference["year"],
                    reference["share_pct"],
                    color=MAIN_COLOR,
                    linewidth=1.7,
                    zorder=10,
                )
            ax.axhline(
                100.0 * baseline,
                linestyle="--",
                color=MAIN_COLOR,
                linewidth=1.0,
                alpha=0.6,
            )
            style_axis(ax)
            ax.set_ylim(1.0, 22.5)

    configure_grid(
        fig,
        axes,
        BASELINE_LABELS,
        BASELINES,
        "Military share of global GHG (%)",
        2035,
    )

    dm_handles = [
        Line2D(
            [0],
            [0],
            color=DM_COLORS[d_m],
            lw=2.2,
            label=f"$d_m$ = {d_m * 100:.0f}%/yr",
        )
        for d_m in sorted(DM_VALUES, reverse=True)
    ]
    dr_handles = [
        Line2D(
            [0],
            [0],
            color="#333333",
            linestyle=DR_STYLES[d_r],
            lw=1.7,
            label=f"$d_r$ = {d_r * 100:.0f}%/yr",
        )
        for d_r in sorted(DR_VALUES, reverse=True)
    ]
    context_handles = [
        Line2D(
            [0],
            [0],
            color=MAIN_COLOR,
            linestyle="--",
            lw=1.2,
            label="2025 baseline",
        ),
        Line2D(
            [0],
            [0],
            color=MAIN_COLOR,
            lw=2.5,
            label=(
                "Reference case: baseline = 5.5%, g = 3%, "
                "$d_m$ = $d_r$ = 1%/yr, $\\varepsilon$ = 0"
            ),
        ),
    ]
    fig.legend(
        handles=dm_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.13),
        ncol=5,
        frameon=False,
        fontsize=10,
        columnspacing=1.35,
        handletextpad=0.5,
    )
    fig.legend(
        handles=dr_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.09),
        ncol=4,
        frameon=False,
        fontsize=10,
        columnspacing=1.35,
        handletextpad=0.5,
        title="Line styles ($g=3\\%$/yr and $\\varepsilon=0$)",
        title_fontsize=9.5,
    )
    fig.legend(
        handles=context_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.04),
        ncol=2,
        frameon=False,
        fontsize=9.6,
        columnspacing=1.3,
        handletextpad=0.5,
    )
    fig.suptitle(
        "Military share of global GHG emissions (2025-2035)",
        fontsize=14,
        y=0.98,
    )
    fig.tight_layout(rect=[0.035, 0.17, 0.995, 0.955], h_pad=1.2, w_pad=0.8)
    fig.savefig(OUTPUT_DIR / "Fig_S3.pdf", format="pdf", dpi=300)
    fig.savefig(OUTPUT_DIR / "Fig_S3.png", format="png", dpi=300)
    plt.close(fig)


@mpl.rc_context(RC_PARAMS)
def make_fig_4(main_data: pd.DataFrame, combined: pd.DataFrame) -> None:
    """Sensitivity-analysis share figure matching approved Figure 2."""
    main_variants = absolute_plot._fig_2_main_variants(main_data)
    ppp_variants = absolute_plot._fig_2_ppp_variants()
    main_variants["share_pct"] = 100.0 * main_variants["s_mil"]
    ppp_variants["share_pct"] = 100.0 * ppp_variants["s_mil"]
    combined = combined.copy()
    combined["share_pct"] = 100.0 * combined["s_mil"]

    rows = ["SA1", "SA2", "SA3"]
    row_labels = {
        "SA1": "SA1: AR6 pathways,\nfixed NATO/nonNATO GDP share",
        "SA2": "SA2: AR6 pathways,\nevolving NATO/nonNATO GDP share",
        "SA3": "SA3: Main model with\nPPP recalibration",
    }
    fig, axes = plt.subplots(
        3, 4, figsize=(11, 10.4), sharex=True, sharey=True
    )

    all_values: list[float] = []
    for row_index, row_name in enumerate(rows):
        for column_index, scenario_code in enumerate(SCENARIOS):
            ax = axes[row_index, column_index]

            if row_name in {"SA1", "SA2"}:
                source = combined[
                    (combined["combined_model"] == row_name)
                    & (combined["scenario_code"] == scenario_code)
                ]
                for pathway in absolute_plot.sensitivity_model.AR6_PATHWAYS:
                    trajectory = source[
                        source["projected_pathway"] == pathway
                    ].sort_values("year")
                    ax.plot(
                        trajectory["year"],
                        trajectory["share_pct"],
                        color=absolute_plot.sensitivity_model.PATHWAY_COLORS[
                            pathway
                        ],
                        linestyle="-",
                        linewidth=1.55,
                        zorder=2,
                    )
                    all_values.extend(trajectory["share_pct"].tolist())

            for key, config in absolute_plot.FIG_2_PARAMETER_SETS.items():
                trajectory = main_variants[
                    (main_variants["parameter_set"] == key)
                    & (main_variants["scenario_code"] == scenario_code)
                ].sort_values("year")
                ax.plot(
                    trajectory["year"],
                    trajectory["share_pct"],
                    color=MAIN_COLOR,
                    linestyle=config["linestyle"],
                    linewidth=2.0 if key == "reference" else 1.45,
                    alpha=1.0 if key == "reference" else 0.82,
                    zorder=4,
                )
                all_values.extend(trajectory["share_pct"].tolist())

            if row_name == "SA3":
                for key, config in absolute_plot.FIG_2_PARAMETER_SETS.items():
                    trajectory = ppp_variants[
                        (ppp_variants["parameter_set"] == key)
                        & (ppp_variants["scenario_code"] == scenario_code)
                    ].sort_values("year")
                    ax.plot(
                        trajectory["year"],
                        trajectory["share_pct"],
                        color=PPP_COLOR,
                        linestyle=config["linestyle"],
                        linewidth=2.05 if key == "reference" else 1.65,
                        alpha=1.0 if key == "reference" else 0.9,
                        zorder=5,
                    )
                    all_values.extend(trajectory["share_pct"].tolist())

            style_axis(ax)

    lower = max(0.0, np.floor(min(all_values)) - 0.5)
    upper = np.ceil(max(all_values)) + 0.5
    axes[0, 0].set_ylim(lower, upper)
    configure_grid(
        fig,
        axes,
        row_labels,
        rows,
        "Military share of global GHG (%)",
        2035,
    )

    colour_handles = [
        Line2D(
            [0],
            [0],
            color=absolute_plot.sensitivity_model.PATHWAY_COLORS[pathway],
            lw=2.0,
            label=pathway,
        )
        for pathway in absolute_plot.sensitivity_model.AR6_PATHWAYS
    ]
    colour_handles.append(
        Line2D(
            [0],
            [0],
            color=PPP_COLOR,
            lw=2.1,
            label="PPP recalibration (SA3)",
        )
    )
    style_handles = [
        Line2D(
            [0],
            [0],
            color="#333333",
            linestyle=config["linestyle"],
            lw=2.0 if key == "reference" else 1.6,
            label=config["label"],
        )
        for key, config in absolute_plot.FIG_2_PARAMETER_SETS.items()
    ]
    fig.legend(
        handles=colour_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.095),
        ncol=6,
        frameon=False,
        fontsize=9.2,
        columnspacing=1.05,
        handlelength=2.15,
        handletextpad=0.45,
        borderaxespad=0.2,
    )
    fig.legend(
        handles=style_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.047),
        ncol=4,
        frameon=False,
        fontsize=9.2,
        columnspacing=1.2,
        handlelength=2.7,
        handletextpad=0.5,
        borderaxespad=0.2,
        title=(
            "Line styles: main-model parameter sets (all baseline=5.5%); "
            "matched in PPP for SA3"
        ),
        title_fontsize=9.2,
    )
    fig.suptitle(
        "Sensitivity analysis of military share of global GHG emissions (2025-2035)",
        fontsize=14,
        y=0.98,
    )
    fig.tight_layout(rect=[0.035, 0.145, 0.995, 0.955], h_pad=1.2, w_pad=0.8)
    fig.savefig(OUTPUT_DIR / "Fig_4.pdf", format="pdf", dpi=300)
    fig.savefig(OUTPUT_DIR / "Fig_4.png", format="png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Fig_3, Fig_4, and Fig_S3 for the military-share "
            "results."
        )
    )
    parser.add_argument(
        "--sync-manuscript",
        action="store_true",
        help="Copy the generated PDFs into the manuscript folder.",
    )
    args = parser.parse_args()

    main_data = pd.read_csv(absolute_plot.INPUT_FILE)
    combined = pd.read_csv(absolute_plot.SENSITIVITY_INPUT_FILE)
    main_data = absolute_plot.filter_df_for_plot(
        main_data,
        plot_s=BASELINES,
        plot_g=G_VALUES,
        plot_dm=DM_VALUES,
        plot_dr=DR_VALUES,
        plot_eps=EPSILON_VALUES,
    ).copy()
    main_data["share_pct"] = 100.0 * main_data["s_mil"]

    make_fig_3(main_data)
    make_fig_s3(main_data)
    make_fig_4(main_data, combined)
    for name in ("Fig_3.pdf", "Fig_4.pdf", "Fig_S3.pdf"):
        if args.sync_manuscript:
            shutil.copy2(OUTPUT_DIR / name, MANUSCRIPT_DIR / name)
        print(OUTPUT_DIR / name)


if __name__ == "__main__":
    main()
