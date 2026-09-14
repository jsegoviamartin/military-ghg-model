"""Generate all figures for the absolute annual military-emissions results.

- Fig_1: main-model ribbon figure.
- Fig_2: combined SA1, SA2, and SA3 sensitivity-analysis figure.
- Fig_S1: supplementary full-line main-model figure.

Fig_1 and Fig_S1 use the 2025-2035 output from
``military_emissions_model.py``. Fig_2 uses the combined sensitivity output
from ``military_emissions_model_SA1_SA2_SA3.py``.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import shutil
import sys
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter


# ============================================================================
# USER OPTIONS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "generated" / "generated_data_2025-2035.csv"
SENSITIVITY_INPUT_FILE = (
    PROJECT_ROOT
    / "output"
    / "three_sensitivity_analyses"
    / "Combined"
    / "combined_actual_projected_scenarios_2025-2035.csv"
)
PPP_INPUT_FILE = PROJECT_ROOT / "data" / "inputs" / "IMF_WEO_PPP_2025.csv"
OUTPUT_DIR = PROJECT_ROOT / "output" / "pdf" / "absolute_military_emissions"
MANUSCRIPT_DIR = PROJECT_ROOT / "manuscript"


def load_sensitivity_model():
    """Load the adjacent sensitivity module without relying on the working directory."""
    module_path = (
        Path(__file__).resolve().parent
        / "military_emissions_model_SA1_SA2_SA3.py"
    )
    spec = importlib.util.spec_from_file_location("military_sensitivity", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load sensitivity module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sensitivity_model = load_sensitivity_model()

# Plot filters. Set any of these to None to include all simulated values.
PLOT_S_MIL_VALUES = [0.033, 0.055, 0.070]
PLOT_GROWTH_VALUES = [0.01, 0.02, 0.03, 0.04]
PLOT_D_MIL_VALUES = [0.00, 0.01, 0.03, 0.05, 0.07]
PLOT_D_REST_VALUES = [0.01, 0.03, 0.05, 0.07]
PLOT_EPSILON_VALUES = [0.0, 0.009, 0.015, 0.02]

CENTRAL_PARAMS = {
    "s_mil_2025_assumed": 0.055,
    "g_world": 0.03,
    "d_mil": 0.01,
    "d_rest": 0.01,
    "epsilon": 0.0,
}

FIG_2_PARAMETER_SETS = {
    "reference": {
        "g": 0.03,
        "d_m": 0.01,
        "linestyle": "-",
        "label": "Reference: g=3%, $d_m$=1%/yr",
    },
    "low_growth_strong_decarbonisation": {
        "g": 0.01,
        "d_m": 0.07,
        "linestyle": "--",
        "label": "g=1%, $d_m$=7%/yr",
    },
    "moderate_growth_decarbonisation": {
        "g": 0.02,
        "d_m": 0.05,
        "linestyle": ":",
        "label": "g=2%, $d_m$=5%/yr",
    },
    "higher_growth_no_decarbonisation": {
        "g": 0.04,
        "d_m": 0.00,
        "linestyle": "-.",
        "label": "g=4%, $d_m$=0%/yr",
    },
}

BASELINE_LABELS = {
    0.033: "Baseline = 3.3%",
    0.055: "Baseline = 5.5%",
    0.070: "Baseline = 7.0%",
}

SCENARIO_ORDER = ["S0", "S1", "S2", "S3"]
SCENARIO_TITLES = {
    "S0": "Baseline",
    "S1": "NATO→3.5%, non-NATO holds",
    "S2": "NATO→3.5%, non-NATO→3.5%",
    "S3": "NATO→5%, non-NATO→3.5%",
}

COLOR_PALETTE = [
    "tab:blue",
    "tab:green",
    "tab:orange",
    "tab:red",
    "tab:brown",
    "tab:purple",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
]

HARMONIZED_SCENARIO_TITLES = {
    "S0": "S0\nBaseline",
    "S1": "S1\nNATO→3.5%, non-NATO holds",
    "S2": "S2\nNATO→3.5%, non-NATO→3.5%",
    "S3": "S3\nNATO→5%, non-NATO→3.5%",
}

HARMONIZED_COLOR_PALETTE = [
    "#0072B2",
    "#009E73",
    "#E69F00",
    "#D55E00",
    "#7A3E2D",
]

HARMONIZED_RC_PARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

LINE_STYLE_LIST_SHORT = ["-", "--", "-.", ":"]


# ============================================================================
# HELPERS
# ============================================================================


def round_list(values: Iterable[float] | None) -> list[float] | None:
    if values is None:
        return None
    return [round(v, 4) for v in values]



def filter_df_for_plot(
    df: pd.DataFrame,
    plot_s: list[float] | None = None,
    plot_g: list[float] | None = None,
    plot_dm: list[float] | None = None,
    plot_dr: list[float] | None = None,
    plot_eps: list[float] | None = None,
) -> pd.DataFrame:
    out = df.copy()

    if plot_s is not None:
        out = out[out["s_mil_2025_assumed"].round(4).isin(round_list(plot_s))]
    if plot_g is not None:
        out = out[out["g_world"].round(4).isin(round_list(plot_g))]
    if plot_dm is not None:
        out = out[out["d_mil"].round(4).isin(round_list(plot_dm))]
    if plot_dr is not None:
        out = out[out["d_rest"].round(4).isin(round_list(plot_dr))]
    if plot_eps is not None:
        out = out[out["epsilon"].round(4).isin(round_list(plot_eps))]

    return out



def build_color_map(values: Iterable[float]) -> dict[float, str]:
    # Positive d values replace the former negative convention. Reverse the
    # numerical sort so each physical trajectory retains its original colour.
    ordered = [round(v, 4) for v in sorted(values, reverse=True)]
    return {value: COLOR_PALETTE[i] for i, value in enumerate(ordered)}



def build_style_map(values: Iterable[float], style_list: list) -> dict[float, str | tuple]:
    ordered = [round(v, 4) for v in sorted(values)]
    return {value: style_list[i] for i, value in enumerate(ordered)}



def add_common_axis_formatting(ax: plt.Axes) -> None:
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)



def get_central_path(df_scen: pd.DataFrame) -> pd.DataFrame:
    df_central = df_scen[
        (df_scen["s_mil_2025_assumed"] == CENTRAL_PARAMS["s_mil_2025_assumed"])
        & (df_scen["g_world"] == CENTRAL_PARAMS["g_world"])
        & (df_scen["d_mil"] == CENTRAL_PARAMS["d_mil"])
        & (df_scen["d_rest"] == CENTRAL_PARAMS["d_rest"])
        & (df_scen["epsilon"] == CENTRAL_PARAMS["epsilon"])
    ].sort_values("year")

    if df_central.empty:
        return df_central

    return (
        df_central.groupby("year", as_index=False)["E_mil_Gt"]
        .mean()
        .sort_values("year")
    )


# ============================================================================
# FIGURE BUILDERS
# ============================================================================


def make_fig_s1(df_plot_2035: pd.DataFrame, output_dir: Path) -> None:
    """Supplementary full-line figure: one line per growth x d_mil combination."""
    fig, axes = plt.subplots(
        nrows=len(PLOT_S_MIL_VALUES),
        ncols=len(SCENARIO_ORDER),
        figsize=(11, 11),
        sharex=True,
        sharey=True,
    )

    d_mil_vals = sorted(
        np.round(df_plot_2035["d_mil"].unique(), 4), reverse=True
    )
    growth_vals = sorted(np.round(df_plot_2035["g_world"].unique(), 4))

    d_m_colors = build_color_map(d_mil_vals)
    g_styles = build_style_map(growth_vals, LINE_STYLE_LIST_SHORT)

    panel_counter = 0

    for i, baseline in enumerate(PLOT_S_MIL_VALUES):
        for j, scenario_code in enumerate(SCENARIO_ORDER):
            ax = axes[i, j]
            df_scen = df_plot_2035[
                (df_plot_2035["s_mil_2025_assumed"] == baseline)
                & (df_plot_2035["scenario_code"] == scenario_code)
            ].copy()

            if df_scen.empty:
                ax.set_axis_off()
                panel_counter += 1
                continue

            for (g, d_m), df_combo in df_scen.groupby(["g_world", "d_mil"]):
                mil_path = (
                    df_combo.sort_values("year")
                    .groupby("year", as_index=False)["E_mil_Gt"]
                    .mean()
                )

                ax.plot(
                    mil_path["year"],
                    mil_path["E_mil_Gt"],
                    color=d_m_colors[round(d_m, 4)],
                    linestyle=g_styles[round(g, 4)],
                    linewidth=1.0,
                    alpha=0.9,
                )

            if baseline == CENTRAL_PARAMS["s_mil_2025_assumed"]:
                central_path = get_central_path(df_scen)
                if not central_path.empty:
                    ax.plot(
                        central_path["year"],
                        central_path["E_mil_Gt"],
                        color="black",
                        linewidth=1.5,
                        zorder=10,
                    )

            baseline_mil = df_scen.loc[df_scen["year"] == 2025, "E_mil_Gt"].median()
            ax.axhline(
                baseline_mil,
                linestyle="--",
                color="black",
                linewidth=1.0,
                alpha=0.6,
            )

            add_common_axis_formatting(ax)

            if j == 0:
                ax.set_ylabel(f"{BASELINE_LABELS[baseline]}\nMilitary Emissions (GtCO₂e)")
            if i == 0:
                ax.set_title(SCENARIO_TITLES[scenario_code], fontsize=11)

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
        ax.set_xlabel("Year")

    legend_elements: list = []
    for d_m in d_mil_vals:
        legend_elements.append(
            Line2D([0], [0], color=d_m_colors[d_m], lw=3, label=f"dₘ = {d_m * 100:.0f}%/yr")
        )
    for g in growth_vals:
        legend_elements.append(
            Line2D([0], [0], color="black", linestyle=g_styles[g], lw=2, label=f"Growth = {g * 100:.0f}%")
        )
    legend_elements.append(
        Line2D([0], [0], color="black", linestyle="--", lw=1.5, label="2025 baseline")
    )
    legend_elements.append(
        Line2D(
            [0],
            [0],
            color="black",
            lw=3,
            label="Reference case: baseline=5.5%, g=3%, dₘ=1%, dᵣ=1%, ε=0%",
        )
    )

    fig.legend(handles=legend_elements, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Absolute Military GHG Emissions (2025–2035)", fontsize=14)
    fig.tight_layout(rect=[0, 0.12, 1, 0.95])

    fig.savefig(output_dir / "Fig_S1.pdf", format="pdf", dpi=300)
    fig.savefig(output_dir / "Fig_S1.png", format="png", dpi=300)
    plt.close(fig)



@mpl.rc_context(HARMONIZED_RC_PARAMS)
def make_fig_1(df_plot_2035: pd.DataFrame, output_dir: Path) -> None:
    """Plot deterministic g=3% pathways with full-range ribbons and whiskers."""
    fig, axes = plt.subplots(
        nrows=len(PLOT_S_MIL_VALUES),
        ncols=len(SCENARIO_ORDER),
        figsize=(11, 10.4),
        sharex=True,
        sharey=True,
    )

    d_mil_vals = sorted(
        np.round(df_plot_2035["d_mil"].unique(), 4), reverse=True
    )
    d_m_colors = {
        value: HARMONIZED_COLOR_PALETTE[index % len(HARMONIZED_COLOR_PALETTE)]
        for index, value in enumerate(d_mil_vals)
    }

    panel_counter = 0
    range_offsets = np.linspace(0.18, 0.82, len(d_mil_vals))

    for i, baseline in enumerate(PLOT_S_MIL_VALUES):
        for j, scenario_code in enumerate(SCENARIO_ORDER):
            ax = axes[i, j]
            df_scen = df_plot_2035[
                (df_plot_2035["s_mil_2025_assumed"] == baseline)
                & (df_plot_2035["scenario_code"] == scenario_code)
            ].copy()

            if df_scen.empty:
                ax.set_axis_off()
                panel_counter += 1
                continue

            ax.axvline(2035.07, color="#BFBFBF", linewidth=0.5, zorder=0)

            for offset, d_m in zip(range_offsets, d_mil_vals):
                df_dm = df_scen[np.isclose(df_scen["d_mil"], d_m)].copy()
                if df_dm.empty:
                    continue

                df_dm = df_dm.drop_duplicates(
                    subset=["g_world", "d_mil", "year", "E_mil_Gt"]
                )
                envelope = (
                    df_dm.groupby("year", as_index=False)["E_mil_Gt"]
                    .agg(minimum="min", maximum="max")
                    .sort_values("year")
                )
                central_growth = (
                    df_dm[np.isclose(df_dm["g_world"], 0.03)]
                    .groupby("year", as_index=False)["E_mil_Gt"]
                    .mean()
                    .sort_values("year")
                )

                ax.fill_between(
                    envelope["year"],
                    envelope["minimum"],
                    envelope["maximum"],
                    color=d_m_colors[d_m],
                    alpha=0.085,
                    linewidth=0,
                    zorder=1,
                )
                ax.plot(
                    central_growth["year"],
                    central_growth["E_mil_Gt"],
                    color=d_m_colors[d_m],
                    linewidth=1.55,
                    alpha=1.0,
                    zorder=3,
                )

                values_2035 = df_dm.loc[df_dm["year"] == 2035, "E_mil_Gt"]
                centre = float(
                    central_growth.loc[
                        central_growth["year"] == 2035, "E_mil_Gt"
                    ].iloc[0]
                )
                lower = float(values_2035.min())
                upper = float(values_2035.max())
                ax.errorbar(
                    2035 + offset,
                    centre,
                    yerr=np.array([[centre - lower], [upper - centre]]),
                    fmt="o",
                    color=d_m_colors[d_m],
                    ecolor=d_m_colors[d_m],
                    elinewidth=0.9,
                    capsize=2.1,
                    capthick=0.9,
                    markersize=2.8,
                    markeredgewidth=0,
                    zorder=5,
                )

            if baseline == CENTRAL_PARAMS["s_mil_2025_assumed"]:
                central_path = get_central_path(df_scen)
                if not central_path.empty:
                    ax.plot(
                        central_path["year"],
                        central_path["E_mil_Gt"],
                        color="black",
                        linewidth=1.8,
                        zorder=10,
                    )

            baseline_mil = df_scen.loc[df_scen["year"] == 2025, "E_mil_Gt"].median()
            ax.axhline(
                baseline_mil,
                linestyle="--",
                color="black",
                linewidth=1.0,
                alpha=0.6,
            )

            add_common_axis_formatting(ax)
            ax.set_xlim(2025, 2036.05)
            ax.set_ylim(0.6, 10.0)
            ax.set_xticks([2025, 2030, 2035])
            ax.tick_params(axis="both", labelsize=10, width=0.5)

            if j == 0:
                ax.set_ylabel(
                    f"{BASELINE_LABELS[baseline]}\nMilitary GHG emissions (GtCO$_2$e/yr)",
                    fontsize=10,
                )
            if i == 0:
                ax.set_title(
                    HARMONIZED_SCENARIO_TITLES[scenario_code],
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
            color=d_m_colors[d_m],
            lw=2.2,
            label=f"$d_m$ = {d_m * 100:.0f}%/yr",
        )
        for d_m in d_mil_vals
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
            label="2035 whiskers: full tested range across g = 1-4%/yr",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            lw=1.2,
            label="2025 baseline",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            lw=2.5,
            label="Main model reference case: baseline = 5.5%, g = 3%, $d_m$ = 1%/yr",
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
        fontsize=10,
        columnspacing=1.25,
        handletextpad=0.5,
        labelspacing=0.35,
        borderaxespad=0.2,
    )
    fig.suptitle("Absolute military GHG emissions (2025-2035)", fontsize=14, y=0.98)
    fig.tight_layout(
        rect=[0.035, 0.145, 0.995, 0.955], h_pad=1.2, w_pad=0.8
    )

    fig.savefig(output_dir / "Fig_1.pdf", format="pdf", dpi=300)
    fig.savefig(output_dir / "Fig_1.png", format="png", dpi=300)
    plt.close(fig)


def _fig_2_main_variants(main_data: pd.DataFrame) -> pd.DataFrame:
    """Select the four approved main-model parameter sets for Figure 2."""
    keep = (
        np.isclose(main_data["s_mil_2025_assumed"], 0.055)
        & np.isclose(main_data["d_rest"], 0.01)
        & np.isclose(main_data["epsilon"], 0.0)
        & (main_data["year"] <= 2035)
    )
    data = main_data.loc[keep].copy()
    frames = []
    for key, config in FIG_2_PARAMETER_SETS.items():
        selected = data[
            np.isclose(data["g_world"], config["g"])
            & np.isclose(data["d_mil"], config["d_m"])
        ].copy()
        if selected.empty:
            raise ValueError(
                f"Figure 2 parameter set not found in main-model output: {key}"
            )
        selected = selected.drop_duplicates(
            subset=["scenario_code", "year", "g_world", "d_mil", "E_mil_Gt"]
        )
        selected["parameter_set"] = key
        frames.append(selected)
    return pd.concat(frames, ignore_index=True)


def _fig_2_ppp_variants() -> pd.DataFrame:
    """Simulate PPP counterparts of the four approved parameter sets."""
    calibration, _ = sensitivity_model.load_imf_ppp_2025(PPP_INPUT_FILE)
    params = sensitivity_model.ModelParams(
        b_global_2025=sensitivity_model.GLOBAL_MILITARY_BURDEN_2025,
        m_NATO_2025=sensitivity_model.NATO_MILITARY_BURDEN_2025,
        s_NATO_GDP=calibration["s_NATO_GDP_2025_PPP"],
        GDP_world_2025=calibration["GDP_world_2025_PPP_trillion"],
        s_mil_2025=0.055,
        E_mil_2025=sensitivity_model.WORLD_GHG_2025 * 0.055,
        start_year=sensitivity_model.START_YEAR,
        end_year=2035,
    )
    calibrated = sensitivity_model.calibrate_intensities(params)
    scenarios = sensitivity_model.make_scenarios(calibrated["m_nonNATO_2025"])

    frames = []
    for key, config in FIG_2_PARAMETER_SETS.items():
        for scenario in scenarios:
            frame = sensitivity_model.simulate_timeseries(
                params=params,
                scenario=scenario,
                g_world=config["g"],
                d_mil=config["d_m"],
                d_rest=0.01,
                epsilon=0.0,
            )
            frame["parameter_set"] = key
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


@mpl.rc_context(HARMONIZED_RC_PARAMS)
def make_fig_2(
    combined: pd.DataFrame,
    main_data: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Generate the approved expanded sensitivity analysis as Figure 2."""
    main_variants = _fig_2_main_variants(main_data)
    ppp_variants = _fig_2_ppp_variants()

    main_color = "#111111"
    ppp_color = "#7B2CBF"
    scenarios = ["S0", "S1", "S2", "S3"]
    rows = ["SA1", "SA2", "SA3"]
    column_titles = {
        "S0": "S0\nBaseline",
        "S1": "S1\nNATO→3.5%, non-NATO holds",
        "S2": "S2\nNATO→3.5%, non-NATO→3.5%",
        "S3": "S3\nNATO→5%, non-NATO→3.5%",
    }
    row_titles = {
        "SA1": "SA1: AR6 pathways,\nfixed NATO/nonNATO GDP share",
        "SA2": "SA2: AR6 pathways,\nevolving NATO/nonNATO GDP share",
        "SA3": "SA3: Main model with\nPPP recalibration",
    }

    fig, axes = plt.subplots(
        3, 4, figsize=(11, 10.4), sharex=True, sharey=True
    )
    panel_counter = 0

    for row_index, row_name in enumerate(rows):
        for column_index, scenario_code in enumerate(scenarios):
            ax = axes[row_index, column_index]

            if row_name in {"SA1", "SA2"}:
                source = combined[
                    (combined["combined_model"] == row_name)
                    & (combined["scenario_code"] == scenario_code)
                ]
                for pathway in sensitivity_model.AR6_PATHWAYS:
                    trajectory = source[
                        source["projected_pathway"] == pathway
                    ].sort_values("year")
                    if not trajectory.empty:
                        ax.plot(
                            trajectory["year"],
                            trajectory["E_mil_Gt"],
                            color=sensitivity_model.PATHWAY_COLORS[pathway],
                            linestyle="-",
                            linewidth=1.55,
                            zorder=2,
                        )

            for key, config in FIG_2_PARAMETER_SETS.items():
                trajectory = main_variants[
                    (main_variants["parameter_set"] == key)
                    & (main_variants["scenario_code"] == scenario_code)
                ].sort_values("year")
                ax.plot(
                    trajectory["year"],
                    trajectory["E_mil_Gt"],
                    color=main_color,
                    linestyle=config["linestyle"],
                    linewidth=2.0 if key == "reference" else 1.45,
                    alpha=1.0 if key == "reference" else 0.82,
                    zorder=4,
                )

            if row_name == "SA3":
                for key, config in FIG_2_PARAMETER_SETS.items():
                    trajectory = ppp_variants[
                        (ppp_variants["parameter_set"] == key)
                        & (ppp_variants["scenario_code"] == scenario_code)
                    ].sort_values("year")
                    ax.plot(
                        trajectory["year"],
                        trajectory["E_mil_Gt"],
                        color=ppp_color,
                        linestyle=config["linestyle"],
                        linewidth=2.05 if key == "reference" else 1.65,
                        alpha=1.0 if key == "reference" else 0.9,
                        zorder=5,
                    )

            sensitivity_model._style_axis(ax)
            ax.grid(False)
            ax.grid(True, alpha=0.2)
            ax.tick_params(axis="both", labelsize=10, width=0.5)
            ax.set_xlim(2025, 2035)
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
            f"{row_titles[row_name]}\nMilitary GHG emissions (GtCO$_2$e/yr)",
            fontsize=10,
        )

    for column_index, scenario_code in enumerate(scenarios):
        axes[0, column_index].set_title(
            column_titles[scenario_code], fontsize=11, pad=5, linespacing=1.0
        )
    for ax in axes[-1, :]:
        ax.set_xlabel("Year", fontsize=10)

    colour_handles = [
        Line2D(
            [0],
            [0],
            color=sensitivity_model.PATHWAY_COLORS[pathway],
            lw=2.0,
            label=pathway,
        )
        for pathway in sensitivity_model.AR6_PATHWAYS
    ]
    colour_handles.append(
        Line2D(
            [0],
            [0],
            color=ppp_color,
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
        for key, config in FIG_2_PARAMETER_SETS.items()
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
        "Sensitivity analysis of absolute military GHG emissions (2025-2035)",
        fontsize=14,
        y=0.98,
    )
    fig.tight_layout(rect=[0.035, 0.145, 0.995, 0.955], h_pad=1.2, w_pad=0.8)

    fig.savefig(output_dir / "Fig_2.pdf", format="pdf", dpi=300)
    fig.savefig(output_dir / "Fig_2.png", format="png", dpi=300)
    plt.close(fig)


def sync_manuscript_figures(output_dir: Path) -> None:
    """Copy approved outputs to the filenames currently used by the manuscript."""
    manuscript_targets = {
        "Fig_1.pdf": "Fig_1.pdf",
        "Fig_2.pdf": "Fig_2.pdf",
        "Fig_S1.pdf": "Fig_S1.pdf",
    }
    for source_name, target_name in manuscript_targets.items():
        shutil.copy2(output_dir / source_name, MANUSCRIPT_DIR / target_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Fig_1, Fig_2, and Fig_S1 for the absolute annual "
            "military GHG emissions results."
        )
    )
    parser.add_argument("--main-data", type=Path, default=INPUT_FILE)
    parser.add_argument(
        "--sensitivity-data", type=Path, default=SENSITIVITY_INPUT_FILE
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--sync-manuscript",
        action="store_true",
        help="Copy the generated PDFs into the manuscript folder.",
    )
    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    args = parse_args()
    input_path = args.main_data
    sensitivity_input_path = args.sensitivity_data
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Main-model data not found: {input_path}. Run "
            "model/military_emissions_model.py first."
        )
    if not sensitivity_input_path.exists():
        raise FileNotFoundError(
            f"Combined sensitivity data not found: {sensitivity_input_path}. "
            "Run model/military_emissions_model_SA1_SA2_SA3.py first."
        )

    df = pd.read_csv(input_path)
    combined = pd.read_csv(sensitivity_input_path)

    df_plot = filter_df_for_plot(
        df,
        plot_s=PLOT_S_MIL_VALUES,
        plot_g=PLOT_GROWTH_VALUES,
        plot_dm=PLOT_D_MIL_VALUES,
        plot_dr=PLOT_D_REST_VALUES,
        plot_eps=PLOT_EPSILON_VALUES,
    )

    make_fig_1(df_plot, output_dir)
    make_fig_2(combined, df, output_dir)
    make_fig_s1(df_plot, output_dir)

    if args.sync_manuscript:
        sync_manuscript_figures(output_dir)

    for name in ("Fig_1.pdf", "Fig_2.pdf", "Fig_S1.pdf"):
        print(f"Saved: {output_dir / name}")


if __name__ == "__main__":
    main()
