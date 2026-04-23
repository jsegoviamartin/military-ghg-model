"""Generate Fig_1 and Fig_S1 from generated_data_2025-2035.csv.

- Fig_1  -> main-text ribbon version (median by d_mil with 25th-75th ribbon)
- Fig_S1 -> supplementary full-line version (one line per growth x d_mil combination)

Both figures use the 2025-2035 simulation output produced by
`military_emissions_model.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter


# ============================================================================
# USER OPTIONS
# ============================================================================

INPUT_FILE = "generated_data_2025-2035.csv"
OUTPUT_DIR = "."

# Plot filters. Set any of these to None to include all simulated values.
PLOT_S_MIL_VALUES = [0.033, 0.055, 0.070]
PLOT_GROWTH_VALUES = [0.01, 0.02, 0.03, 0.04]
PLOT_D_MIL_VALUES = [0.00, -0.01, -0.03, -0.05, -0.07]
PLOT_D_REST_VALUES = [-0.01, -0.03, -0.05, -0.07]
PLOT_EPSILON_VALUES = [0.0, 0.009, 0.015, 0.02]

CENTRAL_PARAMS = {
    "s_mil_2025_assumed": 0.055,
    "g_world": 0.03,
    "d_mil": -0.01,
    "d_rest": -0.01,
    "epsilon": 0.015,
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
    ordered = [round(v, 4) for v in sorted(values)]
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


def make_fig_s2(df_plot_2035: pd.DataFrame, output_dir: Path) -> None:
    """Supplementary full-line figure: one line per growth x d_mil combination."""
    fig, axes = plt.subplots(
        nrows=len(PLOT_S_MIL_VALUES),
        ncols=len(SCENARIO_ORDER),
        figsize=(11, 11),
        sharex=True,
        sharey=True,
    )

    d_mil_vals = sorted(np.round(df_plot_2035["d_mil"].unique(), 4))
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
            label="Central case: baseline=5.5%, g=3%, dₘ=-1%, dᵣ=-1%, ε=1.5%",
        )
    )

    fig.legend(handles=legend_elements, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Absolute Military GHG Emissions (2025–2035)", fontsize=14)
    fig.tight_layout(rect=[0, 0.12, 1, 0.95])

    fig.savefig(output_dir / "Fig_S1.pdf", format="pdf", dpi=300)
    fig.savefig(output_dir / "Fig_S1.png", format="png", dpi=300)
    plt.close(fig)



def make_fig_2(df_plot_2035: pd.DataFrame, output_dir: Path) -> None:
    """Main-text ribbon figure: median by d_mil with 25th-75th ribbon."""
    fig, axes = plt.subplots(
        nrows=len(PLOT_S_MIL_VALUES),
        ncols=len(SCENARIO_ORDER),
        figsize=(11, 11),
        sharex=True,
        sharey=True,
    )

    d_mil_vals = sorted(np.round(df_plot_2035["d_mil"].unique(), 4))
    d_m_colors = build_color_map(d_mil_vals)

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

            for d_m in d_mil_vals:
                df_dm = df_scen[np.isclose(df_scen["d_mil"], d_m)].copy()
                if df_dm.empty:
                    continue

                summary_dm = (
                    df_dm.groupby("year")["E_mil_Gt"]
                    .agg(
                        q25=lambda x: np.quantile(x, 0.25),
                        q50=lambda x: np.quantile(x, 0.50),
                        q75=lambda x: np.quantile(x, 0.75),
                    )
                    .reset_index()
                )

                ax.fill_between(
                    summary_dm["year"],
                    summary_dm["q25"],
                    summary_dm["q75"],
                    color=d_m_colors[d_m],
                    alpha=0.18,
                    linewidth=0,
                )
                ax.plot(
                    summary_dm["year"],
                    summary_dm["q50"],
                    color=d_m_colors[d_m],
                    linewidth=1.6,
                    alpha=0.95,
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
            Line2D([0], [0], color=d_m_colors[d_m], lw=2.5, label=f"dₘ = {d_m * 100:.0f}%/yr (median)")
        )
    legend_elements.append(
        Patch(facecolor="grey", alpha=0.18, edgecolor="none", label="Ribbon: 25th–75th pct. across dᵣ, g, ε")
    )
    legend_elements.append(
        Line2D([0], [0], color="black", linestyle="--", lw=1.5, label="2025 baseline")
    )
    legend_elements.append(
        Line2D(
            [0],
            [0],
            color="black",
            lw=2.5,
            label="Central case: baseline=5.5%, g=3%, dₘ=-1%, dᵣ=-1%, ε=1.5%",
        )
    )

    fig.legend(handles=legend_elements, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Absolute Military GHG Emissions (2025–2035)", fontsize=14)
    fig.tight_layout(rect=[0, 0.12, 1, 0.95])

    fig.savefig(output_dir / "Fig_1.pdf", format="pdf", dpi=300)
    fig.savefig(output_dir / "Fig_1.png", format="png", dpi=300)
    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    input_path = Path(INPUT_FILE)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    df_plot = filter_df_for_plot(
        df,
        plot_s=PLOT_S_MIL_VALUES,
        plot_g=PLOT_GROWTH_VALUES,
        plot_dm=PLOT_D_MIL_VALUES,
        plot_dr=PLOT_D_REST_VALUES,
        plot_eps=PLOT_EPSILON_VALUES,
    )

    make_fig_2(df_plot, output_dir)
    make_fig_s2(df_plot, output_dir)


if __name__ == "__main__":
    main()
