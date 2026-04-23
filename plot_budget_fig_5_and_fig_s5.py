"""
Generate Fig_5 and Fig_S5 from generated_data_2025-2050.csv,

- Fig_5  -> 2x2 panel version
- Fig_S5 -> denser version
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================================
# USER OPTIONS
# ============================================================================

INPUT_FILE = "generated_data_2025-2050.csv"
OUTPUT_DIR = "."

PLOT_S_MIL_VALUES = [0.033, 0.055, 0.070]
PLOT_GROWTH_VALUES = [0.01, 0.02, 0.03, 0.04]
PLOT_D_MIL_VALUES = [0.00, -0.01, -0.03, -0.05, -0.07]
PLOT_D_REST_VALUES = [-0.01, -0.03, -0.05, -0.07]
PLOT_EPSILON_VALUES = [0.0, 0.009, 0.015, 0.02]

BASELINE_FOCUS = 0.055
FIXED_D_REST = -0.01
FIXED_EPSILON = 0.015

BUDGET_1P5 = 142.0
BUDGET_2C = 892.0

SCENARIO_ORDER = ["S0", "S1", "S2", "S3"]
SCENARIO_TITLES = {
    "S0": "S0: Baseline",
    "S1": "S1: NATO→3.5%, non-NATO holds",
    "S2": "S2: NATO→3.5%, non-NATO→3.5%",
    "S3": "S3: NATO→5%, non-NATO→3.5%",
}

BAR_COLOR = "#4C72B0"
MARKER_COLOR = "darkred"


# ============================================================================
# HELPERS
# ============================================================================

def round_list(vals):
    if vals is None:
        return None
    return [round(v, 4) for v in vals]


def filter_df_for_plot(
    df,
    plot_s=None,
    plot_g=None,
    plot_dm=None,
    plot_dr=None,
    plot_eps=None,
):
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


def build_budget_summary_from_generated_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(
        [
            "scenario_code",
            "scenario",
            "s_mil_2025_assumed",
            "g_world",
            "d_mil",
            "d_rest",
            "epsilon",
            "year",
        ]
    )

    df["cum_E_mil_Gt"] = (
        df.groupby(
            [
                "scenario_code",
                "scenario",
                "s_mil_2025_assumed",
                "g_world",
                "d_mil",
                "d_rest",
                "epsilon",
            ]
        )["E_mil_Gt"]
        .cumsum()
    )

    df_2050 = df[df["year"] == 2050].copy()

    df_2050["budget_1p5_Gt"] = BUDGET_1P5
    df_2050["budget_2C_Gt"] = BUDGET_2C
    df_2050["pct_1p5_used"] = 100 * df_2050["cum_E_mil_Gt"] / BUDGET_1P5
    df_2050["pct_2C_used"] = 100 * df_2050["cum_E_mil_Gt"] / BUDGET_2C
    df_2050["pct_1p5_remaining"] = 100 - df_2050["pct_1p5_used"]
    df_2050["pct_2C_remaining"] = 100 - df_2050["pct_2C_used"]

    keep_cols = [
        "scenario_code",
        "scenario",
        "s_mil_2025_assumed",
        "g_world",
        "d_mil",
        "d_rest",
        "epsilon",
        "cum_E_mil_Gt",
        "budget_1p5_Gt",
        "budget_2C_Gt",
        "pct_1p5_used",
        "pct_2C_used",
        "pct_1p5_remaining",
        "pct_2C_remaining",
    ]

    return df_2050[keep_cols].copy()


def load_focus_data(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    # Apply the same plotting filters as the original script
    df_plot = filter_df_for_plot(
        df,
        plot_s=PLOT_S_MIL_VALUES,
        plot_g=PLOT_GROWTH_VALUES,
        plot_dm=PLOT_D_MIL_VALUES,
        plot_dr=PLOT_D_REST_VALUES,
        plot_eps=PLOT_EPSILON_VALUES,
    )

    # Build 2050 budget summary AFTER filtering, as in the original workflow
    df_budget = build_budget_summary_from_generated_data(df_plot)

    # Focus on the exact subset used by the original Plot 5 / Plot 5.b
    df_focus = df_budget[
        (df_budget["s_mil_2025_assumed"] == BASELINE_FOCUS)
        & (df_budget["d_rest"] == FIXED_D_REST)
        & (df_budget["epsilon"] == FIXED_EPSILON)
    ].copy()

    if df_focus.empty:
        raise ValueError(
            "No rows left after filtering. Check BASELINE_FOCUS, FIXED_D_REST, FIXED_EPSILON."
        )

    df_focus["combo_label"] = (
        "g=" + (100 * df_focus["g_world"]).round().astype(int).astype(str) + "%, "
        + "dₘ=" + (100 * df_focus["d_mil"]).round().astype(int).astype(str) + "%"
    )

    return df_focus


def style_axes(ax):
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ============================================================================
# FIGURE BUILDERS
# ============================================================================

def make_fig_s4(df_focus: pd.DataFrame, output_dir: Path) -> None:
    """Supplementary dense single-panel version."""
    df_single = df_focus.copy().sort_values("pct_1p5_used", ascending=True)

    x = np.arange(len(df_single))
    used = df_single["pct_1p5_used"].to_numpy()
    remaining = df_single["pct_1p5_remaining"].to_numpy()
    used_2c = df_single["pct_2C_used"].to_numpy()
    labels = (
        df_single["scenario_code"].astype(str)
        + ", "
        + df_single["combo_label"].astype(str)
    )

    fig, ax = plt.subplots(figsize=(15, 6))

    ax.bar(x, used, color=BAR_COLOR, label="1.5°C budget used", zorder=2)
    ax.bar(
        x,
        remaining,
        bottom=used,
        color=BAR_COLOR,
        alpha=0.25,
        label="budget remaining",
        zorder=2,
    )
    ax.scatter(
        x,
        used_2c,
        marker="D",
        s=70,
        color=MARKER_COLOR,
        label="2°C budget used",
        zorder=4,
    )
    ax.axhline(100, linestyle="--", linewidth=1, color="grey")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_ylabel("Carbon Budget Share (%)")
    ax.set_title(
        "Carbon Budget Depletion by 2050\n"
        "Baseline Military Share = 5.5% | "
        f"dᵣ = {FIXED_D_REST*100:.0f}%/yr | "
        f"ε = {FIXED_EPSILON*100:.1f}%"
    )
    ax.set_ylim(0, max(110, used.max() * 1.05))
    style_axes(ax)
    ax.legend(frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(output_dir / "Fig_S4.pdf", format="pdf", dpi=300)
    fig.savefig(output_dir / "Fig_S4.png", format="png", dpi=300)
    plt.close(fig)


def make_fig_4(df_focus: pd.DataFrame, output_dir: Path) -> None:
    """Main-text cleaner 2x2 panel version."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    axes = axes.flatten()

    global_max = 0.0

    for ax, scenario_code in zip(axes, SCENARIO_ORDER):
        df_scen = df_focus[df_focus["scenario_code"] == scenario_code].copy()
        df_scen = df_scen.sort_values("pct_1p5_used", ascending=True)

        x = np.arange(len(df_scen))
        used = df_scen["pct_1p5_used"].to_numpy()
        remaining = df_scen["pct_1p5_remaining"].to_numpy()
        used_2c = df_scen["pct_2C_used"].to_numpy()
        labels = df_scen["combo_label"].to_numpy()

        if len(used) > 0:
            global_max = max(global_max, float(used.max()))

        ax.bar(x, used, color=BAR_COLOR, label="1.5°C budget used", zorder=2)
        ax.bar(
            x,
            remaining,
            bottom=used,
            color=BAR_COLOR,
            alpha=0.25,
            label="budget remaining",
            zorder=2,
        )
        ax.scatter(
            x,
            used_2c,
            marker="D",
            s=55,
            color=MARKER_COLOR,
            label="2°C budget used",
            zorder=4,
        )
        ax.axhline(100, linestyle="--", linewidth=1, color="grey")

        ax.set_title(SCENARIO_TITLES[scenario_code], fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        style_axes(ax)

    axes[0].set_ylabel("Carbon Budget Share (%)")
    axes[2].set_ylabel("Carbon Budget Share (%)")
    axes[2].set_xlabel("Parameter combination")
    axes[3].set_xlabel("Parameter combination")

    ymax = max(110, global_max * 1.08 if global_max > 0 else 110)
    for ax in axes:
        ax.set_ylim(0, ymax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:3], labels[:3], loc="lower center", ncol=3, frameon=False)

    fig.suptitle(
        "Carbon Budget Depletion by 2050\n"
        "Baseline Military Share = 5.5% | "
        f"dᵣ = {FIXED_D_REST*100:.0f}%/yr | "
        f"ε = {FIXED_EPSILON*100:.1f}%",
        fontsize=14,
    )

    fig.tight_layout(rect=[0, 0.08, 1, 0.94])
    fig.savefig(output_dir / "Fig_5.pdf", format="pdf", dpi=300)
    fig.savefig(output_dir / "Fig_5.png", format="png", dpi=300)
    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    script_dir = Path(__file__).resolve().parent
    input_path = script_dir / INPUT_FILE
    output_dir = script_dir / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    df_focus = load_focus_data(input_path)

    # Optional sanity check
    print("Rows in focused dataset:", len(df_focus))
    print("Growth values:", sorted(df_focus["g_world"].unique()))
    print("d_mil values:", sorted(df_focus["d_mil"].unique()))
    print("Scenarios:", sorted(df_focus["scenario_code"].unique()))

    make_fig_4(df_focus, output_dir)
    make_fig_s4(df_focus, output_dir)

    print(f"Saved {output_dir / 'Fig_5.pdf'}")
    print(f"Saved {output_dir / 'Fig_5.png'}")
    print(f"Saved {output_dir / 'Fig_S5.pdf'}")
    print(f"Saved {output_dir / 'Fig_S5.png'}")


if __name__ == "__main__":
    main()