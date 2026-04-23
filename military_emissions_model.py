# SEGOVIA MARTIN, JOSE

# This script contains the simulation layer.
# It runs the full parameter grid and saves three CSV files:

# 1. generated_data_2025-2035.csv
# 2. generated_data_2025-2050.csv
# 3. generated_data_budgets_2050.csv

# ================================================================
# MILITARY EMISSIONS MODEL
# ================================================================
#
#  KEY ASSUMPTIONS:
#  - WORLD_GHG_2025 = 54.5 GtCO2e
#  - Military baseline in 2025 = s_mil_2025 × 54.5 GtCO2e
#  - Global military burden (share of world GDP) in 2025: ~2.5%.
#  - NATO burden in 2025 ~2.7%; non-NATO inferred to match the global average.
#  - NATO share of world GDP ~50.18% (nominal).
#  - World GDP in trillion USDGDP_world_2025: float = 117.17
#
#  SCENARIOS:
#  - Militarisation scenarios:
#        - Benchmark or Baseline / BAU (burdens hold at 2025)
#        - NATO-only to 3.5%; non-NATO holds
#        - NATO to 3.5%; non-NATO matches to 3.5%.
#        - NATO to 5%; non-NATO matches to 3.5%.
#
#  - Military footprint baseline in 2025 S_MIL_RANGE ∈ {3.3%, 5.5%, 7.0%}
#  - GROWTH_RATES = [0.01, 0.02, 0.03, 0.04]
#  - D_MIL = [0.00, -0.01, -0.02, -0.03, -0.04, -0.05, -0.06, -0.07]
#  - D_REST = [-0.01, -0.02, -0.03, -0.04, -0.05, -0.06, -0.07]
#  - ELASTICITIES = [0.0, 0.009, 0.015, 0.02]
#
# ================================================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import itertools

import numpy as np
import pandas as pd


# ============================================================================
# GLOBAL ASSUMPTIONS
# ============================================================================

WORLD_GHG_2025 = 54.5  # GtCO2e
BUDGET_1P5_GT = 142.0
BUDGET_2C_GT = 892.0

S_MIL_RANGE = [0.033, 0.055, 0.070]
GROWTH_RATES = [0.01, 0.02, 0.03, 0.04]
D_MIL = [0.00, -0.01, -0.02, -0.03, -0.04, -0.05, -0.06, -0.07]
D_REST = [-0.01, -0.02, -0.03, -0.04, -0.05, -0.06, -0.07]
ELASTICITIES = [0.0, 0.009, 0.015, 0.02]


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass(frozen=True)
class ModelParams:
    """Model parameters that do not vary across the main parameter grid."""

    b_global_2025: float = 0.025
    m_NATO_2025: float = 0.027
    s_NATO_GDP: float = 0.5018
    GDP_world_2025: float = 117.17

    s_mil_2025: float = 0.055
    E_mil_2025: float = WORLD_GHG_2025 * 0.055

    start_year: int = 2025
    end_year: int = 2035

    ramp_method: str = "logistic"
    logistic_k: float = 5.0
    logistic_mid: float = 0.5


@dataclass(frozen=True)
class Scenario:
    """Militarisation scenario."""

    code: str
    name: str
    m_target_2035_NATO: float
    m_target_2035_nonNATO: float


# ============================================================================
# HELPERS
# ============================================================================


def years_array(start_year: int, end_year: int) -> np.ndarray:
    return np.arange(start_year, end_year + 1)



def compound_series(base: float, annual_rate: float, n_periods: int) -> np.ndarray:
    t = np.arange(n_periods, dtype=float)
    return base * (1.0 + annual_rate) ** t



def ramp_series(
    start: float,
    end: float,
    n_periods: int,
    method: str = "logistic",
    k: float = 5.0,
    mid: float = 0.5,
) -> np.ndarray:
    """Create a ramp from start to end over n_periods."""
    if n_periods <= 1:
        return np.array([end], dtype=float)

    if method == "linear":
        return np.linspace(start, end, n_periods)

    if method == "logistic":
        x = np.linspace(0.0, 1.0, n_periods)
        logistic = 1.0 / (1.0 + np.exp(-k * (x - mid)))
        logistic_norm = (logistic - logistic[0]) / (logistic[-1] - logistic[0] + 1e-12)
        return start + (end - start) * logistic_norm

    raise ValueError(f"Unknown ramp method: {method}")



def make_scenarios(m_nonNATO_2025: float) -> list[Scenario]:
    return [
        Scenario("S0", "Baseline", 0.027, m_nonNATO_2025),
        Scenario("S1", "NATO→3.5%, nonNATO holds", 0.035, m_nonNATO_2025),
        Scenario("S2", "NATO→3.5%, nonNATO→3.5%", 0.035, 0.035),
        Scenario("S3", "NATO→5%, nonNATO→3.5%", 0.050, 0.035),
    ]


# ============================================================================
# CALIBRATION
# ============================================================================


def calibrate_intensities(params: ModelParams) -> dict[str, float]:
    """Calibrate 2025 intensities from the assumed 2025 military share."""
    e_world_2025 = params.E_mil_2025 / params.s_mil_2025
    m_global_2025 = params.b_global_2025 * params.GDP_world_2025

    m_nonNATO_2025 = (
        (params.b_global_2025 - params.s_NATO_GDP * params.m_NATO_2025)
        / (1.0 - params.s_NATO_GDP)
    )

    eta_mil_2025 = params.E_mil_2025 / m_global_2025
    gdp_rest_2025 = params.GDP_world_2025 - m_global_2025
    e_rest_2025 = e_world_2025 - params.E_mil_2025
    eta_rest_2025 = e_rest_2025 / gdp_rest_2025

    return {
        "E_world_2025": e_world_2025,
        "M_global_2025": m_global_2025,
        "eta_mil_2025": eta_mil_2025,
        "eta_rest_2025": eta_rest_2025,
        "m_nonNATO_2025": m_nonNATO_2025,
    }


# ============================================================================
# SIMULATION ENGINE
# ============================================================================


def simulate_timeseries(
    params: ModelParams,
    scenario: Scenario,
    g_world: float,
    d_mil: float,
    d_rest: float,
    epsilon: float,
) -> pd.DataFrame:
    years = years_array(params.start_year, params.end_year)
    n_years = len(years)
    t_index = np.arange(n_years)

    calib = calibrate_intensities(params)
    m_nonNATO_2025 = calib["m_nonNATO_2025"]

    gdp_world = compound_series(params.GDP_world_2025, g_world, n_years)
    gdp_NATO = params.s_NATO_GDP * gdp_world
    gdp_nonNATO = (1.0 - params.s_NATO_GDP) * gdp_world

    ramp_end_year = 2035
    ramp_length = ramp_end_year - params.start_year + 1

    m_NATO_ramp = ramp_series(
        params.m_NATO_2025,
        scenario.m_target_2035_NATO,
        ramp_length,
        method=params.ramp_method,
        k=params.logistic_k,
        mid=params.logistic_mid,
    )

    m_nonNATO_ramp = ramp_series(
        m_nonNATO_2025,
        scenario.m_target_2035_nonNATO,
        ramp_length,
        method=params.ramp_method,
        k=params.logistic_k,
        mid=params.logistic_mid,
    )

    if ramp_length < n_years:
        m_NATO = np.concatenate(
            [m_NATO_ramp, np.full(n_years - ramp_length, scenario.m_target_2035_NATO)]
        )
        m_nonNATO = np.concatenate(
            [
                m_nonNATO_ramp,
                np.full(n_years - ramp_length, scenario.m_target_2035_nonNATO),
            ]
        )
    else:
        m_NATO = m_NATO_ramp[:n_years]
        m_nonNATO = m_nonNATO_ramp[:n_years]

    m_NATO_spend = m_NATO * gdp_NATO
    m_nonNATO_spend = m_nonNATO * gdp_nonNATO
    m_global_spend = m_NATO_spend + m_nonNATO_spend

    eta_mil = calib["eta_mil_2025"] * (1.0 + d_mil) ** t_index
    eta_rest = calib["eta_rest_2025"] * (1.0 + d_rest) ** t_index

    e_mil = eta_mil * m_global_spend

    e_rest_NATO_base = eta_rest * (gdp_NATO - m_NATO_spend)
    e_rest_nonNATO_base = eta_rest * (gdp_nonNATO - m_nonNATO_spend)

    delta_pp_NATO = (m_NATO - params.m_NATO_2025) / 0.01
    delta_pp_nonNATO = (m_nonNATO - m_nonNATO_2025) / 0.01

    spill_NATO = 1.0 + epsilon * delta_pp_NATO
    spill_nonNATO = 1.0 + epsilon * delta_pp_nonNATO

    e_rest_NATO = e_rest_NATO_base * spill_NATO
    e_rest_nonNATO = e_rest_nonNATO_base * spill_nonNATO
    e_rest = e_rest_NATO + e_rest_nonNATO

    e_world = e_mil + e_rest
    s_mil = e_mil / e_world

    return pd.DataFrame(
        {
            "year": years,
            "scenario_code": scenario.code,
            "scenario": scenario.name,
            "s_mil_2025_assumed": params.s_mil_2025,
            "g_world": g_world,
            "d_mil": d_mil,
            "d_rest": d_rest,
            "epsilon": epsilon,
            "m_NATO": m_NATO,
            "m_nonNATO": m_nonNATO,
            "M_NATO": m_NATO_spend,
            "M_nonNATO": m_nonNATO_spend,
            "M_global": m_global_spend,
            "eta_mil": eta_mil,
            "eta_rest": eta_rest,
            "E_mil_Gt": e_mil,
            "E_rest_Gt": e_rest,
            "E_world_Gt": e_world,
            "s_mil": s_mil,
        }
    )



def run_grid(
    params: ModelParams,
    scenarios: Iterable[Scenario],
    growth_rates: Iterable[float],
    d_mil_values: Iterable[float],
    d_rest_values: Iterable[float],
    elasticities: Iterable[float],
) -> pd.DataFrame:
    dfs: list[pd.DataFrame] = []

    for scenario, growth, d_mil, d_rest, epsilon in itertools.product(
        scenarios,
        growth_rates,
        d_mil_values,
        d_rest_values,
        elasticities,
    ):
        dfs.append(
            simulate_timeseries(
                params=params,
                scenario=scenario,
                g_world=growth,
                d_mil=d_mil,
                d_rest=d_rest,
                epsilon=epsilon,
            )
        )

    return pd.concat(dfs, ignore_index=True)



def run_multi_baseline_simulation(start_year: int, end_year: int) -> pd.DataFrame:
    all_dfs: list[pd.DataFrame] = []

    for s_mil_2025 in S_MIL_RANGE:
        params = ModelParams(
            start_year=start_year,
            end_year=end_year,
            s_mil_2025=s_mil_2025,
            E_mil_2025=WORLD_GHG_2025 * s_mil_2025,
        )

        calib = calibrate_intensities(params)
        scenarios = make_scenarios(calib["m_nonNATO_2025"])

        df = run_grid(
            params=params,
            scenarios=scenarios,
            growth_rates=GROWTH_RATES,
            d_mil_values=D_MIL,
            d_rest_values=D_REST,
            elasticities=ELASTICITIES,
        )
        all_dfs.append(df)

    result = pd.concat(all_dfs, ignore_index=True)

    sort_cols = [
        "s_mil_2025_assumed",
        "scenario_code",
        "g_world",
        "d_mil",
        "d_rest",
        "epsilon",
        "year",
    ]
    return result.sort_values(sort_cols).reset_index(drop=True)


# ============================================================================
# BUDGET SUMMARY DATASET
# ============================================================================


def build_budget_summary(df_2050: pd.DataFrame) -> pd.DataFrame:
    """Create one-row-per-parameter-combination cumulative budget summary for 2050."""
    group_cols = [
        "scenario_code",
        "scenario",
        "s_mil_2025_assumed",
        "g_world",
        "d_mil",
        "d_rest",
        "epsilon",
    ]

    df = df_2050.sort_values(group_cols + ["year"]).copy()
    df["cum_E_mil_Gt"] = df.groupby(group_cols)["E_mil_Gt"].cumsum()

    summary_2050 = df.loc[df["year"] == 2050, group_cols + ["cum_E_mil_Gt"]].copy()
    summary_2050["budget_1p5_Gt"] = BUDGET_1P5_GT
    summary_2050["budget_2C_Gt"] = BUDGET_2C_GT
    summary_2050["pct_1p5_used"] = 100.0 * summary_2050["cum_E_mil_Gt"] / BUDGET_1P5_GT
    summary_2050["pct_2C_used"] = 100.0 * summary_2050["cum_E_mil_Gt"] / BUDGET_2C_GT
    summary_2050["pct_1p5_remaining"] = 100.0 - summary_2050["pct_1p5_used"]
    summary_2050["pct_2C_remaining"] = 100.0 - summary_2050["pct_2C_used"]

    return summary_2050.sort_values(group_cols).reset_index(drop=True)


# ============================================================================
# MAIN
# ============================================================================


def main(output_dir: str | Path = ".") -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df_2035 = run_multi_baseline_simulation(start_year=2025, end_year=2035)
    df_2050 = run_multi_baseline_simulation(start_year=2025, end_year=2050)
    df_budgets_2050 = build_budget_summary(df_2050)

    file_2035 = output_path / "generated_data_2025-2035.csv"
    file_2050 = output_path / "generated_data_2025-2050.csv"
    file_budgets = output_path / "generated_data_budgets_2050.csv"

    df_2035.to_csv(file_2035, index=False)
    df_2050.to_csv(file_2050, index=False)
    df_budgets_2050.to_csv(file_budgets, index=False)

    print(f"Saved: {file_2035}")
    print(f"Saved: {file_2050}")
    print(f"Saved: {file_budgets}")
    print(f"Rows in 2025-2035 dataset: {len(df_2035):,}")
    print(f"Rows in 2025-2050 dataset: {len(df_2050):,}")
    print(f"Rows in 2050 budget summary: {len(df_budgets_2050):,}")


if __name__ == "__main__":
    main()
