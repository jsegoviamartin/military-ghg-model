"""
MILITARY-EMISSIONS MODEL: THREE INDEPENDENT SENSITIVITY ANALYSES
================================================================

Original model engine: preserved unchanged. Only the original model reference
case is overlaid in the sensitivity figures:
    military GHG footprint in 2025 = 5.5%
    g_world = 3%/yr
    d_mil = 1%/yr
    d_rest = 1%/yr
    epsilon = 0

SA1 — AR6 World, fixed bloc structure
    GDP and GHG come from the SAME AR6 World Model + Scenario.
    GDP is used at the AR6 reported level (GDP|PPP), with no 117.17 rebasing.
    NATO/non-NATO GDP shares are held fixed at the original model value 0.5018.
    The AR6 economy-wide GHG-intensity evolution is applied proportionally to
    both military and non-military emissions intensities.

SA2 — AR6 World + updated country GDP structure
    Same AR6 World GDP/GHG treatment as SA1, but NATO/non-NATO GDP shares evolve
    using country-level SSP Basic Drivers v3.2 (OECD ENV-Growth 2025, GDP|PPP).
    The country dataset determines ONLY the bloc shares; AR6 remains the source
    of the absolute world GDP and GHG pathways. This is explicitly an updated
    socioeconomic sensitivity, not a claim that v3.2 generated the AR6 runs.

SA3 — Original model recalibrated in PPP
    The original model equations and reference dynamic assumptions are retained,
    but the 2025 world GDP level and NATO GDP weight are recalibrated using IMF
    WEO PPPGDP. The global military burden remains 2.5% and NATO's burden 2.7%;
    the implied non-NATO burden and 2025 emissions intensities are recalibrated.
    GDP then evolves exactly as in the original reference case (3% real growth),
    with d_mil=d_rest=1%/yr and epsilon=0.

Each SA writes separate CSVs and Figures 1–5 (PNG + PDF).

Required inputs:
    1668008312256-AR6_Scenarios_Database_World_v1.1.csv.zip
    ssp_basic_drivers_release_3.2_full.xlsx

Preferred SA3 input:
    An IMF WEO PPPGDP export from the SAME WEO vintage used for the MER
    calibration (use the same IMF WEO 2025 vintage as the MER workbook). The script
    accepts a Data Explorer Excel export or a legacy full WEO tab-delimited file.
"""


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
#  - D_MIL = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
#  - D_REST = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
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

import hashlib
import io
import re
import unicodedata
import zipfile
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import requests



# ============================================================================
# GLOBAL ASSUMPTIONS
# ============================================================================

WORLD_GHG_2025 = 54.5  # GtCO2e
BUDGET_START_YEAR_MAIN = 2026
BUDGET_1P5_GT = 130.0
BUDGET_2C_GT = 1050.0
MILITARY_CO2_FRACTION_MAIN = 0.74

S_MIL_RANGE = [0.033, 0.055, 0.070]
GROWTH_RATES = [0.01, 0.02, 0.03, 0.04]
D_MIL = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
D_REST = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
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

    eta_mil = calib["eta_mil_2025"] * (1.0 - d_mil) ** t_index
    eta_rest = calib["eta_rest_2025"] * (1.0 - d_rest) ** t_index

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
    """Create a 2026--2050 military CO2 budget summary for each model run."""
    group_cols = [
        "scenario_code",
        "scenario",
        "s_mil_2025_assumed",
        "g_world",
        "d_mil",
        "d_rest",
        "epsilon",
    ]

    df = df_2050.loc[df_2050["year"] >= BUDGET_START_YEAR_MAIN].sort_values(
        group_cols + ["year"]
    ).copy()
    df["cum_E_mil_2026_2050_GtCO2e"] = df.groupby(group_cols)["E_mil_Gt"].cumsum()

    summary_2050 = df.loc[
        df["year"] == 2050,
        group_cols + ["cum_E_mil_2026_2050_GtCO2e"],
    ].copy()
    summary_2050["cum_E_mil_CO2_Gt"] = (
        MILITARY_CO2_FRACTION_MAIN * summary_2050["cum_E_mil_2026_2050_GtCO2e"]
    )
    summary_2050["budget_1p5_Gt"] = BUDGET_1P5_GT
    summary_2050["budget_2C_Gt"] = BUDGET_2C_GT
    summary_2050["pct_1p5_used"] = 100.0 * summary_2050["cum_E_mil_CO2_Gt"] / BUDGET_1P5_GT
    summary_2050["pct_2C_used"] = 100.0 * summary_2050["cum_E_mil_CO2_Gt"] / BUDGET_2C_GT
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








# ============================================================================
# THREE SENSITIVITY ANALYSES
# ============================================================================
#
# SA1: AR6 pathways, fixed NATO GDP share (0.5018).
# SA2: AR6 pathways, evolving NATO GDP share from SSP Basic Drivers v3.2.
# SA3: PPP recalibration using IMF WEO PPPGDP.
#
# AR6 intensity assumption (SA1/SA2):
#   I_econ(s,t) = GHG_AR6(s,t) / GDP_AR6(s,t)
#   q(s,t)      = I_econ(s,t) / I_econ(s,2025)
# Military and rest-of-economy sectors retain different calibrated 2025
# intensity levels, but both inherit the same proportional multiplier q(s,t).
#
# Only the original MER reference case is overlaid in the figures.
# ============================================================================

from pathlib import Path
import argparse
import re
import unicodedata
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# ----------------------------------------------------------------------------
# USER SETTINGS
# ----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_INPUT_DIR = PROJECT_ROOT / "data" / "inputs"
DEFAULT_AR6_WORLD_FILE = DATA_INPUT_DIR / "1668008312256-AR6_Scenarios_Database_World_v1.1.csv.zip"
DEFAULT_BASIC_DRIVERS_FILE = DATA_INPUT_DIR / "ssp_basic_drivers_release_3.2_full.xlsx"
DEFAULT_OECD_COUNTRY_FILE = DATA_INPUT_DIR / "SspDb_country_data_2013.csv"
ROOT_OUTPUT_DIR = PROJECT_ROOT / "output" / "three_sensitivity_analyses"
DEFAULT_IMF_PPP_FILE = DATA_INPUT_DIR / "IMF_WEO_PPP_2025.csv"

START_YEAR = 2025
END_YEAR = 2050
RAMP_END_YEAR = 2035

# Figures focus on the manuscript/reference military-footprint assumption.
# The CSV simulations still include all values in S_MIL_RANGE.
FIGURE_BASELINE = 0.055

# Original reference case shown as the ONLY original-model comparison.
REFERENCE_PARAMS = {
    "s_mil_2025_assumed": 0.055,
    "g_world": 0.03,
    "d_mil": 0.01,
    "d_rest": 0.01,
    "epsilon": 0.0,
}

FIXED_NATO_GDP_SHARE = 0.5018
GLOBAL_MILITARY_BURDEN_2025 = 0.025
NATO_MILITARY_BURDEN_2025 = 0.027
GDP_REBASE_2025_TRILLION = 117.17

# Revised carbon-budget comparison currently used in the manuscript workflow.
BUDGET_START_YEAR = 2026
BUDGET_END_YEAR = 2050
MILITARY_CO2_FRACTION = 0.74
BUDGET_1P5_GTCO2 = 130.0
BUDGET_2C_GTCO2 = 1050.0

AR6_GDP_VARIABLE = "GDP|PPP"
AR6_GHG_VARIABLE = (
    "AR6 climate diagnostics|Infilled|Emissions|Kyoto Gases (AR6-GWP100)"
)

AR6_PATHWAYS = [
    "SSP1-1.9",
    "SSP1-2.6",
    "SSP2-4.5",
    "SSP3-7.0",
    "SSP5-8.5",
]

AR6_PATHWAY_CONFIG = {
    "SSP1-1.9": {
        "scenario_candidates": ("SSP1-19",),
        "preferred_model_fragments": ("IMAGE",),
    },
    "SSP1-2.6": {
        "scenario_candidates": ("SSP1-26",),
        "preferred_model_fragments": ("IMAGE",),
    },
    "SSP2-4.5": {
        "scenario_candidates": ("SSP2-45",),
        "preferred_model_fragments": ("MESSAGE-GLOBIOM", "MESSAGE"),
    },
    "SSP3-7.0": {
        "scenario_candidates": ("SSP3-7.0_zeromig", "SSP3-70", "SSP3-7.0"),
        "preferred_model_fragments": ("AIM/CGE", "AIM"),
    },
    "SSP5-8.5": {
        "scenario_candidates": ("SSP5-8.5_zeromig", "SSP5-85", "SSP5-8.5"),
        "preferred_model_fragments": ("REMIND-MAgPIE", "REMIND"),
    },
}

PATHWAY_TO_SSP = {
    "SSP1-1.9": "SSP1",
    "SSP1-2.6": "SSP1",
    "SSP2-4.5": "SSP2",
    "SSP3-7.0": "SSP3",
    "SSP5-8.5": "SSP5",
}

SA_CONFIG = {
    "SA1_AR6_fixed_split": {
        "label": "SA1: AR6 World GDP/GHG; fixed NATO share",
        "rebase_gdp": False,
        "bloc_method": "fixed",
    },
    "SA2_AR6_dynamic_v32_split": {
        "label": "SA2: AR6 World GDP/GHG; dynamic SSP v3.2 NATO share",
        "rebase_gdp": False,
        "bloc_method": "country",
    },
}

SCENARIO_ORDER = ["S0", "S1", "S2", "S3"]
SCENARIO_TITLES = {
    "S0": "S0: Baseline",
    "S1": "S1: NATO→3.5%, non-NATO holds",
    "S2": "S2: NATO→3.5%, non-NATO→3.5%",
    "S3": "S3: NATO→5%, non-NATO→3.5%",
}

PATHWAY_COLORS = {
    "SSP1-1.9": "#1b4f72",
    "SSP1-2.6": "#2874a6",
    "SSP2-4.5": "#7d8f69",
    "SSP3-7.0": "#c27c0e",
    "SSP5-8.5": "#922b21",
}
PATHWAY_STYLES = {
    "SSP1-1.9": "-",
    "SSP1-2.6": "--",
    "SSP2-4.5": "-.",
    "SSP3-7.0": ":",
    "SSP5-8.5": (0, (5, 1, 1, 1)),
}

# NATO membership used by the 2025 military-burden model (32 members).
NATO_ISO3 = {
    "ALB", "BEL", "BGR", "CAN", "HRV", "CZE", "DNK", "EST",
    "FIN", "FRA", "DEU", "GRC", "HUN", "ISL", "ITA", "LVA",
    "LTU", "LUX", "MNE", "NLD", "MKD", "NOR", "POL", "PRT",
    "ROU", "SVK", "SVN", "ESP", "SWE", "TUR", "GBR", "USA",
}

NATO_NAMES = {
    "albania", "belgium", "bulgaria", "canada", "croatia", "czechia",
    "czech republic", "denmark", "estonia", "finland", "france", "germany",
    "greece", "hungary", "iceland", "italy", "latvia", "lithuania",
    "luxembourg", "montenegro", "netherlands", "north macedonia", "macedonia",
    "norway", "poland", "portugal", "romania", "slovakia", "slovenia",
    "spain", "sweden", "turkey", "turkiye", "united kingdom", "uk",
    "great britain", "united states", "united states of america", "usa",
}

AGGREGATE_REGION_NAMES = {
    "world", "global", "oecd", "non-oecd", "asia", "africa", "europe",
    "latin america", "middle east", "north america", "reforming economies",
    "r5asia", "r5lam", "r5maf", "r5oecd", "r5ref",
}


# ----------------------------------------------------------------------------
# BASIC DATA HELPERS
# ----------------------------------------------------------------------------

def _normalise_text(x: object) -> str:
    s = unicodedata.normalize("NFKD", str(x))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _is_nato_region(region: object) -> bool:
    raw = str(region).strip().upper()
    if raw in NATO_ISO3:
        return True
    return _normalise_text(region) in NATO_NAMES


def _looks_like_aggregate_region(region: object) -> bool:
    return _normalise_text(region) in AGGREGATE_REGION_NAMES


def _year_value_pairs(row: pd.Series, start: int = 2020, end: int = 2050):
    years, values = [], []
    for col in row.index:
        try:
            y = int(str(col))
        except ValueError:
            continue
        if start <= y <= end:
            v = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
            if np.isfinite(v):
                years.append(y)
                values.append(float(v))
    return np.asarray(years, float), np.asarray(values, float)


def _annualise(years, values, *, use_log: bool) -> pd.Series:
    years = np.asarray(years, float)
    values = np.asarray(values, float)
    keep = np.isfinite(years) & np.isfinite(values)
    years, values = years[keep], values[keep]
    if len(years) < 2:
        raise ValueError("At least two finite time points are required.")
    order = np.argsort(years)
    years, values = years[order], values[order]
    target = np.arange(START_YEAR, END_YEAR + 1, dtype=float)
    if use_log and np.all(values > 0):
        out = np.exp(np.interp(target, years, np.log(values)))
    else:
        out = np.interp(target, years, values)
    return pd.Series(out, index=target.astype(int), dtype=float)


def _gdp_to_trillion(values: pd.Series, unit: str) -> pd.Series:
    u = str(unit).lower()
    if "billion" in u:
        return values / 1000.0
    if "trillion" in u:
        return values.copy()
    # The AR6 GDP variables are normally in billion currency units/year.
    # Fail rather than silently infer a scale.
    raise ValueError(f"Unsupported AR6 GDP unit: {unit}")


def _ghg_to_gt(values: pd.Series, unit: str) -> pd.Series:
    u = str(unit).lower()
    if "mt" in u:
        return values / 1000.0
    if "gt" in u:
        return values.copy()
    raise ValueError(f"Unsupported AR6 GHG unit: {unit}")


# ----------------------------------------------------------------------------
# AR6 WORLD: SELECT SAME MODEL + SCENARIO FOR GDP AND GHG
# ----------------------------------------------------------------------------

def load_ar6_world(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing AR6 World file: {path}")

    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            name = next(
                (n for n in zf.namelist()
                 if n.endswith("AR6_Scenarios_Database_World_v1.1.csv")),
                None,
            )
            if name is None:
                raise ValueError(
                    "Could not find AR6_Scenarios_Database_World_v1.1.csv inside ZIP."
                )
            with zf.open(name) as fh:
                return pd.read_csv(fh, low_memory=False)

    return pd.read_csv(path, low_memory=False)


def choose_ar6_pair(raw: pd.DataFrame, pathway: str) -> tuple[str, str]:
    cfg = AR6_PATHWAY_CONFIG[pathway]
    for scenario in cfg["scenario_candidates"]:
        d = raw[(raw["Region"] == "World") & (raw["Scenario"] == scenario)]
        gdp_models = set(d.loc[d["Variable"] == AR6_GDP_VARIABLE, "Model"])
        ghg_models = set(d.loc[d["Variable"] == AR6_GHG_VARIABLE, "Model"])
        common = sorted(gdp_models & ghg_models)
        if not common:
            continue

        for fragment in cfg["preferred_model_fragments"]:
            match = next(
                (m for m in common if fragment.lower() in str(m).lower()),
                None,
            )
            if match is not None:
                return str(match), str(scenario)
        return str(common[0]), str(scenario)

    raise ValueError(
        f"{pathway}: no AR6 World Model/Scenario contains BOTH "
        f"{AR6_GDP_VARIABLE!r} and {AR6_GHG_VARIABLE!r}."
    )


def build_ar6_world_base(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, audit = [], []

    for pathway in AR6_PATHWAYS:
        model, scenario = choose_ar6_pair(raw, pathway)
        d = raw[
            (raw["Model"] == model)
            & (raw["Scenario"] == scenario)
            & (raw["Region"] == "World")
        ].copy()

        gdp_rows = d[d["Variable"] == AR6_GDP_VARIABLE]
        ghg_rows = d[d["Variable"] == AR6_GHG_VARIABLE]
        if len(gdp_rows) != 1 or len(ghg_rows) != 1:
            raise ValueError(
                f"{pathway}: expected exactly one GDP row and one GHG row for "
                f"{model} / {scenario}; found GDP={len(gdp_rows)}, GHG={len(ghg_rows)}."
            )

        rg = gdp_rows.iloc[0]
        re = ghg_rows.iloc[0]
        gy, gv = _year_value_pairs(rg)
        ey, ev = _year_value_pairs(re)

        # GDP: geometric interpolation between reporting years.
        gdp_native = _annualise(gy, gv, use_log=True)
        gdp_trillion = _gdp_to_trillion(gdp_native, str(rg["Unit"]))

        # GHG: linear interpolation between reporting years; this remains valid
        # if a mitigation pathway approaches or crosses zero in later periods.
        ghg_native = _annualise(ey, ev, use_log=False)
        ghg_gt = _ghg_to_gt(ghg_native, str(re["Unit"]))

        for year in range(START_YEAR, END_YEAR + 1):
            rows.append(
                {
                    "ssp": pathway,
                    "year": year,
                    "ar6_model": model,
                    "ar6_scenario": scenario,
                    "GDP_AR6_raw_trillion": float(gdp_trillion.loc[year]),
                    "GDP_AR6_variable": AR6_GDP_VARIABLE,
                    "GDP_AR6_unit_native": str(rg["Unit"]),
                    "GHG_AR6_GtCO2e": float(ghg_gt.loc[year]),
                    "GHG_AR6_variable": AR6_GHG_VARIABLE,
                    "GHG_AR6_unit_native": str(re["Unit"]),
                }
            )

        audit.append(
            {
                "ssp": pathway,
                "ar6_model": model,
                "ar6_scenario": scenario,
                "GDP_variable": AR6_GDP_VARIABLE,
                "GDP_unit_native": str(rg["Unit"]),
                "GHG_variable": AR6_GHG_VARIABLE,
                "GHG_unit_native": str(re["Unit"]),
                "GDP_AR6_2025_trillion": float(gdp_trillion.loc[2025]),
                "GDP_AR6_2050_trillion": float(gdp_trillion.loc[2050]),
                "GHG_AR6_2025_GtCO2e": float(ghg_gt.loc[2025]),
                "GHG_AR6_2050_GtCO2e": float(ghg_gt.loc[2050]),
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(audit)



# ----------------------------------------------------------------------------
# SSP BASIC DRIVERS v3.2 COUNTRY GDP -> NATO SHARE ONLY (SA2)
# ----------------------------------------------------------------------------

BASIC_DRIVERS_MODEL_PREFIX = "OECD ENV-Growth 2025"
BASIC_DRIVERS_GDP_VARIABLE = "GDP|PPP"

NATO_BASIC_DRIVER_NAMES = {
    "Albania", "Belgium", "Bulgaria", "Canada", "Croatia", "Czechia",
    "Denmark", "Estonia", "Finland", "France", "Germany", "Greece",
    "Hungary", "Iceland", "Italy", "Latvia", "Lithuania", "Luxembourg",
    "Montenegro", "Netherlands", "North Macedonia", "Norway", "Poland",
    "Portugal", "Romania", "Slovakia", "Slovenia", "Spain", "Sweden",
    "Turkey", "United Kingdom", "United States",
}


def _xlsx_col_number(cell_ref: str) -> int:
    m = re.match(r"([A-Z]+)", cell_ref)
    if not m:
        raise ValueError(f"Invalid Excel cell reference: {cell_ref}")
    n = 0
    for ch in m.group(1):
        n = n * 26 + (ord(ch) - 64)
    return n


def _xlsx_sheet_targets(z: zipfile.ZipFile) -> dict[str, str]:
    ns_main = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    ns_rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    wb_root = ET.fromstring(z.read("xl/workbook.xml"))
    rel_root = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
    rels = {e.attrib["Id"]: e.attrib["Target"] for e in rel_root}
    out = {}
    for sh in wb_root.find("x:sheets", ns_main):
        rid = sh.attrib[f"{{{ns_rel}}}id"]
        target = rels[rid].lstrip("/")
        if not target.startswith("xl/"):
            target = "xl/" + target
        out[sh.attrib["name"]] = target
    return out


def _xlsx_shared_strings(z: zipfile.ZipFile) -> list[str]:
    path = "xl/sharedStrings.xml"
    if path not in z.namelist():
        return []
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    root = ET.fromstring(z.read(path))
    return [
        "".join(t.text or "" for t in si.iter(ns + "t"))
        for si in root.findall(ns + "si")
    ]


def _xlsx_cell_value(cell: ET.Element, shared: list[str]) -> object:
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    # Shared string or numeric cell.
    v = cell.find(ns + "v")
    if v is not None:
        raw = v.text
        if cell.attrib.get("t") == "s":
            return shared[int(raw)]
        return raw
    # Inline string cells (common in files written by openpyxl).
    if cell.attrib.get("t") == "inlineStr":
        is_node = cell.find(ns + "is")
        if is_node is None:
            return None
        return "".join(t.text or "" for t in is_node.iter(ns + "t"))
    return None


def _read_basic_driver_gdp_sheet(
    z: zipfile.ZipFile,
    sheet_xml: str,
    source_sheet: str,
    shared: list[str],
) -> list[dict[str, object]]:
    """Stream only OECD ENV-Growth 2025 SSP1-SSP5 GDP|PPP rows."""
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    header: dict[int, str] = {}
    wanted_years = {str(y) for y in range(2020, 2051, 5)}
    records: list[dict[str, object]] = []

    with z.open(sheet_xml) as fh:
        for _, elem in ET.iterparse(fh, events=("end",)):
            if elem.tag != ns + "row":
                continue

            cells: dict[int, object] = {}
            for c in elem.findall(ns + "c"):
                col = _xlsx_col_number(c.attrib["r"])
                value = _xlsx_cell_value(c, shared)
                if value is not None:
                    cells[col] = value

            if not header:
                if str(cells.get(1, "")).strip() == "Model":
                    header = {k: str(v).strip() for k, v in cells.items()}
                elem.clear()
                continue

            model = str(cells.get(1, "")).strip()
            scenario = str(cells.get(2, "")).strip()
            region = str(cells.get(3, "")).strip()
            variable = str(cells.get(4, "")).strip()

            if (
                model.startswith(BASIC_DRIVERS_MODEL_PREFIX)
                and scenario in {"SSP1", "SSP2", "SSP3", "SSP4", "SSP5"}
                and variable == BASIC_DRIVERS_GDP_VARIABLE
            ):
                rec = {
                    "MODEL": model,
                    "SCENARIO": scenario,
                    "REGION": region,
                    "VARIABLE": variable,
                    "UNIT": str(cells.get(5, "")).strip(),
                    "source_sheet": source_sheet,
                }
                for col, name in header.items():
                    if name in wanted_years:
                        value = cells.get(col)
                        rec[name] = (
                            float(value)
                            if value not in (None, "")
                            else np.nan
                        )
                records.append(rec)

            elem.clear()

    return records


def load_basic_drivers_country_gdp(
    path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build annual NATO GDP shares from SSP Basic Drivers v3.2.

    The denominator uses the workbook's World GDP|PPP row. The numerator is the
    sum of all 32 NATO country GDP|PPP rows. Thus the Basic Drivers data affect
    only the NATO/non-NATO split in SA2; AR6 World GDP remains the absolute
    global GDP trajectory used by the emissions model.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    records: list[dict[str, object]] = []
    with zipfile.ZipFile(path) as z:
        shared = _xlsx_shared_strings(z)
        targets = _xlsx_sheet_targets(z)
        for sheet_name in ["data", "data_turbulent_economy_data"]:
            if sheet_name in targets:
                records.extend(
                    _read_basic_driver_gdp_sheet(
                        z, targets[sheet_name], sheet_name, shared
                    )
                )

    d = pd.DataFrame(records)
    if d.empty:
        raise ValueError(
            "Could not find OECD ENV-Growth 2025 SSP1-SSP5 GDP|PPP rows in "
            f"{path.name}."
        )

    share_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []

    for ssp in ["SSP1", "SSP2", "SSP3", "SSP4", "SSP5"]:
        z = d[d["SCENARIO"] == ssp].copy()
        if z.empty:
            raise ValueError(f"No Basic Drivers GDP rows found for {ssp}.")

        # Prefer the regular sheet if a region appears on more than one sheet.
        z["sheet_priority"] = z["source_sheet"].map(
            {"data": 0, "data_turbulent_economy_data": 1}
        ).fillna(9)
        z = (
            z.sort_values(["REGION", "sheet_priority"])
            .drop_duplicates(["REGION"], keep="first")
            .copy()
        )

        region_names = set(z["REGION"].astype(str))
        missing_nato = sorted(NATO_BASIC_DRIVER_NAMES - region_names)
        if missing_nato:
            raise ValueError(
                f"{ssp}: Basic Drivers file is missing NATO countries: "
                + ", ".join(missing_nato)
            )

        world = z[z["REGION"] == "World"]
        if world.empty:
            raise ValueError(f"{ssp}: no World GDP row in Basic Drivers v3.2.")
        world_row = world.iloc[0]
        wy, wv = _year_value_pairs(world_row, start=2020, end=2050)
        world_series = _annualise(wy, wv, use_log=True)

        nato_series = pd.Series(
            0.0, index=np.arange(START_YEAR, END_YEAR + 1), dtype=float
        )
        for country in sorted(NATO_BASIC_DRIVER_NAMES):
            row = z[z["REGION"] == country].iloc[0]
            cy, cv = _year_value_pairs(row, start=2020, end=2050)
            country_series = _annualise(cy, cv, use_log=True)
            nato_series = nato_series.add(country_series, fill_value=0.0)

        share = nato_series / world_series
        if ((share <= 0) | (share >= 1)).any():
            raise ValueError(f"{ssp}: implausible NATO GDP share from Basic Drivers.")

        for year in range(START_YEAR, END_YEAR + 1):
            share_rows.append(
                {
                    "ssp_storyline": ssp,
                    "year": year,
                    "NATO_GDP_share": float(share.loc[year]),
                    "country_GDP_source": "SSP Basic Drivers v3.2 / OECD ENV-Growth 2025",
                    "country_GDP_unit": str(world_row["UNIT"]),
                    "country_GDP_vintage": "May 2025 update",
                }
            )

        audit_rows.append(
            {
                "ssp_storyline": ssp,
                "country_GDP_source": "SSP Basic Drivers v3.2 / OECD ENV-Growth 2025",
                "GDP_unit": str(world_row["UNIT"]),
                "n_recognised_NATO_members": len(NATO_BASIC_DRIVER_NAMES),
                "NATO_share_2025": float(share.loc[2025]),
                "NATO_share_2035": float(share.loc[2035]),
                "NATO_share_2050": float(share.loc[2050]),
            }
        )

    return pd.DataFrame(share_rows), pd.DataFrame(audit_rows)


# ----------------------------------------------------------------------------
# BUILD DRIVER TABLE FOR EACH SENSITIVITY
# ----------------------------------------------------------------------------

def build_sensitivity_drivers(
    ar6_base: pd.DataFrame,
    *,
    rebase_gdp: bool,
    bloc_method: str,
    country_shares: pd.DataFrame | None,
    analysis_name: str,
) -> pd.DataFrame:
    d = ar6_base.copy()

    if rebase_gdp:
        base = d[d["year"] == START_YEAR][["ssp", "GDP_AR6_raw_trillion"]].rename(
            columns={"GDP_AR6_raw_trillion": "GDP_AR6_2025_raw_trillion"}
        )
        d = d.merge(base, on="ssp", how="left", validate="many_to_one")
        d["GDP_world"] = (
            GDP_REBASE_2025_TRILLION
            * d["GDP_AR6_raw_trillion"]
            / d["GDP_AR6_2025_raw_trillion"]
        )
        d["GDP_model_basis"] = "AR6 GDP trajectory rebased to 117.17 trillion in 2025"
    else:
        d["GDP_world"] = d["GDP_AR6_raw_trillion"]
        d["GDP_model_basis"] = "AR6 World GDP level and trajectory (unrebased)"

    d["GHG_world_Gt"] = d["GHG_AR6_GtCO2e"]
    d["GHG_model_basis"] = "AR6 World GHG, unrebased"

    if bloc_method == "fixed":
        d["s_NATO_GDP"] = FIXED_NATO_GDP_SHARE
        d["bloc_split_basis"] = "Fixed original NATO GDP share = 0.5018"
    elif bloc_method == "country":
        if country_shares is None:
            raise ValueError(f"{analysis_name}: country GDP shares are required.")
        d["ssp_storyline"] = d["ssp"].map(PATHWAY_TO_SSP)
        d = d.merge(
            country_shares[
                [
                    "ssp_storyline", "year", "NATO_GDP_share",
                    "country_GDP_source", "country_GDP_unit", "country_GDP_vintage",
                ]
            ],
            on=["ssp_storyline", "year"],
            how="left",
            validate="many_to_one",
        )
        if d["NATO_GDP_share"].isna().any():
            raise ValueError(
                f"{analysis_name}: missing country-based NATO GDP shares after merge."
            )
        d["s_NATO_GDP"] = d["NATO_GDP_share"]
        d["bloc_split_basis"] = d["country_GDP_source"]
    else:
        raise ValueError(f"Unknown bloc_method={bloc_method!r}")

    d["GDP_NATO"] = d["GDP_world"] * d["s_NATO_GDP"]
    d["GDP_nonNATO"] = d["GDP_world"] - d["GDP_NATO"]

    # Economy-wide intensity derived from the GDP/GHG used by THIS sensitivity.
    d["economy_GHG_intensity"] = d["GHG_world_Gt"] / d["GDP_world"]
    q0 = d[d["year"] == START_YEAR][["ssp", "economy_GHG_intensity"]].rename(
        columns={"economy_GHG_intensity": "economy_GHG_intensity_2025"}
    )
    d = d.merge(q0, on="ssp", how="left", validate="many_to_one")
    d["intensity_multiplier_q"] = (
        d["economy_GHG_intensity"] / d["economy_GHG_intensity_2025"]
    )
    d["analysis"] = analysis_name
    return d


# ----------------------------------------------------------------------------
# ORIGINAL REFERENCE CASE ONLY
# ----------------------------------------------------------------------------

def run_original_reference_case(end_year: int = END_YEAR) -> pd.DataFrame:
    params = ModelParams(
        start_year=START_YEAR,
        end_year=end_year,
        s_mil_2025=REFERENCE_PARAMS["s_mil_2025_assumed"],
        E_mil_2025=WORLD_GHG_2025 * REFERENCE_PARAMS["s_mil_2025_assumed"],
    )
    calib = calibrate_intensities(params)
    scenarios = make_scenarios(calib["m_nonNATO_2025"])

    frames = []
    for scenario in scenarios:
        f = simulate_timeseries(
            params=params,
            scenario=scenario,
            g_world=REFERENCE_PARAMS["g_world"],
            d_mil=REFERENCE_PARAMS["d_mil"],
            d_rest=REFERENCE_PARAMS["d_rest"],
            epsilon=REFERENCE_PARAMS["epsilon"],
        )
        f["model_family"] = "Original model reference case"
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------------
# SENSITIVITY SIMULATION
# ----------------------------------------------------------------------------

def _infer_non_nato_burden_2025(s_nato_2025: float) -> float:
    if not (0 < s_nato_2025 < 1):
        raise ValueError(f"Invalid NATO GDP share in 2025: {s_nato_2025}")
    m_non = (
        GLOBAL_MILITARY_BURDEN_2025
        - s_nato_2025 * NATO_MILITARY_BURDEN_2025
    ) / (1.0 - s_nato_2025)
    if m_non <= 0:
        raise ValueError(f"Implied non-NATO military burden is non-positive: {m_non}")
    return float(m_non)


def _military_burden_paths(
    n_years: int,
    m_non_2025: float,
    scenario: Scenario,
) -> tuple[np.ndarray, np.ndarray]:
    ramp_len = RAMP_END_YEAR - START_YEAR + 1
    m_nato_ramp = ramp_series(
        NATO_MILITARY_BURDEN_2025,
        scenario.m_target_2035_NATO,
        ramp_len,
        method="logistic",
        k=5.0,
        mid=0.5,
    )
    m_non_ramp = ramp_series(
        m_non_2025,
        scenario.m_target_2035_nonNATO,
        ramp_len,
        method="logistic",
        k=5.0,
        mid=0.5,
    )

    if ramp_len < n_years:
        m_nato = np.r_[
            m_nato_ramp,
            np.full(n_years - ramp_len, scenario.m_target_2035_NATO),
        ]
        m_non = np.r_[
            m_non_ramp,
            np.full(n_years - ramp_len, scenario.m_target_2035_nonNATO),
        ]
    else:
        m_nato = m_nato_ramp[:n_years]
        m_non = m_non_ramp[:n_years]

    return m_nato, m_non


def simulate_one_sensitivity_path(
    drivers: pd.DataFrame,
    pathway: str,
    scenario: Scenario,
    s_mil_2025: float,
    analysis_name: str,
) -> pd.DataFrame:
    d = drivers[drivers["ssp"] == pathway].sort_values("year").copy()
    if d.empty:
        raise ValueError(f"{analysis_name}: no driver rows for {pathway}.")

    years = d["year"].to_numpy(int)
    gdp_world = d["GDP_world"].to_numpy(float)
    gdp_nato = d["GDP_NATO"].to_numpy(float)
    gdp_non = d["GDP_nonNATO"].to_numpy(float)
    ghg_world = d["GHG_world_Gt"].to_numpy(float)
    q = d["intensity_multiplier_q"].to_numpy(float)

    s_nato_2025 = float(d.loc[d["year"] == START_YEAR, "s_NATO_GDP"].iloc[0])
    m_non_2025 = _infer_non_nato_burden_2025(s_nato_2025)

    gdp0 = float(d.loc[d["year"] == START_YEAR, "GDP_world"].iloc[0])
    ghg0 = float(d.loc[d["year"] == START_YEAR, "GHG_world_Gt"].iloc[0])

    # 2025 military and non-military emissions levels are calibrated to the
    # assumed military footprint, but both intensity LEVELS subsequently inherit
    # the same proportional economy-wide multiplier q(s,t).
    M0_global = GLOBAL_MILITARY_BURDEN_2025 * gdp0
    E0_mil = s_mil_2025 * ghg0
    E0_rest = ghg0 - E0_mil
    GDP0_rest = gdp0 - M0_global

    eta_mil_2025 = E0_mil / M0_global
    eta_rest_2025 = E0_rest / GDP0_rest
    eta_mil = eta_mil_2025 * q
    eta_rest = eta_rest_2025 * q

    m_nato, m_non = _military_burden_paths(len(years), m_non_2025, scenario)
    M_nato = m_nato * gdp_nato
    M_non = m_non * gdp_non
    M_global = M_nato + M_non

    E_mil = eta_mil * M_global
    E_rest = eta_rest * (gdp_world - M_global)
    E_world = E_mil + E_rest
    s_mil = E_mil / E_world

    out = pd.DataFrame(
        {
            "year": years,
            "analysis": analysis_name,
            "ssp": pathway,
            "scenario_code": scenario.code,
            "scenario": scenario.name,
            "s_mil_2025_assumed": s_mil_2025,
            "GDP_world": gdp_world,
            "GDP_NATO": gdp_nato,
            "GDP_nonNATO": gdp_non,
            "s_NATO_GDP": d["s_NATO_GDP"].to_numpy(float),
            "m_NATO": m_nato,
            "m_nonNATO": m_non,
            "M_NATO": M_nato,
            "M_nonNATO": M_non,
            "M_global": M_global,
            "eta_mil": eta_mil,
            "eta_rest": eta_rest,
            "economy_GHG_intensity": d["economy_GHG_intensity"].to_numpy(float),
            "intensity_multiplier_q": q,
            "E_mil_Gt": E_mil,
            "E_rest_Gt": E_rest,
            "E_world_Gt": E_world,
            "E_world_AR6_Gt": ghg_world,
            "s_mil": s_mil,
            "GDP_model_basis": d["GDP_model_basis"].iloc[0],
            "bloc_split_basis": d["bloc_split_basis"].iloc[0],
            "ar6_model": d["ar6_model"].iloc[0],
            "ar6_scenario": d["ar6_scenario"].iloc[0],
        }
    )
    out["world_GHG_minus_AR6_Gt"] = out["E_world_Gt"] - out["E_world_AR6_Gt"]
    return out


def run_sensitivity_analysis(drivers: pd.DataFrame, analysis_name: str) -> pd.DataFrame:
    frames = []
    for s_mil_2025 in S_MIL_RANGE:
        for pathway in AR6_PATHWAYS:
            d0 = drivers[
                (drivers["ssp"] == pathway) & (drivers["year"] == START_YEAR)
            ]
            if d0.empty:
                raise ValueError(f"{analysis_name}: missing 2025 driver row for {pathway}.")
            s_nato_2025 = float(d0["s_NATO_GDP"].iloc[0])
            m_non_2025 = _infer_non_nato_burden_2025(s_nato_2025)
            for scenario in make_scenarios(m_non_2025):
                frames.append(
                    simulate_one_sensitivity_path(
                        drivers,
                        pathway,
                        scenario,
                        s_mil_2025,
                        analysis_name,
                    )
                )

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(
        ["s_mil_2025_assumed", "ssp", "scenario_code", "year"]
    ).reset_index(drop=True)

    # Annual and cumulative increments relative to S0 within the SAME SSP and
    # military-footprint assumption.
    base = out[out["scenario_code"] == "S0"][
        ["ssp", "s_mil_2025_assumed", "year", "E_mil_Gt", "E_world_Gt"]
    ].rename(
        columns={
            "E_mil_Gt": "E_mil_S0_Gt",
            "E_world_Gt": "E_world_S0_Gt",
        }
    )
    out = out.merge(
        base,
        on=["ssp", "s_mil_2025_assumed", "year"],
        how="left",
        validate="many_to_one",
    )
    out["incremental_E_mil_vs_S0_Gt"] = out["E_mil_Gt"] - out["E_mil_S0_Gt"]
    out["incremental_E_world_vs_S0_Gt"] = out["E_world_Gt"] - out["E_world_S0_Gt"]

    group = ["ssp", "s_mil_2025_assumed", "scenario_code"]
    out["cum_E_mil_Gt"] = out.groupby(group)["E_mil_Gt"].cumsum()
    out["cum_incremental_E_mil_vs_S0_Gt"] = out.groupby(group)[
        "incremental_E_mil_vs_S0_Gt"
    ].cumsum()
    out["cum_incremental_E_world_vs_S0_Gt"] = out.groupby(group)[
        "incremental_E_world_vs_S0_Gt"
    ].cumsum()
    return out


# ----------------------------------------------------------------------------
# CSV SUMMARIES / AUDITS
# ----------------------------------------------------------------------------

def build_budget_summary(df: pd.DataFrame) -> pd.DataFrame:
    d = df[
        (df["year"] >= BUDGET_START_YEAR)
        & (df["year"] <= BUDGET_END_YEAR)
    ].copy()
    keys = ["analysis", "ssp", "scenario_code", "scenario", "s_mil_2025_assumed"]
    out = (
        d.groupby(keys, as_index=False)["E_mil_Gt"]
        .sum()
        .rename(columns={"E_mil_Gt": "cum_E_mil_2026_2050_GtCO2e"})
    )
    out["cum_E_mil_CO2_Gt"] = (
        MILITARY_CO2_FRACTION * out["cum_E_mil_2026_2050_GtCO2e"]
    )
    out["pct_1p5_budget_used"] = 100.0 * out["cum_E_mil_CO2_Gt"] / BUDGET_1P5_GTCO2
    out["pct_2C_budget_used"] = 100.0 * out["cum_E_mil_CO2_Gt"] / BUDGET_2C_GTCO2
    out["pct_1p5_budget_remaining"] = 100.0 - out["pct_1p5_budget_used"]
    return out


def build_original_reference_budget(reference: pd.DataFrame) -> pd.DataFrame:
    d = reference[
        (reference["year"] >= BUDGET_START_YEAR)
        & (reference["year"] <= BUDGET_END_YEAR)
    ].copy()
    out = (
        d.groupby(["scenario_code", "scenario"], as_index=False)["E_mil_Gt"]
        .sum()
        .rename(columns={"E_mil_Gt": "cum_E_mil_2026_2050_GtCO2e"})
    )
    out["cum_E_mil_CO2_Gt"] = (
        MILITARY_CO2_FRACTION * out["cum_E_mil_2026_2050_GtCO2e"]
    )
    out["pct_1p5_budget_used"] = 100.0 * out["cum_E_mil_CO2_Gt"] / BUDGET_1P5_GTCO2
    out["pct_2C_budget_used"] = 100.0 * out["cum_E_mil_CO2_Gt"] / BUDGET_2C_GTCO2
    out["pct_1p5_budget_remaining"] = 100.0 - out["pct_1p5_budget_used"]
    return out


def build_endpoint_summary(df: pd.DataFrame) -> pd.DataFrame:
    focus = df[np.isclose(df["s_mil_2025_assumed"], FIGURE_BASELINE)].copy()
    rows = []
    for (ssp, sc), z in focus.groupby(["ssp", "scenario_code"]):
        z = z.set_index("year")
        rows.append(
            {
                "ssp": ssp,
                "scenario_code": sc,
                "E_mil_2035_Gt": float(z.loc[2035, "E_mil_Gt"]),
                "military_share_2035_pct": 100.0 * float(z.loc[2035, "s_mil"]),
                "cum_E_mil_2050_Gt": float(z.loc[2050, "cum_E_mil_Gt"]),
                "cum_incremental_E_mil_vs_S0_2050_Gt": float(
                    z.loc[2050, "cum_incremental_E_mil_vs_S0_Gt"]
                ),
                "world_minus_AR6_2050_Gt": float(z.loc[2050, "world_GHG_minus_AR6_Gt"]),
            }
        )
    return pd.DataFrame(rows)


def build_s0_anchor_audit(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["scenario_code"] == "S0"].copy()
    return (
        d.groupby(["analysis", "ssp", "s_mil_2025_assumed"], as_index=False)
        .agg(
            max_abs_S0_minus_AR6_Gt=("world_GHG_minus_AR6_Gt", lambda x: float(np.max(np.abs(x)))),
            S0_minus_AR6_2050_Gt=("world_GHG_minus_AR6_Gt", "last"),
        )
    )


def compare_sensitivities(a: pd.DataFrame, b: pd.DataFrame, pair_name: str) -> dict:
    keys = ["ssp", "scenario_code", "s_mil_2025_assumed", "year"]
    cols = ["E_mil_Gt", "E_rest_Gt", "E_world_Gt", "s_mil"]
    x = a[keys + cols].merge(
        b[keys + cols], on=keys, suffixes=("_a", "_b"), validate="one_to_one"
    )
    row = {"comparison": pair_name}
    for c in cols:
        row[f"max_abs_diff_{c}"] = float(np.max(np.abs(x[f"{c}_a"] - x[f"{c}_b"])))
    return row


# ----------------------------------------------------------------------------
# FIGURE HELPERS
# ----------------------------------------------------------------------------

def _style_axis(ax):
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _ssp_handles():
    return [
        Line2D(
            [0], [0],
            color=PATHWAY_COLORS[s],
            linestyle=PATHWAY_STYLES[s],
            lw=2.2,
            label=s,
        )
        for s in AR6_PATHWAYS
    ]


def _reference_handle():
    return Line2D(
        [0], [0], color="black", lw=2.4, linestyle="--",
        label="Original model reference case"
    )


def _focus(df: pd.DataFrame) -> pd.DataFrame:
    return df[np.isclose(df["s_mil_2025_assumed"], FIGURE_BASELINE)].copy()


def make_fig_1(df: pd.DataFrame, reference: pd.DataFrame, outdir: Path, title_prefix: str):
    d = _focus(df)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, sc in zip(axes, SCENARIO_ORDER):
        for ssp in AR6_PATHWAYS:
            z = d[(d["scenario_code"] == sc) & (d["ssp"] == ssp)].sort_values("year")
            ax.plot(
                z["year"], z["E_mil_Gt"],
                color=PATHWAY_COLORS[ssp], linestyle=PATHWAY_STYLES[ssp], lw=2.0,
            )
        r = reference[reference["scenario_code"] == sc].sort_values("year")
        ax.plot(r["year"], r["E_mil_Gt"], color="black", linestyle="--", lw=2.2)
        ax.set_xlim(2025, 2035)
        ax.set_title(SCENARIO_TITLES[sc], fontsize=11)
        _style_axis(ax)

    axes[0].set_ylabel("Military emissions (GtCO$_2$e/yr)")
    axes[2].set_ylabel("Military emissions (GtCO$_2$e/yr)")
    axes[2].set_xlabel("Year")
    axes[3].set_xlabel("Year")
    fig.legend(handles=_ssp_handles() + [_reference_handle()], loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        f"{title_prefix}\nFigure 1. Absolute military GHG emissions (2025–2035); "
        f"2025 military footprint = {100*FIGURE_BASELINE:.1f}%",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0.11, 1, 0.94])
    fig.savefig(outdir / "Fig_1.pdf", bbox_inches="tight")
    fig.savefig(outdir / "Fig_1.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_fig_2(df: pd.DataFrame, reference: pd.DataFrame, outdir: Path, title_prefix: str):
    d = _focus(df)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, sc in zip(axes, SCENARIO_ORDER):
        for ssp in AR6_PATHWAYS:
            z = d[(d["scenario_code"] == sc) & (d["ssp"] == ssp)].sort_values("year")
            ax.plot(
                z["year"], 100.0 * z["s_mil"],
                color=PATHWAY_COLORS[ssp], linestyle=PATHWAY_STYLES[ssp], lw=2.0,
            )
        r = reference[reference["scenario_code"] == sc].sort_values("year")
        ax.plot(r["year"], 100.0 * r["s_mil"], color="black", linestyle="--", lw=2.2)
        ax.set_xlim(2025, 2035)
        ax.set_title(SCENARIO_TITLES[sc], fontsize=11)
        _style_axis(ax)

    axes[0].set_ylabel("Military share of global GHG (%)")
    axes[2].set_ylabel("Military share of global GHG (%)")
    axes[2].set_xlabel("Year")
    axes[3].set_xlabel("Year")
    fig.legend(handles=_ssp_handles() + [_reference_handle()], loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        f"{title_prefix}\nFigure 2. Military share of global GHG emissions (2025–2035); "
        f"2025 military footprint = {100*FIGURE_BASELINE:.1f}%",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0.11, 1, 0.94])
    fig.savefig(outdir / "Fig_2.pdf", bbox_inches="tight")
    fig.savefig(outdir / "Fig_2.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_fig_3(df: pd.DataFrame, reference: pd.DataFrame, outdir: Path, title_prefix: str):
    d = _focus(df)
    ref = reference.sort_values(["scenario_code", "year"]).copy()
    ref["cum_E_mil_Gt"] = ref.groupby("scenario_code")["E_mil_Gt"].cumsum()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, sc in zip(axes, SCENARIO_ORDER):
        for ssp in AR6_PATHWAYS:
            z = d[(d["scenario_code"] == sc) & (d["ssp"] == ssp)].sort_values("year")
            ax.plot(
                z["year"], z["cum_E_mil_Gt"],
                color=PATHWAY_COLORS[ssp], linestyle=PATHWAY_STYLES[ssp], lw=2.0,
            )
        r = ref[ref["scenario_code"] == sc]
        ax.plot(r["year"], r["cum_E_mil_Gt"], color="black", linestyle="--", lw=2.2)
        ax.set_title(SCENARIO_TITLES[sc], fontsize=11)
        _style_axis(ax)

    axes[0].set_ylabel("Cumulative military emissions (GtCO$_2$e)")
    axes[2].set_ylabel("Cumulative military emissions (GtCO$_2$e)")
    axes[2].set_xlabel("Year")
    axes[3].set_xlabel("Year")
    fig.legend(handles=_ssp_handles() + [_reference_handle()], loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        f"{title_prefix}\nFigure 3. Cumulative military GHG emissions (2025–2050); "
        f"2025 military footprint = {100*FIGURE_BASELINE:.1f}%",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0.11, 1, 0.94])
    fig.savefig(outdir / "Fig_3.pdf", bbox_inches="tight")
    fig.savefig(outdir / "Fig_3.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_fig_4(df: pd.DataFrame, reference: pd.DataFrame, outdir: Path, title_prefix: str):
    d = _focus(df)

    # Original reference-case increments relative to its own S0.
    ref = reference.sort_values(["scenario_code", "year"]).copy()
    r0 = ref[ref["scenario_code"] == "S0"][["year", "E_mil_Gt"]].rename(
        columns={"E_mil_Gt": "E_mil_S0"}
    )
    ref = ref.merge(r0, on="year", how="left", validate="many_to_one")
    ref["inc"] = ref["E_mil_Gt"] - ref["E_mil_S0"]
    ref["cum_inc"] = ref.groupby("scenario_code")["inc"].cumsum()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    for ax, sc in zip(axes, ["S1", "S2", "S3"]):
        for ssp in AR6_PATHWAYS:
            z = d[(d["scenario_code"] == sc) & (d["ssp"] == ssp)].sort_values("year")
            ax.plot(
                z["year"], z["cum_incremental_E_mil_vs_S0_Gt"],
                color=PATHWAY_COLORS[ssp], linestyle=PATHWAY_STYLES[ssp], lw=2.0,
            )
        r = ref[ref["scenario_code"] == sc]
        ax.plot(r["year"], r["cum_inc"], color="black", linestyle="--", lw=2.2)
        ax.axhline(0, color="black", lw=0.9, alpha=0.6)
        ax.set_title(SCENARIO_TITLES[sc], fontsize=10.5)
        ax.set_xlabel("Year")
        _style_axis(ax)
    axes[0].set_ylabel(r"$\Delta$ cumulative military emissions vs S0 (GtCO$_2$e)")

    fig.legend(handles=_ssp_handles() + [_reference_handle()], loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        f"{title_prefix}\nFigure 4. Incremental cumulative military emissions relative to S0 (2025–2050)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0.16, 1, 0.90])
    fig.savefig(outdir / "Fig_4.pdf", bbox_inches="tight")
    fig.savefig(outdir / "Fig_4.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_fig_5(
    budget: pd.DataFrame,
    reference_budget: pd.DataFrame,
    outdir: Path,
    title_prefix: str,
):
    d = budget[np.isclose(budget["s_mil_2025_assumed"], FIGURE_BASELINE)].copy()
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9), sharey=True)
    axes = axes.flatten()
    global_max = 0.0

    for ax, sc in zip(axes, SCENARIO_ORDER):
        s = d[d["scenario_code"] == sc].set_index("ssp").reindex(AR6_PATHWAYS).reset_index()
        r = reference_budget[reference_budget["scenario_code"] == sc].iloc[0]

        labels = ["Original ref."] + AR6_PATHWAYS
        used = np.r_[float(r["pct_1p5_budget_used"]), s["pct_1p5_budget_used"].to_numpy(float)]
        used2 = np.r_[float(r["pct_2C_budget_used"]), s["pct_2C_budget_used"].to_numpy(float)]
        remain = np.clip(100.0 - used, 0.0, None)
        x = np.arange(len(labels))
        global_max = max(global_max, float(np.nanmax(used)))

        colors = ["black"] + [PATHWAY_COLORS[p] for p in AR6_PATHWAYS]
        bars = ax.bar(x, used, color=colors, alpha=0.85, zorder=2)
        ax.bar(x, remain, bottom=used, color=colors, alpha=0.16, zorder=1)
        ax.scatter(x, used2, marker="D", s=38, color="black", zorder=4)
        ax.axhline(100, color="grey", linestyle="--", lw=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=50, ha="right", fontsize=8)
        ax.set_title(SCENARIO_TITLES[sc], fontsize=10.5)
        _style_axis(ax)

    ymax = max(110.0, global_max * 1.08)
    for ax in axes:
        ax.set_ylim(0, ymax)
    axes[0].set_ylabel("Share of remaining carbon budget (%)")
    axes[2].set_ylabel("Share of remaining carbon budget (%)")

    handles = [
        Patch(facecolor="grey", alpha=0.85, label="Share of 1.5°C budget used"),
        Patch(facecolor="grey", alpha=0.16, label="1.5°C budget remaining"),
        Line2D([0], [0], marker="D", color="black", lw=0, label="Share of 2°C budget used"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        f"{title_prefix}\nFigure 5. Military CO$_2$ relative to remaining carbon budgets "
        f"({BUDGET_START_YEAR}–{BUDGET_END_YEAR}); 2025 military footprint = {100*FIGURE_BASELINE:.1f}%",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0.10, 1, 0.92])
    fig.savefig(outdir / "Fig_5.pdf", bbox_inches="tight")
    fig.savefig(outdir / "Fig_5.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# SAVE ONE SENSITIVITY
# ----------------------------------------------------------------------------

def save_sensitivity_outputs(
    analysis_name: str,
    cfg: dict,
    drivers: pd.DataFrame,
    reference: pd.DataFrame,
    root: Path,
) -> pd.DataFrame:
    outdir = root / analysis_name
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning {analysis_name} ...")
    results = run_sensitivity_analysis(drivers, analysis_name)
    budget = build_budget_summary(results)
    endpoint = build_endpoint_summary(results)
    anchor_audit = build_s0_anchor_audit(results)

    drivers.to_csv(outdir / "driver_paths_2025-2050.csv", index=False)
    results[results["year"] <= 2035].to_csv(
        outdir / "generated_data_2025-2035.csv", index=False
    )
    results.to_csv(outdir / "generated_data_2025-2050.csv", index=False)
    budget.to_csv(outdir / "budget_summary_2026-2050.csv", index=False)
    endpoint.to_csv(outdir / "endpoint_summary.csv", index=False)
    anchor_audit.to_csv(outdir / "S0_AR6_anchor_audit.csv", index=False)

    reference_budget = build_original_reference_budget(reference)
    title_prefix = cfg["label"]
    make_fig_1(results, reference, outdir, title_prefix)
    make_fig_2(results, reference, outdir, title_prefix)
    make_fig_3(results, reference, outdir, title_prefix)
    make_fig_4(results, reference, outdir, title_prefix)
    make_fig_5(budget, reference_budget, outdir, title_prefix)

    print(f"Saved CSVs and Fig_1 ... Fig_5 to {outdir}")
    return results


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------


# ============================================================================
# SA3 — ORIGINAL REFERENCE MODEL RECALIBRATED IN PPP
# ============================================================================

IMF_PPP_SERIES_CODE = "PPPGDP"
IMF_MER_SERIES_CODE = "NGDPD"


def _num(x: object) -> float:
    if x is None:
        return np.nan
    s = str(x).strip().replace(",", "")
    if s in {"", "--", "nan", "NaN", "None"}:
        return np.nan
    return float(s)


def _read_imf_ppp_data_explorer_xlsx(path: Path) -> pd.DataFrame:
    """Read an IMF Data Explorer Excel export containing PPPGDP rows."""
    frames = []
    xl = pd.ExcelFile(path)
    for sheet in xl.sheet_names:
        try:
            d = pd.read_excel(path, sheet_name=sheet, dtype=object)
        except Exception:
            continue
        if d.empty:
            continue
        cols = {str(c).strip().upper(): c for c in d.columns}
        series_col = cols.get("SERIES_CODE")
        country_col = cols.get("COUNTRY") or cols.get("REGION")
        indicator_col = cols.get("INDICATOR")
        year_col = next((c for c in d.columns if str(c).strip() in {"2025", "2025.0"}), None)
        if year_col is None:
            continue
        if series_col is None and indicator_col is None:
            continue

        x = d.copy()
        if series_col is not None:
            sc = x[series_col].astype(str)
            keep = sc.str.contains(r"(?:^|\.)PPPGDP(?:\.|$)", regex=True, na=False)
            x = x[keep].copy()
            x["ISO"] = x[series_col].astype(str).str.extract(r"^([A-Z]{3})\.", expand=False)
            # IMF group/world codes are not ISO3; retain them as missing ISO.
        else:
            ind = x[indicator_col].astype(str).str.lower()
            keep = ind.str.contains("gross domestic product", na=False) & ind.str.contains(
                "purchasing", na=False
            )
            x = x[keep].copy()
            x["ISO"] = np.nan

        if x.empty:
            continue
        x["Country"] = x[country_col].astype(str) if country_col is not None else ""
        x["PPPGDP_2025_bn"] = pd.to_numeric(x[year_col], errors="coerce")
        x = x[np.isfinite(x["PPPGDP_2025_bn"]) & (x["PPPGDP_2025_bn"] > 0)].copy()
        if not x.empty:
            x["source_sheet"] = sheet
            frames.append(x[["ISO", "Country", "PPPGDP_2025_bn", "source_sheet"]])

    if not frames:
        raise ValueError(
            f"{path.name}: no 2025 PPPGDP rows found. Export IMF WEO indicator "
            "'Gross domestic product, current prices, purchasing-power-parity "
            "international dollars' (PPPGDP) for World and countries."
        )
    return pd.concat(frames, ignore_index=True).drop_duplicates(["ISO", "Country"])


def _read_imf_ppp_data_explorer_csv(path: Path) -> pd.DataFrame:
    """Read an IMF Data Explorer CSV export containing 2025 PPPGDP rows.

    Expected columns in the current IMF Data Explorer export include:
    DATASET, SERIES_CODE, COUNTRY, INDICATOR, SCALE, and 2025.
    Only PPPGDP is retained. The model calibration uses 2025 only.
    """
    d = pd.read_csv(path, dtype=object, low_memory=False)
    cols = {str(c).strip().upper(): c for c in d.columns}
    required = {"SERIES_CODE", "COUNTRY", "2025"}
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(
            f"{path.name}: IMF Data Explorer CSV is missing columns: {missing}"
        )

    series_col = cols["SERIES_CODE"]
    country_col = cols["COUNTRY"]
    year_col = cols["2025"]
    indicator_col = cols.get("INDICATOR")
    scale_col = cols.get("SCALE")
    dataset_col = cols.get("DATASET")

    sc = d[series_col].astype(str)
    keep = sc.str.contains(r"(?:^|\.)PPPGDP(?:\.|$)", regex=True, na=False)
    x = d[keep].copy()
    if x.empty and indicator_col is not None:
        ind = d[indicator_col].astype(str).str.lower()
        keep = (
            ind.str.contains("gross domestic product", na=False)
            & ind.str.contains("purchasing", na=False)
            & ind.str.contains("parity", na=False)
        )
        x = d[keep].copy()

    if x.empty:
        raise ValueError(
            f"{path.name}: no PPPGDP rows found. Export IMF WEO PPPGDP "
            "for World and all countries, including year 2025."
        )

    x["ISO"] = x[series_col].astype(str).str.extract(r"^([A-Z]{3})\.", expand=False)
    x["Country"] = x[country_col].astype(str)
    x["PPPGDP_2025_bn"] = x[year_col].map(_num)
    x = x[np.isfinite(x["PPPGDP_2025_bn"]) & (x["PPPGDP_2025_bn"] > 0)].copy()
    x["source_sheet"] = "IMF Data Explorer CSV"
    x["source_dataset"] = x[dataset_col].astype(str) if dataset_col is not None else ""
    x["source_scale"] = x[scale_col].astype(str) if scale_col is not None else ""

    # PPPGDP in the Data Explorer export used here is in billions of
    # international dollars. Reject a clearly incompatible scale rather than
    # silently treating units as billions.
    scales = {s.strip().lower() for s in x["source_scale"].dropna().astype(str) if s.strip()}
    if scales and scales != {"billions"}:
        raise ValueError(
            f"{path.name}: unexpected PPPGDP scale(s) {sorted(scales)}; "
            "expected 'Billions'."
        )

    return x[[
        "ISO", "Country", "PPPGDP_2025_bn", "source_sheet",
        "source_dataset", "source_scale"
    ]].drop_duplicates(["ISO", "Country"])


def _read_imf_legacy_weo_file(path: Path) -> pd.DataFrame:
    """Read the legacy WEO entire-country tab-delimited file."""
    encodings = ["utf-16-le", "utf-16", "utf-8"]
    last = None
    for enc in encodings:
        try:
            d = pd.read_csv(path, sep="\t", encoding=enc, dtype=str, low_memory=False)
            if "WEO Subject Code" in d.columns:
                break
        except Exception as exc:
            last = exc
    else:
        raise ValueError(f"Could not parse legacy WEO file {path}: {last}")

    required = {"ISO", "Country", "WEO Subject Code", "2025"}
    missing = required - set(d.columns)
    if missing:
        raise ValueError(f"Legacy WEO file missing columns: {sorted(missing)}")
    x = d[d["WEO Subject Code"] == IMF_PPP_SERIES_CODE].copy()
    x["PPPGDP_2025_bn"] = x["2025"].map(_num)
    x = x[np.isfinite(x["PPPGDP_2025_bn"]) & (x["PPPGDP_2025_bn"] > 0)].copy()
    x["source_sheet"] = "legacy WEO country file"
    return x[["ISO", "Country", "PPPGDP_2025_bn", "source_sheet"]]


def load_imf_ppp_2025(path: Path) -> tuple[dict[str, float], pd.DataFrame]:
    """
    Construct a PPP 2025 world/NATO/non-NATO calibration.

    SA3 NEVER uses the original MER NATO share (0.5018) for model calibration.
    It computes the NATO share directly from 2025 country-level IMF PPPGDP:
        NATO share = sum(PPPGDP_2025 for 32 NATO members) / World PPPGDP_2025.
    Non-NATO GDP is World minus NATO. The non-NATO military burden is then
    inferred so the 2025 global military burden remains 2.5% while NATO remains
    at 2.7%.

    Preferred input is an IMF Data Explorer export containing a World PPPGDP
    row plus country PPPGDP rows. If no World row is present, the script uses
    the sum of all available positive country PPPGDP observations; this fallback
    is explicitly recorded in the audit.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Missing SA3 PPP input: {path}\n"
            "Export PPPGDP from the same IMF WEO vintage as the MER calibration "
            "including World and countries, or "
            "provide a legacy WEO entire-country file with PPPGDP."
        )

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm"}:
        d = _read_imf_ppp_data_explorer_xlsx(path)
        input_format = "IMF Data Explorer Excel export"
    elif suffix == ".csv":
        # Current IMF Data Explorer export format used for SA3.
        d = _read_imf_ppp_data_explorer_csv(path)
        input_format = "IMF Data Explorer CSV export"
    else:
        d = _read_imf_legacy_weo_file(path)
        input_format = "legacy WEO country file"

    # World row, if explicitly present in an Excel export.
    norm_country = d["Country"].astype(str).map(_normalise_text)
    world_mask = norm_country.isin({"world", "weoworld"})
    if world_mask.any():
        world_ppp_bn = float(d.loc[world_mask, "PPPGDP_2025_bn"].iloc[0])
        world_basis = "explicit IMF World PPPGDP row"
    else:
        country_rows = d[~world_mask].copy()
        # A reliable sum fallback needs broad country coverage.
        valid_iso = country_rows["ISO"].astype(str).str.fullmatch(r"[A-Z]{3}", na=False)
        country_rows = country_rows[valid_iso].copy()
        if country_rows["ISO"].nunique() < 150:
            raise ValueError(
                "PPP input has no explicit World row and fewer than 150 country "
                "PPPGDP observations. Include World in the IMF export."
            )
        world_ppp_bn = float(country_rows["PPPGDP_2025_bn"].sum())
        world_basis = f"sum of {country_rows['ISO'].nunique()} country PPPGDP rows"

    nato = d[d["ISO"].astype(str).isin(NATO_ISO3)].copy()
    present = set(nato["ISO"].dropna().astype(str))
    missing_nato = sorted(NATO_ISO3 - present)
    if missing_nato:
        raise ValueError(f"PPP input is missing NATO countries: {missing_nato}")

    nato_ppp_bn = float(nato["PPPGDP_2025_bn"].sum())
    s_nato = nato_ppp_bn / world_ppp_bn
    if not (0 < s_nato < 1):
        raise ValueError(f"Invalid PPP NATO share: {s_nato}")
    m_non = _infer_non_nato_burden_2025(s_nato)

    non_nato_ppp_bn = world_ppp_bn - nato_ppp_bn
    calibration = {
        "GDP_world_2025_PPP_trillion": world_ppp_bn / 1000.0,
        "GDP_NATO_2025_PPP_trillion": nato_ppp_bn / 1000.0,
        "GDP_nonNATO_2025_PPP_trillion": non_nato_ppp_bn / 1000.0,
        "s_NATO_GDP_2025_PPP": s_nato,
        "s_nonNATO_GDP_2025_PPP": 1.0 - s_nato,
        "m_nonNATO_2025_PPP": m_non,
    }

    audit = pd.DataFrame([
        {
            "input_file": str(path),
            "input_format": input_format,
            "world_GDP_basis": world_basis,
            "GDP_world_2025_PPP_trillion": calibration["GDP_world_2025_PPP_trillion"],
            "GDP_NATO_2025_PPP_trillion": calibration["GDP_NATO_2025_PPP_trillion"],
            "GDP_nonNATO_2025_PPP_trillion": calibration["GDP_nonNATO_2025_PPP_trillion"],
            "s_NATO_GDP_2025_PPP": s_nato,
            "s_nonNATO_GDP_2025_PPP": 1.0 - s_nato,
            "PPP_bloc_split_source": "2025 IMF PPPGDP country aggregation",
            "fixed_MER_share_0.5018_used_in_SA3": False,
            "original_MER_world_GDP_2025_trillion_for_comparison_only": 117.17,
            "original_MER_NATO_GDP_share_2025_for_comparison_only": 0.5018,
            "global_military_burden_2025": GLOBAL_MILITARY_BURDEN_2025,
            "NATO_military_burden_2025": NATO_MILITARY_BURDEN_2025,
            "implied_nonNATO_military_burden_2025_PPP": m_non,
            "NATO_country_count": len(present),
            "missing_NATO": ";".join(missing_nato),
        }
    ])
    return calibration, audit


def run_sa3_ppp_reference(ppp_cal: dict[str, float]) -> pd.DataFrame:
    """Run the original reference dynamics with a PPP 2025 calibration."""
    frames = []
    for s_mil in S_MIL_RANGE:
        params = ModelParams(
            b_global_2025=GLOBAL_MILITARY_BURDEN_2025,
            m_NATO_2025=NATO_MILITARY_BURDEN_2025,
            s_NATO_GDP=ppp_cal["s_NATO_GDP_2025_PPP"],
            GDP_world_2025=ppp_cal["GDP_world_2025_PPP_trillion"],
            s_mil_2025=s_mil,
            E_mil_2025=WORLD_GHG_2025 * s_mil,
            start_year=START_YEAR,
            end_year=END_YEAR,
        )
        calib = calibrate_intensities(params)
        for scenario in make_scenarios(calib["m_nonNATO_2025"]):
            f = simulate_timeseries(
                params=params,
                scenario=scenario,
                g_world=REFERENCE_PARAMS["g_world"],
                d_mil=REFERENCE_PARAMS["d_mil"],
                d_rest=REFERENCE_PARAMS["d_rest"],
                epsilon=REFERENCE_PARAMS["epsilon"],
            )
            f["analysis"] = "SA3_PPP_recalibrated_original_reference"
            f["GDP_valuation"] = "PPP"
            f["GDP_world_2025_calibrated"] = params.GDP_world_2025
            f["s_NATO_GDP_2025_calibrated"] = params.s_NATO_GDP
            f["m_nonNATO_2025_calibrated"] = calib["m_nonNATO_2025"]
            frames.append(f)

    out = pd.concat(frames, ignore_index=True)
    keys = ["s_mil_2025_assumed", "scenario_code"]
    out = out.sort_values(keys + ["year"]).copy()
    out["cum_E_mil_Gt"] = out.groupby(keys)["E_mil_Gt"].cumsum()

    base = out[out["scenario_code"] == "S0"][["s_mil_2025_assumed", "year", "E_mil_Gt"]].rename(
        columns={"E_mil_Gt": "E_mil_S0_Gt"}
    )
    out = out.merge(base, on=["s_mil_2025_assumed", "year"], how="left", validate="many_to_one")
    out["incremental_E_mil_vs_S0_Gt"] = out["E_mil_Gt"] - out["E_mil_S0_Gt"]
    out["cum_incremental_E_mil_vs_S0_Gt"] = out.groupby(keys)["incremental_E_mil_vs_S0_Gt"].cumsum()
    return out.sort_values(keys + ["year"]).reset_index(drop=True)


def build_sa3_budget(df: pd.DataFrame) -> pd.DataFrame:
    d = df[(df["year"] >= BUDGET_START_YEAR) & (df["year"] <= BUDGET_END_YEAR)].copy()
    keys = ["scenario_code", "scenario", "s_mil_2025_assumed"]
    out = d.groupby(keys, as_index=False)["E_mil_Gt"].sum().rename(columns={"E_mil_Gt": "cum_E_mil_Gt"})
    out["cum_E_mil_CO2_Gt"] = MILITARY_CO2_FRACTION * out["cum_E_mil_Gt"]
    out["pct_1p5_budget_used"] = 100 * out["cum_E_mil_CO2_Gt"] / BUDGET_1P5_GTCO2
    out["pct_2C_budget_used"] = 100 * out["cum_E_mil_CO2_Gt"] / BUDGET_2C_GTCO2
    return out


def _sa3_focus(df: pd.DataFrame) -> pd.DataFrame:
    return df[np.isclose(df["s_mil_2025_assumed"], FIGURE_BASELINE)].copy()


def make_sa3_fig_1(df: pd.DataFrame, ref: pd.DataFrame, outdir: Path):
    d = _sa3_focus(df)
    r = ref[ref["year"] <= 2035]
    d = d[d["year"] <= 2035]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for ax, sc in zip(axes.flat, SCENARIO_ORDER):
        z = d[d["scenario_code"] == sc]
        q = r[r["scenario_code"] == sc]
        ax.plot(z["year"], z["E_mil_Gt"], lw=2.4, label="PPP recalibration")
        ax.plot(q["year"], q["E_mil_Gt"], color="black", ls="--", lw=2.0, label="Original MER reference")
        ax.set_title(SCENARIO_TITLES[sc], fontsize=10.5); _style_axis(ax)
    axes[0,0].set_ylabel("Military emissions (GtCO$_2$e/yr)"); axes[1,0].set_ylabel("Military emissions (GtCO$_2$e/yr)")
    axes[1,0].set_xlabel("Year"); axes[1,1].set_xlabel("Year")
    fig.legend(handles=[Line2D([0],[0],lw=2.4,label="PPP recalibration"), _reference_handle()], loc="lower center", ncol=2, frameon=False)
    fig.suptitle("SA3: Original reference model recalibrated in PPP\nFigure 1. Absolute military GHG emissions (2025–2035)", fontsize=13)
    fig.tight_layout(rect=[0,0.08,1,0.93]); fig.savefig(outdir/"Fig_1.pdf",bbox_inches="tight"); fig.savefig(outdir/"Fig_1.png",dpi=300,bbox_inches="tight"); plt.close(fig)


def make_sa3_fig_2(df: pd.DataFrame, ref: pd.DataFrame, outdir: Path):
    d = _sa3_focus(df); d=d[d["year"]<=2035]; r=ref[ref["year"]<=2035]
    fig, axes = plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True)
    for ax,sc in zip(axes.flat,SCENARIO_ORDER):
        z=d[d["scenario_code"]==sc]; q=r[r["scenario_code"]==sc]
        ax.plot(z["year"],100*z["s_mil"],lw=2.4)
        ax.plot(q["year"],100*q["s_mil"],color="black",ls="--",lw=2.0)
        ax.set_title(SCENARIO_TITLES[sc],fontsize=10.5); _style_axis(ax)
    axes[0,0].set_ylabel("Military share of global GHG (%)"); axes[1,0].set_ylabel("Military share of global GHG (%)")
    axes[1,0].set_xlabel("Year"); axes[1,1].set_xlabel("Year")
    fig.legend(handles=[Line2D([0],[0],lw=2.4,label="PPP recalibration"),_reference_handle()],loc="lower center",ncol=2,frameon=False)
    fig.suptitle("SA3: Original reference model recalibrated in PPP\nFigure 2. Military share of global GHG emissions (2025–2035)",fontsize=13)
    fig.tight_layout(rect=[0,0.08,1,0.93]); fig.savefig(outdir/"Fig_2.pdf",bbox_inches="tight"); fig.savefig(outdir/"Fig_2.png",dpi=300,bbox_inches="tight"); plt.close(fig)


def make_sa3_fig_3(df: pd.DataFrame, ref: pd.DataFrame, outdir: Path):
    d=_sa3_focus(df)
    rr=ref.sort_values(["scenario_code","year"]).copy(); rr["cum_E_mil_Gt"]=rr.groupby("scenario_code")["E_mil_Gt"].cumsum()
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True)
    for ax,sc in zip(axes.flat,SCENARIO_ORDER):
        z=d[d["scenario_code"]==sc]; q=rr[rr["scenario_code"]==sc]
        ax.plot(z["year"],z["cum_E_mil_Gt"],lw=2.4); ax.plot(q["year"],q["cum_E_mil_Gt"],color="black",ls="--",lw=2.0)
        ax.set_title(SCENARIO_TITLES[sc],fontsize=10.5); _style_axis(ax)
    axes[0,0].set_ylabel("Cumulative military emissions (GtCO$_2$e)"); axes[1,0].set_ylabel("Cumulative military emissions (GtCO$_2$e)")
    axes[1,0].set_xlabel("Year"); axes[1,1].set_xlabel("Year")
    fig.legend(handles=[Line2D([0],[0],lw=2.4,label="PPP recalibration"),_reference_handle()],loc="lower center",ncol=2,frameon=False)
    fig.suptitle("SA3: Original reference model recalibrated in PPP\nFigure 3. Cumulative military GHG emissions (2025–2050)",fontsize=13)
    fig.tight_layout(rect=[0,0.08,1,0.93]); fig.savefig(outdir/"Fig_3.pdf",bbox_inches="tight"); fig.savefig(outdir/"Fig_3.png",dpi=300,bbox_inches="tight"); plt.close(fig)


def make_sa3_fig_4(df: pd.DataFrame, ref: pd.DataFrame, outdir: Path):
    d=_sa3_focus(df)
    rr=ref.sort_values(["scenario_code","year"]).copy(); r0=rr[rr["scenario_code"]=="S0"][["year","E_mil_Gt"]].rename(columns={"E_mil_Gt":"E0"}); rr=rr.merge(r0,on="year",how="left"); rr["inc"]=rr["E_mil_Gt"]-rr["E0"]; rr["cum_inc"]=rr.groupby("scenario_code")["inc"].cumsum()
    fig,axes=plt.subplots(1,3,figsize=(14,4.8),sharey=True)
    for ax,sc in zip(axes,["S1","S2","S3"]):
        z=d[d["scenario_code"]==sc]; q=rr[rr["scenario_code"]==sc]
        ax.plot(z["year"],z["cum_incremental_E_mil_vs_S0_Gt"],lw=2.4); ax.plot(q["year"],q["cum_inc"],color="black",ls="--",lw=2.0)
        ax.axhline(0,color="black",lw=.8,alpha=.6); ax.set_title(SCENARIO_TITLES[sc],fontsize=10); ax.set_xlabel("Year"); _style_axis(ax)
    axes[0].set_ylabel(r"$\Delta$ cumulative military emissions vs S0 (GtCO$_2$e)")
    fig.legend(handles=[Line2D([0],[0],lw=2.4,label="PPP recalibration"),_reference_handle()],loc="lower center",ncol=2,frameon=False)
    fig.suptitle("SA3: Original reference model recalibrated in PPP\nFigure 4. Incremental cumulative military emissions relative to S0",fontsize=13)
    fig.tight_layout(rect=[0,0.14,1,.91]); fig.savefig(outdir/"Fig_4.pdf",bbox_inches="tight"); fig.savefig(outdir/"Fig_4.png",dpi=300,bbox_inches="tight"); plt.close(fig)


def make_sa3_fig_5(budget: pd.DataFrame, ref_budget: pd.DataFrame, outdir: Path):
    d=budget[np.isclose(budget["s_mil_2025_assumed"],FIGURE_BASELINE)].copy()
    fig,axes=plt.subplots(2,2,figsize=(11,8.5),sharey=True)
    maxy=0
    for ax,sc in zip(axes.flat,SCENARIO_ORDER):
        p=d[d["scenario_code"]==sc].iloc[0]; r=ref_budget[ref_budget["scenario_code"]==sc].iloc[0]
        vals=[float(r["pct_1p5_budget_used"]),float(p["pct_1p5_budget_used"])]
        vals2=[float(r["pct_2C_budget_used"]),float(p["pct_2C_budget_used"])]
        x=np.arange(2); maxy=max(maxy,max(vals)); ax.bar(x,vals,alpha=.85); ax.scatter(x,vals2,marker="D",s=40,color="black",zorder=4); ax.axhline(100,color="grey",ls="--",lw=1)
        ax.set_xticks(x); ax.set_xticklabels(["Original MER","PPP recalibration"]); ax.set_title(SCENARIO_TITLES[sc],fontsize=10.5); _style_axis(ax)
    for ax in axes.flat: ax.set_ylim(0,max(110,maxy*1.08))
    axes[0,0].set_ylabel("Share of remaining carbon budget (%)"); axes[1,0].set_ylabel("Share of remaining carbon budget (%)")
    fig.suptitle(f"SA3: Original reference model recalibrated in PPP\nFigure 5. Military CO$_2$ relative to remaining carbon budgets ({BUDGET_START_YEAR}–{BUDGET_END_YEAR})",fontsize=13)
    fig.tight_layout(rect=[0,0.02,1,.92]); fig.savefig(outdir/"Fig_5.pdf",bbox_inches="tight"); fig.savefig(outdir/"Fig_5.png",dpi=300,bbox_inches="tight"); plt.close(fig)


def save_sa3_outputs(ppp_file: Path, reference: pd.DataFrame, root: Path) -> pd.DataFrame:
    outdir=root/"SA3_original_model_PPP_recalibration"; outdir.mkdir(parents=True,exist_ok=True)
    cal,audit=load_imf_ppp_2025(ppp_file); audit.to_csv(outdir/"PPP_calibration_audit.csv",index=False)
    results=run_sa3_ppp_reference(cal); budget=build_sa3_budget(results)
    results[results["year"]<=2035].to_csv(outdir/"generated_data_2025-2035.csv",index=False); results.to_csv(outdir/"generated_data_2025-2050.csv",index=False); budget.to_csv(outdir/"budget_summary_2026-2050.csv",index=False)
    ref_budget=build_original_reference_budget(reference)
    make_sa3_fig_1(results,reference,outdir); make_sa3_fig_2(results,reference,outdir); make_sa3_fig_3(results,reference,outdir); make_sa3_fig_4(results,reference,outdir); make_sa3_fig_5(budget,ref_budget,outdir)
    print(f"Saved SA3 CSVs and Fig_1 ... Fig_5 to {outdir}")
    return results



# ============================================================================
# COMBINED OUTPUTS: ACTUAL PROJECTED SCENARIOS ONLY (NO AGGREGATION)
# ============================================================================

COMBINED_MODEL_LABELS = {
    "Original": (
        "Original model — MER reference parameterisation "
        "(2025 military footprint 5.5%; g=3%/yr; d_mil=1%/yr; "
        "d_rest=1%/yr; epsilon=0)"
    ),
    "SA1": (
        "SA1 — AR6 pathways; fixed NATO GDP share "
        "(s_NATO=0.5018)"
    ),
    "SA2": (
        "SA2 — AR6 pathways; evolving NATO GDP share from "
        "SSP Basic Drivers v3.2"
    ),
    "SA3": (
        "SA3 — PPP recalibration using IMF WEO PPPGDP "
        "(g=3%/yr; d_mil=1%/yr; d_rest=1%/yr; epsilon=0)"
    ),
}

COMBINED_SCENARIO_COLORS = {
    "S0": "#4d4d4d",
    "S1": "#2874a6",
    "S2": "#c27c0e",
    "S3": "#922b21",
}

COMBINED_PANEL_TITLES = {
    "Original": "Original model — MER reference parameterisation",
    "SA1": "SA1 — AR6 pathways; fixed NATO GDP share",
    "SA2": "SA2 — AR6 pathways; evolving NATO GDP share",
    "SA3": "SA3 — PPP recalibration",
}

COMBINED_SCENARIO_LABELS = {
    "S0": "S0: Baseline",
    "S1": "S1: NATO 3.5%",
    "S2": "S2: NATO/non-NATO 3.5%",
    "S3": "S3: NATO 5%; non-NATO 3.5%",
}


def _combined_add_original_cumulative(reference: pd.DataFrame) -> pd.DataFrame:
    """Add cumulative quantities to the original reference S0-S3 trajectories."""
    d = reference.sort_values(["scenario_code", "year"]).copy()
    d["cum_E_mil_Gt"] = d.groupby("scenario_code")["E_mil_Gt"].cumsum()
    s0 = d[d["scenario_code"] == "S0"][["year", "E_mil_Gt"]].rename(
        columns={"E_mil_Gt": "E_mil_S0_Gt"}
    )
    d = d.merge(s0, on="year", how="left", validate="many_to_one")
    d["incremental_E_mil_vs_S0_Gt"] = d["E_mil_Gt"] - d["E_mil_S0_Gt"]
    d["cum_incremental_E_mil_vs_S0_Gt"] = d.groupby("scenario_code")[
        "incremental_E_mil_vs_S0_Gt"
    ].cumsum()
    return d


def _combined_prepare_trajectory_tables(
    reference: pd.DataFrame,
    sa1: pd.DataFrame,
    sa2: pd.DataFrame,
    sa3: pd.DataFrame,
) -> pd.DataFrame:
    """
    Stack all ACTUAL projected trajectories used in the Combined figures.

    There is no median, percentile, ribbon, min/max or any other aggregation.
    The Combined folder uses only the central 5.5% 2025 military-footprint
    calibration, as in the manuscript sensitivity figures.

    Included trajectories:
      * Original model reference parameterisation: S0-S3 (4 trajectories).
      * SA1: 5 AR6 pathways x S0-S3 (20 trajectories).
      * SA2: 5 AR6/SSP pathways x S0-S3 (20 trajectories).
      * SA3: S0-S3 (4 trajectories).
    """
    ref = _combined_add_original_cumulative(reference)
    ref = ref[np.isclose(ref["s_mil_2025_assumed"], FIGURE_BASELINE)].copy()
    s1 = sa1[np.isclose(sa1["s_mil_2025_assumed"], FIGURE_BASELINE)].copy()
    s2 = sa2[np.isclose(sa2["s_mil_2025_assumed"], FIGURE_BASELINE)].copy()
    s3 = sa3[np.isclose(sa3["s_mil_2025_assumed"], FIGURE_BASELINE)].copy()

    tables = []

    ref["combined_model"] = "Original"
    ref["combined_model_label"] = COMBINED_MODEL_LABELS["Original"]
    ref["projected_pathway"] = "Original model reference parameterisation"
    ref["trajectory_label"] = ref.apply(
        lambda r: f"{r['combined_model_label']} — {COMBINED_SCENARIO_LABELS[r['scenario_code']]}",
        axis=1,
    )
    tables.append(ref)

    s1["combined_model"] = "SA1"
    s1["combined_model_label"] = COMBINED_MODEL_LABELS["SA1"]
    s1["projected_pathway"] = s1["ssp"].astype(str)
    s1["trajectory_label"] = s1.apply(
        lambda r: (
            f"{r['combined_model_label']} — {r['ssp']} — "
            f"{r['ar6_model']} / {r['ar6_scenario']} — "
            f"{COMBINED_SCENARIO_LABELS[r['scenario_code']]}"
        ),
        axis=1,
    )
    tables.append(s1)

    s2["combined_model"] = "SA2"
    s2["combined_model_label"] = COMBINED_MODEL_LABELS["SA2"]
    s2["projected_pathway"] = s2["ssp"].astype(str)
    s2["trajectory_label"] = s2.apply(
        lambda r: (
            f"{r['combined_model_label']} — {r['ssp']} — "
            f"{r['ar6_model']} / {r['ar6_scenario']} — "
            f"{COMBINED_SCENARIO_LABELS[r['scenario_code']]}"
        ),
        axis=1,
    )
    tables.append(s2)

    s3["combined_model"] = "SA3"
    s3["combined_model_label"] = COMBINED_MODEL_LABELS["SA3"]
    s3["projected_pathway"] = "PPP recalibrated original-model dynamics"
    s3["trajectory_label"] = s3.apply(
        lambda r: f"{r['combined_model_label']} — {COMBINED_SCENARIO_LABELS[r['scenario_code']]}",
        axis=1,
    )
    tables.append(s3)

    combined = pd.concat(tables, ignore_index=True, sort=False)
    order = {"Original": 0, "SA1": 1, "SA2": 2, "SA3": 3}
    combined["_model_order"] = combined["combined_model"].map(order)
    combined["_scenario_order"] = combined["scenario_code"].map(
        {s: i for i, s in enumerate(SCENARIO_ORDER)}
    )
    combined = combined.sort_values(
        ["_model_order", "projected_pathway", "_scenario_order", "year"]
    ).drop(columns=["_model_order", "_scenario_order"])
    return combined.reset_index(drop=True)


def _combined_prepare_budget_tables(
    reference: pd.DataFrame,
    sa1: pd.DataFrame,
    sa2: pd.DataFrame,
    sa3: pd.DataFrame,
) -> pd.DataFrame:
    """Stack actual budget results for the same 48 projected trajectories."""
    rb = build_original_reference_budget(reference).copy()
    b1 = build_budget_summary(sa1)
    b1 = b1[np.isclose(b1["s_mil_2025_assumed"], FIGURE_BASELINE)].copy()
    b2 = build_budget_summary(sa2)
    b2 = b2[np.isclose(b2["s_mil_2025_assumed"], FIGURE_BASELINE)].copy()
    b3 = build_sa3_budget(sa3)
    b3 = b3[np.isclose(b3["s_mil_2025_assumed"], FIGURE_BASELINE)].copy()

    rb["combined_model"] = "Original"
    rb["combined_model_label"] = COMBINED_MODEL_LABELS["Original"]
    rb["projected_pathway"] = "Original model reference parameterisation"
    rb["trajectory_label"] = rb.apply(
        lambda r: f"{r['combined_model_label']} — {COMBINED_SCENARIO_LABELS[r['scenario_code']]}",
        axis=1,
    )

    for df, key in [(b1, "SA1"), (b2, "SA2")]:
        df["combined_model"] = key
        df["combined_model_label"] = COMBINED_MODEL_LABELS[key]
        df["projected_pathway"] = df["ssp"].astype(str)
        df["trajectory_label"] = df.apply(
            lambda r: (
                f"{r['combined_model_label']} — {r['ssp']} — "
                f"{COMBINED_SCENARIO_LABELS[r['scenario_code']]}"
            ),
            axis=1,
        )

    b3["combined_model"] = "SA3"
    b3["combined_model_label"] = COMBINED_MODEL_LABELS["SA3"]
    b3["projected_pathway"] = "PPP recalibrated original-model dynamics"
    b3["trajectory_label"] = b3.apply(
        lambda r: f"{r['combined_model_label']} — {COMBINED_SCENARIO_LABELS[r['scenario_code']]}",
        axis=1,
    )

    return pd.concat([rb, b1, b2, b3], ignore_index=True, sort=False)



def _combined_4x4_axes(figsize=(18.5, 14.5), sharex=True, sharey=True):
    """Return a literal 4 model x 4 military-burden scenario grid."""
    fig, axes = plt.subplots(
        4, 4, figsize=figsize, sharex=sharex, sharey=sharey,
        squeeze=False,
    )
    model_order = ["Original", "SA1", "SA2", "SA3"]
    scenario_order = ["S0", "S1", "S2", "S3"]
    return fig, axes, model_order, scenario_order


def _combined_pathway_handles_4x4(combined: pd.DataFrame):
    handles = []
    source = combined[combined["combined_model"] == "SA1"]
    for p in AR6_PATHWAYS:
        z = source[source["projected_pathway"] == p]
        if not z.empty and "ar6_model" in z.columns and "ar6_scenario" in z.columns:
            model = str(z["ar6_model"].dropna().iloc[0])
            scenario = str(z["ar6_scenario"].dropna().iloc[0])
            label = f"{p} — {model} / {scenario}"
        else:
            label = p
        handles.append(
            Line2D(
                [0], [0],
                color=PATHWAY_COLORS[p],
                linestyle=PATHWAY_STYLES[p],
                lw=2.0,
                label=label,
            )
        )
    return handles


def _combined_model_row_labels():
    # Concise labels harmonised with the Methods sensitivity-analysis headings.
    return {
        "Original": "Main model\nMER reference parameterisation",
        "SA1": "SA1\nAR6 pathways\nFixed NATO GDP share",
        "SA2": "SA2\nAR6 pathways\nEvolving NATO GDP share",
        "SA3": "SA3\nPPP recalibration",
    }


def _combined_scenario_column_labels():
    return {
        "S0": "S0: Baseline",
        "S1": "S1: NATO 3.5%",
        "S2": "S2: NATO/non-NATO 3.5%",
        "S3": "S3: NATO 5%; non-NATO 3.5%",
    }


def _decorate_4x4_grid(fig, axes, model_order, scenario_order):
    """Add scenario headers and clearly separated model-row labels."""
    col_labels = _combined_scenario_column_labels()
    row_labels = _combined_model_row_labels()

    for j, sc in enumerate(scenario_order):
        axes[0, j].set_title(col_labels[sc], fontsize=10.5, pad=10)

    # Fixed row-centre positions in figure coordinates. These are deliberately
    # horizontal so the 4 model rows are immediately visible.
    row_y = [0.835, 0.625, 0.415, 0.205]
    for key, y in zip(model_order, row_y):
        fig.text(
            0.012, y, row_labels[key],
            rotation=0, va="center", ha="left",
            fontsize=9.4, fontweight="bold", linespacing=1.15,
        )


def _plot_4x4_timeseries_cell(
    ax,
    combined: pd.DataFrame,
    model_key: str,
    scenario_code: str,
    value_col: str,
):
    """Plot all actual trajectories for one model x escalation-scenario cell."""
    d = combined[
        (combined["combined_model"] == model_key)
        & (combined["scenario_code"] == scenario_code)
    ].copy()

    if model_key in {"SA1", "SA2"}:
        # Five actual AR6 projections for this one escalation scenario.
        for p in AR6_PATHWAYS:
            z = d[d["projected_pathway"] == p].sort_values("year")
            if z.empty:
                continue
            ax.plot(
                z["year"], z[value_col],
                color=PATHWAY_COLORS[p],
                linestyle=PATHWAY_STYLES[p],
                lw=1.9,
            )
    else:
        # One actual trajectory for this escalation scenario.
        z = d.sort_values("year")
        color = "black" if model_key == "Original" else "#6b8e23"
        ax.plot(z["year"], z[value_col], color=color, lw=2.35)

    _style_axis(ax)


def _finish_4x4_timeseries(
    fig,
    axes,
    model_order,
    scenario_order,
    combined,
    title,
    ylabel,
    xmin,
    xmax,
    legend=True,
):
    _decorate_4x4_grid(fig, axes, model_order, scenario_order)

    for i in range(4):
        for j in range(4):
            axes[i, j].set_xlim(xmin, xmax)
            if j == 0:
                axes[i, j].set_ylabel(ylabel)
            if i == 3:
                axes[i, j].set_xlabel("Year")

    if legend:
        fig.legend(
            handles=_combined_pathway_handles_4x4(combined),
            loc="lower center", ncol=3, frameon=False, fontsize=9,
        )
        bottom = 0.105
    else:
        bottom = 0.055

    fig.suptitle(title, fontsize=14, y=0.995)
    fig.tight_layout(rect=[0.145, bottom, 1, 0.965], h_pad=1.25, w_pad=0.9)


def make_combined_fig_1(combined: pd.DataFrame, outdir: Path):
    fig, axes, model_order, scenario_order = _combined_4x4_axes()
    for i, key in enumerate(model_order):
        for j, sc in enumerate(scenario_order):
            _plot_4x4_timeseries_cell(ax=axes[i, j], combined=combined, model_key=key, scenario_code=sc, value_col="E_mil_Gt")

    _finish_4x4_timeseries(
        fig, axes, model_order, scenario_order, combined,
        "Figure 1. Absolute military GHG emissions (2025–2035); 2025 military footprint = 5.5%",
        "Military emissions (GtCO$_2$e/yr)",
        2025, 2035,
    )
    fig.savefig(outdir / "Fig_1.pdf", bbox_inches="tight")
    fig.savefig(outdir / "Fig_1.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_combined_fig_2(combined: pd.DataFrame, outdir: Path):
    d = combined.copy()
    d["mil_share_pct"] = 100.0 * d["s_mil"]

    fig, axes, model_order, scenario_order = _combined_4x4_axes()
    for i, key in enumerate(model_order):
        for j, sc in enumerate(scenario_order):
            _plot_4x4_timeseries_cell(ax=axes[i, j], combined=d, model_key=key, scenario_code=sc, value_col="mil_share_pct")

    _finish_4x4_timeseries(
        fig, axes, model_order, scenario_order, d,
        "Figure 2. Military share of global GHG emissions (2025–2035); 2025 military footprint = 5.5%",
        "Military share of global GHG (%)",
        2025, 2035,
    )
    fig.savefig(outdir / "Fig_2.pdf", bbox_inches="tight")
    fig.savefig(outdir / "Fig_2.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_combined_fig_3(combined: pd.DataFrame, outdir: Path):
    fig, axes, model_order, scenario_order = _combined_4x4_axes()
    for i, key in enumerate(model_order):
        for j, sc in enumerate(scenario_order):
            _plot_4x4_timeseries_cell(ax=axes[i, j], combined=combined, model_key=key, scenario_code=sc, value_col="cum_E_mil_Gt")

    _finish_4x4_timeseries(
        fig, axes, model_order, scenario_order, combined,
        "Figure 3. Cumulative military GHG emissions (2025–2050); 2025 military footprint = 5.5%",
        "Cumulative military emissions (GtCO$_2$e)",
        2025, 2050,
    )
    fig.savefig(outdir / "Fig_3.pdf", bbox_inches="tight")
    fig.savefig(outdir / "Fig_3.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_combined_fig_4(combined: pd.DataFrame, outdir: Path):
    # Keep the literal 4x4 model x escalation-scenario structure. S0 is therefore
    # shown explicitly as a zero incremental trajectory in the first column.
    fig, axes, model_order, scenario_order = _combined_4x4_axes()
    for i, key in enumerate(model_order):
        for j, sc in enumerate(scenario_order):
            _plot_4x4_timeseries_cell(
                ax=axes[i, j], combined=combined, model_key=key,
                scenario_code=sc, value_col="cum_incremental_E_mil_vs_S0_Gt"
            )
            axes[i, j].axhline(0, color="black", lw=0.7, alpha=0.45)

    _finish_4x4_timeseries(
        fig, axes, model_order, scenario_order, combined,
        "Figure 4. Incremental cumulative military emissions relative to S0 (2025–2050); 2025 military footprint = 5.5%",
        r"$\Delta$ cumulative military emissions vs S0 (GtCO$_2$e)",
        2025, 2050,
    )
    fig.savefig(outdir / "Fig_4.pdf", bbox_inches="tight")
    fig.savefig(outdir / "Fig_4.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_combined_fig_5(budget: pd.DataFrame, outdir: Path):
    """
    Literal 4 model x 4 escalation-scenario grid, preserving the original
    bar-based Figure 5 representation.

    Each cell is one model/scenario combination:
      - Main model and SA3: one actual bar.
      - SA1 and SA2: five actual bars (one per AR6 pathway), no aggregation.

    Dark bar = share of 1.5°C budget used.
    Light stacked segment = remaining 1.5°C budget up to 100%.
    Diamond = share of 2°C budget used.
    """
    fig, axes, model_order, scenario_order = _combined_4x4_axes(
        figsize=(18.5, 14.5), sharex=False, sharey=True
    )

    global_max = 0.0

    for i, key in enumerate(model_order):
        for j, sc in enumerate(scenario_order):
            ax = axes[i, j]
            d = budget[
                (budget["combined_model"] == key)
                & (budget["scenario_code"] == sc)
            ].copy()

            if key in {"SA1", "SA2"}:
                d["_ord"] = d["projected_pathway"].map({p: k for k, p in enumerate(AR6_PATHWAYS)})
                d = d.sort_values("_ord")
                x = np.arange(len(AR6_PATHWAYS), dtype=float)
                used = d["pct_1p5_budget_used"].to_numpy(float)
                used2 = d["pct_2C_budget_used"].to_numpy(float)
                remain = np.clip(100.0 - used, 0.0, None)
                colors = [PATHWAY_COLORS[p] for p in d["projected_pathway"]]
                ax.bar(x, used, width=0.70, color=colors, alpha=0.88, zorder=2)
                ax.bar(x, remain, width=0.70, bottom=used, color=colors, alpha=0.18, zorder=1)
                ax.scatter(x, used2, marker="D", s=25, color="black", zorder=4)
                ax.set_xticks(x)
                ax.set_xticklabels(AR6_PATHWAYS, rotation=45, ha="right", fontsize=7.5)
            else:
                used = d["pct_1p5_budget_used"].to_numpy(float)
                used2 = d["pct_2C_budget_used"].to_numpy(float)
                remain = np.clip(100.0 - used, 0.0, None)
                x = np.array([0.0])
                color = "black" if key == "Original" else "#6b8e23"
                ax.bar(x, used, width=0.55, color=color, alpha=0.88, zorder=2)
                ax.bar(x, remain, width=0.55, bottom=used, color=color, alpha=0.18, zorder=1)
                ax.scatter(x, used2, marker="D", s=30, color="black", zorder=4)
                ax.set_xticks([])

            if len(used):
                global_max = max(global_max, float(np.nanmax(used)))

            ax.axhline(100, color="grey", linestyle="--", lw=0.9)
            _style_axis(ax)

    _decorate_4x4_grid(fig, axes, model_order, scenario_order)

    ymax = max(110.0, global_max * 1.08)
    for i in range(4):
        for j in range(4):
            axes[i, j].set_ylim(0, ymax)
            if j == 0:
                axes[i, j].set_ylabel("Share of remaining carbon budget (%)")

    handles = [
        Patch(facecolor="grey", alpha=0.88, label="Share of 1.5°C budget used"),
        Patch(facecolor="grey", alpha=0.18, label="1.5°C budget remaining"),
        Line2D([0], [0], marker="D", color="black", lw=0, label="Share of 2°C budget used"),
    ] + [
        Patch(facecolor=PATHWAY_COLORS[p], alpha=0.88, label=p)
        for p in AR6_PATHWAYS
    ]

    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle(
        f"Figure 5. Military CO$_2$ emissions relative to remaining global carbon budgets, "
        f"{BUDGET_START_YEAR}–{BUDGET_END_YEAR}; 2025 military footprint = 5.5%",
        fontsize=14, y=0.995,
    )
    fig.tight_layout(rect=[0.145, 0.115, 1, 0.965], h_pad=1.25, w_pad=0.9)
    fig.savefig(outdir / "Fig_5.pdf", bbox_inches="tight")
    fig.savefig(outdir / "Fig_5.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_combined_outputs(
    reference: pd.DataFrame,
    sa1: pd.DataFrame,
    sa2: pd.DataFrame,
    sa3: pd.DataFrame,
    root: Path,
) -> None:
    """Create root/Combined with unaggregated actual trajectories and Figures 1-5."""
    outdir = root / "Combined"
    outdir.mkdir(parents=True, exist_ok=True)

    combined = _combined_prepare_trajectory_tables(reference, sa1, sa2, sa3)
    combined_budget = _combined_prepare_budget_tables(reference, sa1, sa2, sa3)

    combined[combined["year"] <= 2035].to_csv(
        outdir / "combined_actual_projected_scenarios_2025-2035.csv", index=False
    )
    combined.to_csv(
        outdir / "combined_actual_projected_scenarios_2025-2050.csv", index=False
    )
    combined_budget.to_csv(
        outdir / "combined_actual_projected_budget_results_2026-2050.csv", index=False
    )

    # One row per unique trajectory for transparent labels / provenance.
    label_cols = [
        "combined_model", "combined_model_label", "projected_pathway",
        "ar6_model", "ar6_scenario",
        "scenario_code", "scenario", "trajectory_label",
    ]
    label_cols = [c for c in label_cols if c in combined.columns]
    combined[label_cols].drop_duplicates().to_csv(
        outdir / "combined_trajectory_labels.csv", index=False
    )

    make_combined_fig_1(combined, outdir)
    make_combined_fig_2(combined, outdir)
    make_combined_fig_3(combined, outdir)
    make_combined_fig_4(combined, outdir)
    make_combined_fig_5(combined_budget, outdir)

    print(f"Saved Combined actual-scenario CSVs and Fig_1 ... Fig_5 to {outdir}")


# ============================================================================
# MAIN: SA1 + SA2 + SA3
# ============================================================================

def main_three_sensitivities():
    parser=argparse.ArgumentParser(description="Original reference case + SA1 AR6 fixed split + SA2 AR6 dynamic v3.2 split + SA3 PPP recalibration.")
    parser.add_argument("--ar6-world",type=Path,default=DEFAULT_AR6_WORLD_FILE,help="AR6 World v1.1 CSV or ZIP.")
    parser.add_argument("--basic-drivers",type=Path,default=DEFAULT_BASIC_DRIVERS_FILE,help="SSP Basic Drivers v3.2 xlsx for SA2 bloc shares.")
    parser.add_argument("--imf-ppp",type=Path,default=DEFAULT_IMF_PPP_FILE,help="IMF WEO Data Explorer PPPGDP export (CSV or Excel), containing World + countries and year 2025. Also accepts legacy WEO entire-country file.")
    parser.add_argument("--output-dir",type=Path,default=ROOT_OUTPUT_DIR,help="Root output directory.")
    args=parser.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)

    # If the default SA3 filename is absent, automatically detect a single IMF
    # Data Explorer WEO CSV in the script directory (e.g. dataset_...WEO_9.0.0.csv).
    if not args.imf_ppp.exists() and args.imf_ppp == DEFAULT_IMF_PPP_FILE:
        candidates = sorted(DATA_INPUT_DIR.glob("dataset_*IMF.RES_WEO*.csv"))
        if len(candidates) == 1:
            args.imf_ppp = candidates[0]
            print(f"Auto-detected SA3 IMF PPP file: {args.imf_ppp.name}")
        elif len(candidates) > 1:
            print("Multiple IMF WEO Data Explorer CSV files found; SA3 will require --imf-ppp to select one.")

    # Original MER reference only.
    reference=run_original_reference_case(END_YEAR)
    reference.to_csv(args.output_dir/"original_MER_reference_case_2025-2050.csv",index=False)
    build_original_reference_budget(reference).to_csv(args.output_dir/"original_MER_reference_budget_2026-2050.csv",index=False)

    # Shared AR6 World source: GDP and GHG must be from same Model/Scenario.
    raw=load_ar6_world(args.ar6_world); ar6_base,ar6_audit=build_ar6_world_base(raw)
    ar6_audit.to_csv(args.output_dir/"AR6_same_model_GDP_GHG_source_audit.csv",index=False)

    # SA1: AR6 World, fixed original bloc split.
    cfg1=SA_CONFIG["SA1_AR6_fixed_split"]
    drv1=build_sensitivity_drivers(ar6_base,rebase_gdp=False,bloc_method="fixed",country_shares=None,analysis_name="SA1_AR6_fixed_split")
    sa1_results=save_sensitivity_outputs("SA1_AR6_fixed_split",cfg1,drv1,reference,args.output_dir)

    # SA2: same AR6 World GDP/GHG + dynamic v3.2 country GDP shares.
    shares,audit2=load_basic_drivers_country_gdp(args.basic_drivers)
    audit2.to_csv(args.output_dir/"SA2_basic_drivers_country_GDP_audit.csv",index=False)
    cfg2=SA_CONFIG["SA2_AR6_dynamic_v32_split"]
    drv2=build_sensitivity_drivers(ar6_base,rebase_gdp=False,bloc_method="country",country_shares=shares,analysis_name="SA2_AR6_dynamic_v32_split")
    sa2_results=save_sensitivity_outputs("SA2_AR6_dynamic_v32_split",cfg2,drv2,reference,args.output_dir)

    # SA3: original reference dynamics, true PPP baseline recalibration.
    # If the PPP input is not present, SA1/SA2 remain valid and the script
    # finishes cleanly instead of failing. Re-run after adding the PPPGDP file.
    sa3_ran = False
    sa3_results = None
    if args.imf_ppp.exists():
        sa3_results=save_sa3_outputs(args.imf_ppp,reference,args.output_dir)
        sa3_ran = True
        save_combined_outputs(reference, sa1_results, sa2_results, sa3_results, args.output_dir)
    else:
        print("\nSA3 not run: PPP input file not found:")
        print(f"  {args.imf_ppp}")
        print("SA1 and SA2 have completed successfully.")
        print("To run SA3, export IMF WEO 2025 PPPGDP (current-price PPP "
              "international dollars) for World and all countries from the "
              "same WEO vintage used for the MER calibration, save it as "
              "IMF_WEO_PPP_2025.xlsx beside this script, and run again.")

    if sa3_ran:
        print("\nCompleted SA1, SA2 and SA3.")
    else:
        print("\nCompleted SA1 and SA2; SA3 skipped because its PPP input is absent.")
    print(f"Outputs: {args.output_dir.resolve()}")




# ============================================================================
# OVERRIDE COMBINED FIGURES: 3x4 layout with reference-case overlay in each plot
# ============================================================================

def _combined_3x4_axes(figsize=(11, 11), sharex=True, sharey=True):
    fig, axes = plt.subplots(3, 4, figsize=figsize, sharex=sharex, sharey=sharey, squeeze=False)
    model_order = ["SA1", "SA2", "SA3"]
    scenario_order = ["S0", "S1", "S2", "S3"]
    return fig, axes, model_order, scenario_order


def _combined_model_row_labels_overlay():
    return {
        "SA1": "SA1: AR6 pathways,\nfixed NATO/nonNATO GDP share",
        "SA2": "SA2: AR6 pathways,\nevolving NATO/nonNATO GDP share",
        "SA3": "SA3: Main model with\nPPP recalibration",
    }


def _combined_scenario_column_labels_overlay():
    return {
        "S0": "Baseline",
        "S1": "NATO→3.5%, non-NATO holds",
        "S2": "NATO→3.5%, non-NATO→3.5%",
        "S3": "NATO→5%, non-NATO→3.5%",
    }


def _decorate_3x4_grid(fig, axes, model_order, scenario_order):
    col_labels = _combined_scenario_column_labels_overlay()
    for j, sc in enumerate(scenario_order):
        axes[0, j].set_title(col_labels[sc], fontsize=11, pad=8)


def _combined_pathway_handles_overlay(combined: pd.DataFrame):
    handles = [
        Line2D([0], [0], color="black", linestyle="--", lw=2.1,
               label="Main model: Reference case: baseline=5.5%, g=3%, dₘ=1%"),
    ]
    source = combined[combined["combined_model"] == "SA1"]
    for p in AR6_PATHWAYS:
        z = source[source["projected_pathway"] == p]
        handles.append(Line2D([0], [0], color=PATHWAY_COLORS[p], linestyle="-", lw=2.0, label=p))
    handles.append(Line2D([0], [0], color="#7b2cbf", linestyle="-", lw=2.35,
                          label="SA3 PPP recalibration"))
    return handles


def _plot_overlay_timeseries_cell(ax, combined: pd.DataFrame, model_key: str, scenario_code: str, value_col: str):
    ref = combined[(combined["combined_model"] == "Original") & (combined["scenario_code"] == scenario_code)].sort_values("year")
    if not ref.empty:
        ax.plot(ref["year"], ref[value_col], color="black", linestyle="--", lw=2.1)
    d = combined[(combined["combined_model"] == model_key) & (combined["scenario_code"] == scenario_code)].copy()
    if model_key in {"SA1", "SA2"}:
        for p in AR6_PATHWAYS:
            z = d[d["projected_pathway"] == p].sort_values("year")
            if not z.empty:
                ax.plot(z["year"], z[value_col], color=PATHWAY_COLORS[p], linestyle="-", lw=1.6)
    else:
        z = d.sort_values("year")
        if not z.empty:
            ax.plot(z["year"], z[value_col], color="#7b2cbf", linestyle="-", lw=1.8)
    _style_axis(ax)
    ax.grid(False)
    ax.grid(True, alpha=0.2)
    ax.tick_params(axis="both", labelsize=10)


def _finish_overlay_timeseries(fig, axes, model_order, scenario_order, combined, title, ylabel, xmin, xmax):
    row_labels = _combined_model_row_labels_overlay()
    panel_counter = 0
    for i in range(len(model_order)):
        for j in range(len(scenario_order)):
            axes[i, j].set_xlim(xmin, xmax)
            axes[i, j].set_xticks(np.linspace(xmin, xmax, 3).round().astype(int))
            axes[i, j].text(
                0.02, 0.95, f"({chr(97 + panel_counter)})",
                transform=axes[i, j].transAxes, fontsize=11,
                fontweight="bold", va="top",
            )
            panel_counter += 1
        axes[i, 0].set_ylabel(f"{row_labels[model_order[i]]}\n{ylabel}")
    for ax in axes[-1, :]:
        ax.set_xlabel("Year")
    _decorate_3x4_grid(fig, axes, model_order, scenario_order)
    handles = _combined_pathway_handles_overlay(combined)
    fig.legend(
        handles=handles[1:], loc="lower center", bbox_to_anchor=(0.5, 0.052),
        ncol=6, frameon=False, fontsize=9.2, columnspacing=1.25,
        handlelength=2.4, handletextpad=0.5,
    )
    fig.legend(
        handles=[handles[0]], loc="lower center", bbox_to_anchor=(0.5, 0.018),
        ncol=1, frameon=False, fontsize=9.2, handlelength=2.8,
        handletextpad=0.55,
    )
    fig.suptitle(title, fontsize=14, y=0.985)
    fig.tight_layout(rect=[0, 0.12, 1, 0.95], h_pad=1.25, w_pad=0.9)


def _finish_absolute_overlay_timeseries(
    fig,
    axes,
    model_order,
    scenario_order,
    combined,
    title,
    ylabel,
    xmin,
    xmax,
):
    """Apply the approved main-text layout to the absolute-emissions grid."""
    column_labels = {
        "S0": "S0\nBaseline",
        "S1": "S1\nNATO→3.5%, non-NATO holds",
        "S2": "S2\nNATO→3.5%, non-NATO→3.5%",
        "S3": "S3\nNATO→5%, non-NATO→3.5%",
    }
    row_labels = _combined_model_row_labels_overlay()
    panel_counter = 0

    for i in range(len(model_order)):
        for j in range(len(scenario_order)):
            ax = axes[i, j]
            ax.set_xlim(xmin, xmax)
            ax.set_xticks(np.linspace(xmin, xmax, 3).round().astype(int))
            ax.tick_params(axis="both", labelsize=10, width=0.5)
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
        axes[i, 0].set_ylabel(
            f"{row_labels[model_order[i]]}\n{ylabel}", fontsize=10
        )

    for ax in axes[-1, :]:
        ax.set_xlabel("Year", fontsize=10)
    for j, scenario_code in enumerate(scenario_order):
        axes[0, j].set_title(
            column_labels[scenario_code],
            fontsize=11,
            pad=5,
            linespacing=1.0,
        )

    handles = _combined_pathway_handles_overlay(combined)
    handles[0].set_label(
        "Main model: Reference case: baseline=5.5%, g=3%, $d_m$=1%"
    )
    fig.legend(
        handles=handles[1:],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.092),
        ncol=6,
        frameon=False,
        fontsize=10,
        columnspacing=1.25,
        handlelength=2.4,
        handletextpad=0.5,
        borderaxespad=0.2,
    )
    fig.legend(
        handles=[handles[0]],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.046),
        ncol=1,
        frameon=False,
        fontsize=10,
        handlelength=2.8,
        handletextpad=0.55,
        borderaxespad=0.2,
    )
    fig.suptitle(title, fontsize=14, y=0.98)
    fig.tight_layout(
        rect=[0.035, 0.13, 0.995, 0.955], h_pad=1.2, w_pad=0.8
    )


def make_combined_fig_1(combined: pd.DataFrame, outdir: Path):
    fig, axes, model_order, scenario_order = _combined_3x4_axes(
        figsize=(11, 10.4)
    )
    for i, key in enumerate(model_order):
        for j, sc in enumerate(scenario_order):
            _plot_overlay_timeseries_cell(axes[i, j], combined, key, sc, "E_mil_Gt")
    _finish_absolute_overlay_timeseries(
        fig,
        axes,
        model_order,
        scenario_order,
        combined,
        "Sensitivity analysis of absolute military GHG emissions (2025-2035)",
        "Military GHG emissions (GtCO$_2$e/yr)",
        2025,
        2035,
    )
    fig.savefig(outdir / "Fig_1.pdf", format="pdf", dpi=300)
    fig.savefig(outdir / "Fig_1.png", format="png", dpi=300)
    plt.close(fig)


def make_combined_fig_2(combined: pd.DataFrame, outdir: Path):
    d = combined.copy()
    d["mil_share_pct"] = 100.0 * d["s_mil"]
    fig, axes, model_order, scenario_order = _combined_3x4_axes()
    for i, key in enumerate(model_order):
        for j, sc in enumerate(scenario_order):
            _plot_overlay_timeseries_cell(axes[i, j], d, key, sc, "mil_share_pct")
    _finish_overlay_timeseries(fig, axes, model_order, scenario_order, d,
        "Figure 2. Military share of global GHG emissions (2025–2035); 2025 military footprint = 5.5%",
        "Military share of global GHG (%)", 2025, 2035)
    fig.savefig(outdir / "Fig_2.pdf", bbox_inches="tight")
    fig.savefig(outdir / "Fig_2.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_combined_fig_3(combined: pd.DataFrame, outdir: Path):
    fig, axes, model_order, scenario_order = _combined_3x4_axes()
    for i, key in enumerate(model_order):
        for j, sc in enumerate(scenario_order):
            _plot_overlay_timeseries_cell(axes[i, j], combined, key, sc, "cum_E_mil_Gt")
    _finish_overlay_timeseries(fig, axes, model_order, scenario_order, combined,
        "Figure 3. Cumulative military GHG emissions (2025–2050); 2025 military footprint = 5.5%",
        "Cumulative military emissions (GtCO$_2$e)", 2025, 2050)
    fig.savefig(outdir / "Fig_3.pdf", bbox_inches="tight")
    fig.savefig(outdir / "Fig_3.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_combined_fig_4(combined: pd.DataFrame, outdir: Path):
    fig, axes, model_order, scenario_order = _combined_3x4_axes()
    for i, key in enumerate(model_order):
        for j, sc in enumerate(scenario_order):
            _plot_overlay_timeseries_cell(axes[i, j], combined, key, sc, "cum_incremental_E_mil_vs_S0_Gt")
            axes[i, j].axhline(0, color="black", lw=0.7, alpha=0.45)
    _finish_overlay_timeseries(fig, axes, model_order, scenario_order, combined,
        "Figure 4. Incremental cumulative military emissions relative to S0 (2025–2050); 2025 military footprint = 5.5%",
        r"$\Delta$ cumulative military emissions vs S0 (GtCO$_2$e)", 2025, 2050)
    fig.savefig(outdir / "Fig_4.pdf", bbox_inches="tight")
    fig.savefig(outdir / "Fig_4.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_combined_fig_5(budget: pd.DataFrame, outdir: Path):
    fig, axes, model_order, scenario_order = _combined_3x4_axes(figsize=(18.5, 11.8), sharex=False, sharey=True)
    global_max = 0.0
    for i, key in enumerate(model_order):
        for j, sc in enumerate(scenario_order):
            ax = axes[i, j]
            ref = budget[(budget["combined_model"] == "Original") & (budget["scenario_code"] == sc)].copy()
            d = budget[(budget["combined_model"] == key) & (budget["scenario_code"] == sc)].copy()
            if key in {"SA1", "SA2"}:
                d["xord"] = d["projected_pathway"].map({p:k for k,p in enumerate(AR6_PATHWAYS)})
                d = d.sort_values("xord")
                x = np.arange(len(AR6_PATHWAYS), dtype=float)
                used = d["pct_1p5_budget_used"].to_numpy(float)
                remain = np.clip(100.0 - used, 0.0, None)
                used2 = d["pct_2C_budget_used"].to_numpy(float)
                colors = [PATHWAY_COLORS[p] for p in d["projected_pathway"]]
                ax.bar(x, used, width=0.68, color=colors, alpha=0.88, zorder=2)
                ax.bar(x, remain, width=0.68, bottom=used, color=colors, alpha=0.18, zorder=1)
                ax.scatter(x, used2, marker="D", s=25, color="black", zorder=4)
                if not ref.empty:
                    ru = float(ref.iloc[0]["pct_1p5_budget_used"])
                    rr = max(0.0, 100.0 - ru)
                    r2 = float(ref.iloc[0]["pct_2C_budget_used"])
                    xr = -0.80
                    ax.bar([xr], [ru], width=0.34, color="white", edgecolor="black", hatch='//', zorder=3)
                    ax.bar([xr], [rr], width=0.34, bottom=[ru], color="white", edgecolor="black", alpha=0.30, hatch='//', zorder=2)
                    ax.scatter([xr], [r2], marker='D', s=24, color='black', zorder=5)
                ax.set_xticks(x)
                ax.set_xticklabels(AR6_PATHWAYS, rotation=45, ha="right", fontsize=7.2)
            else:
                x = np.array([0.0, 0.9])
                ref_u = float(ref.iloc[0]["pct_1p5_budget_used"]) if not ref.empty else np.nan
                ref_r = max(0.0, 100.0 - ref_u) if pd.notna(ref_u) else np.nan
                ref_2 = float(ref.iloc[0]["pct_2C_budget_used"]) if not ref.empty else np.nan
                sa3_u = float(d.iloc[0]["pct_1p5_budget_used"]) if not d.empty else np.nan
                sa3_r = max(0.0, 100.0 - sa3_u) if pd.notna(sa3_u) else np.nan
                sa3_2 = float(d.iloc[0]["pct_2C_budget_used"]) if not d.empty else np.nan
                ax.bar([x[0]], [ref_u], width=0.32, color="white", edgecolor="black", hatch='//', zorder=3)
                ax.bar([x[0]], [ref_r], width=0.32, bottom=[ref_u], color="white", edgecolor="black", alpha=0.30, hatch='//', zorder=2)
                ax.scatter([x[0]], [ref_2], marker='D', s=24, color='black', zorder=5)
                ax.bar([x[1]], [sa3_u], width=0.42, color="#6b8e23", alpha=0.88, zorder=3)
                ax.bar([x[1]], [sa3_r], width=0.42, bottom=[sa3_u], color="#6b8e23", alpha=0.18, zorder=2)
                ax.scatter([x[1]], [sa3_2], marker='D', s=25, color='black', zorder=5)
                ax.set_xticks(x)
                ax.set_xticklabels(["Ref.", "SA3"], fontsize=8.5)
            vals = []
            if not d.empty:
                vals += d["pct_1p5_budget_used"].astype(float).tolist()
            if not ref.empty:
                vals += ref["pct_1p5_budget_used"].astype(float).tolist()
            if vals:
                global_max = max(global_max, max(vals))
            ax.axhline(100, color="grey", linestyle="--", lw=0.9)
            _style_axis(ax)
    _decorate_3x4_grid(fig, axes, model_order, scenario_order)
    ymax = max(110.0, global_max * 1.08)
    for i in range(len(model_order)):
        for j in range(len(scenario_order)):
            axes[i, j].set_ylim(0, ymax)
            if j == 0:
                axes[i, j].set_ylabel("Share of remaining carbon budget (%)")
    handles = [
        Patch(facecolor="grey", alpha=0.88, label="Share of 1.5°C budget used"),
        Patch(facecolor="grey", alpha=0.18, label="1.5°C budget remaining"),
        Line2D([0], [0], marker="D", color="black", lw=0, label="Share of 2°C budget used"),
        Patch(facecolor="white", edgecolor="black", hatch='//', label="Reference model reference case (MER)"),
        Patch(facecolor="#6b8e23", alpha=0.88, label="SA3 PPP recalibration"),
    ] + [Patch(facecolor=PATHWAY_COLORS[p], alpha=0.88, label=p) for p in AR6_PATHWAYS]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle(
        f"Figure 5. Military CO$_2$ emissions relative to remaining global carbon budgets, {BUDGET_START_YEAR}–{BUDGET_END_YEAR}; 2025 military footprint = 5.5%",
        fontsize=14, y=0.995,
    )
    fig.tight_layout(rect=[0.145, 0.13, 1, 0.965], h_pad=1.25, w_pad=0.9)
    fig.savefig(outdir / "Fig_5.pdf", bbox_inches="tight")
    fig.savefig(outdir / "Fig_5.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_combined_outputs(reference: pd.DataFrame, sa1: pd.DataFrame, sa2: pd.DataFrame, sa3: pd.DataFrame, root: Path) -> None:
    outdir = root / "Combined"
    outdir.mkdir(parents=True, exist_ok=True)
    combined = _combined_prepare_trajectory_tables(reference, sa1, sa2, sa3)
    combined_budget = _combined_prepare_budget_tables(reference, sa1, sa2, sa3)
    combined[combined["year"] <= 2035].to_csv(outdir / "combined_actual_projected_scenarios_2025-2035.csv", index=False)
    combined.to_csv(outdir / "combined_actual_projected_scenarios_2025-2050.csv", index=False)
    combined_budget.to_csv(outdir / "combined_actual_projected_budget_results_2026-2050.csv", index=False)
    label_cols = ["combined_model", "combined_model_label", "projected_pathway", "ar6_model", "ar6_scenario", "scenario_code", "scenario", "trajectory_label"]
    label_cols = [c for c in label_cols if c in combined.columns]
    combined[label_cols].drop_duplicates().to_csv(outdir / "combined_trajectory_labels.csv", index=False)
    make_combined_fig_1(combined, outdir)
    make_combined_fig_2(combined, outdir)
    make_combined_fig_3(combined, outdir)
    make_combined_fig_4(combined, outdir)
    make_combined_fig_5(combined_budget, outdir)
    print(f"Saved Combined actual-scenario CSVs and Fig_1 ... Fig_5 to {outdir}")


if __name__=="__main__":
    main_three_sensitivities()
