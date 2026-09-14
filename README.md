# Military GHG Model

Reproducible Python implementation of a deterministic, scenario-based
macro-emissions model linking military expenditure, real GDP growth,
military and rest-of-economy emissions-intensity reductions, and a reduced-form
spillover elasticity. The repository contains the model code, input datasets,
country mapping, generated results, sensitivity analyses, and publication
figures. The manuscript and peer-review working files are not included.

## Model scenarios

Military burden is military expenditure as a share of GDP.
- S0 — Benchmark: both blocs remain at their 2025 values.
- S1 — NATO-only escalation: NATO increases to 3.5%; non-NATO remains at its 2025 value.
- S2 — Coordinated escalation: both blocs increase to 3.5%.
- S3 — Further escalation: NATO increases to 5.0%; non-NATO increases to 3.5%.

The complete factorial grid varies:

- baseline military share of global GHG emissions: 3.3%, 5.5%, and 7.0%;
- annual real GDP growth: 1%, 2%, 3%, and 4%;
- annual military emissions-intensity reduction: 0% to 7%;
- annual rest-of-economy emissions-intensity reduction: 1% to 7%; and
- spillover elasticity: 0%, 0.9%, 1.5%, and 2.0% per percentage-point increase
  in military burden.

The reference case uses a 5.5% baseline military-emissions share, 3% annual
real GDP growth, 1% annual military and rest-of-economy emissions-intensity
reductions, and zero spillover elasticity. These are predefined scenarios and
parameter combinations, not probability distributions.

## Structural sensitivity analyses

- **SA1:** five selected IPCC AR6 pathways with fixed NATO/non-NATO GDP shares.
- **SA2:** the same AR6 pathways with evolving bloc GDP shares derived from
  country-level SSP GDP projections.
- **SA3:** the main model recalibrated using purchasing-power-parity GDP.

Carbon-budget outputs use the [Forster et al. (2026)](https://doi.org/10.5194/essd-18-3889-2026)
50% budgets referenced to the beginning of 2026: 130 GtCO2 for 1.5 C and
1050 GtCO2 for 2 C. Military GHG emissions are converted using a CO2 fraction
of 0.74 before comparison.

## Repository structure

```text
model/                  Model, sensitivity, plotting, and elementary-effects code
data/inputs/            AR6, SSP Basic Drivers, and IMF PPP input files
data/mappings/          NATO membership mapping documenting bloc aggregation
data/generated/         Complete main-model factorial outputs
output/                 Figures, sensitivity outputs, audits, and summary tables
requirements.txt        Tested Python dependencies
```

## Input data

The supplied inputs are tracked with Git LFS because two source files are
large. Clone with Git LFS enabled (`git lfs install`) to retrieve them.

| File | Source | Use |
|---|---|---|
| `1668008312256-AR6_Scenarios_Database_World_v1.1.csv.zip` | [IPCC AR6 Scenario Explorer release 1.1](https://doi.org/10.5281/zenodo.5886911) | World GDP and GHG pathways for SA1 and SA2 |
| `ssp_basic_drivers_release_3.2_full.xlsx` | [SSP Basic Drivers v3.2](https://ssp.apps.ece.iiasa.ac.at/documentation/basic-drivers) | Country-level GDP paths used to construct evolving bloc shares in SA2 |
| `IMF_WEO_PPP_2025.csv` | [IMF World Economic Outlook database](https://data.imf.org/en/Data-Explorer?datasetUrn=IMF.RES:WEO) | 2025 PPP world GDP and NATO/non-NATO recalibration for SA3 |
| `nato_members_2025.csv` | NATO membership in the 2025 model baseline | Explicit 32-member ISO3 mapping; all other countries form the non-NATO residual |

SHA-256 checksums for the three external input files:

```text
EC09AEE07A6AE4C8C175D933A2365A0FE7E4A5F8A6EBF9AA49875EC4063F5B6B  1668008312256-AR6_Scenarios_Database_World_v1.1.csv.zip
2D327D71895E4685D750633AAAEF2B7F088D30159975129C94C698ED9F5B829B  ssp_basic_drivers_release_3.2_full.xlsx
180801CA04C909B4993577C01F03DEF6F4DF6214EDD2BD4CBFBBF004C8FDAD80  IMF_WEO_PPP_2025.csv
```

Users of the source datasets should also consult and comply with the terms and
metadata provided by their respective publishers.

## Installation

Python 3.12 was used for release verification.

```powershell
git lfs install
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Reproduce the analysis

Run the commands from the repository root in this order:

```powershell
# Main factorial datasets
python model\military_emissions_model.py

# Main and supplementary figures
python model\plot_absolute_military_emissions.py
python model\plot_military_share_global_emissions.py
python model\plot_cumulative_military_emissions.py
python model\plot_global_ghg_emissions_pathways.py

# Structural sensitivity analyses and their combined outputs
python model\military_emissions_model_SA1_SA2_SA3.py

# Elementary effects and matched escalation-scenario contrasts
python model\analyse_elementary_effects.py
```

The plotting scripts regenerate Figure 1 through Figure 8, Figure S1, and
Figure S3. The sensitivity script writes SA-specific results, input audits, and
combined figures under `output/three_sensitivity_analyses/`.

## Main outputs

- `data/generated/generated_data_2025-2035.csv`: complete annual grid through
  2035.
- `data/generated/generated_data_2025-2050.csv`: complete annual grid through
  2050.
- `data/generated/generated_data_budgets_2050.csv`: military CO2 budget
  comparisons over 2026-2050.
- `output/elementary_effects/`: elementary-effects summary and matched scenario
  contrasts.
- `output/three_sensitivity_analyses/`: SA1-SA3 trajectories, endpoint and
  budget summaries, calibration/source audits, and combined outputs.

## Licence

The model code is released under the repository's [MIT License](LICENSE).
External datasets remain subject to the terms of their original publishers.
