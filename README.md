# Modelling the impact of military spending escalation on global greenhouse gas emissions

This repository contains the code needed to reproduce the paper **“Modelling the impact of military spending escalation on global greenhouse gas emissions”**. The paper develops a reduced-form macro-emissions model linking military expenditure (as a share of GDP), global GDP growth, sector-specific decarbonisation rates, and an elasticity between military burden and broader emissions. It simulates four military-spending pathways: baseline (S0), NATO-only escalation to 3.5% of GDP (S1), coordinated escalation to 3.5% in NATO and non-NATO blocs (S2), and NATO to 5% with non-NATO to 3.5% (S3).

## Repository contents

- `military_emissions_model.py` — runs the full simulation grid and writes:
  - `generated_data_2025-2035.csv`
  - `generated_data_2025-2050.csv`
  - `generated_data_budgets_2050.csv`
- `plot_absolute_fig_1_and_fig_s1.py` — Fig. 1 and Fig. S1
- `plot_share_fig_2_and_fig_s2.py` — Fig. 2 and Fig. S2
- `plot_cumulative_fig_3_and_fig_s3.py` — Fig. 3 and Fig. S3
- `plot_efficiency_fig_4.py` — Fig. 4
- `plot_budget_fig_5_and_fig_s5.py` — Fig. 5 and Fig. S5
- `plot_pathways_fig_6.py` — Fig. 6

## How to run

Create the simulation outputs first:

```bash
python military_emissions_model.py
```

Then run the figure scripts as needed:

```bash
python plot_absolute_fig_1_and_fig_s1.py
python plot_share_fig_2_and_fig_s2.py
python plot_cumulative_fig_3_and_fig_s3.py
python plot_efficiency_fig_4.py
python plot_budget_fig_5_and_fig_s5.py
python plot_pathways_fig_6.py
```

## External data needed for Fig. 6

`plot_pathways_fig_6.py` also requires the file:

`1668008312256-AR6_Scenarios_Database_World_v1.1.csv.zip`

This file should be downloaded from the **AR6 Scenario Explorer and Database** hosted by IIASA:

https://data.ene.iiasa.ac.at/ar6/

Background and documentation page:

https://data.ene.iiasa.ac.at/ar6/static/About.html

Place the downloaded zip file in the same directory as `plot_pathways_fig_6.py`.

## Notes

- The plotting scripts expect the generated CSV files to be in the same folder.
- Fig. 6 uses selected AR6 SSP reference pathways and AR6 climate-category metadata from the IIASA AR6 database.

