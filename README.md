# Modelling the impact of military spending escalation on global greenhouse gas emissions

This repository accompanies the paper **“Modelling the impact of military spending escalation on global greenhouse gas emissions”**. The paper develops a reduced-form macro-emissions model linking military expenditure (as a share of GDP), global GDP growth, sector-specific decarbonisation rates, and an elasticity between military burden and broader emissions. It simulates four military-spending pathways—baseline (S0), NATO-only escalation to 3.5% of GDP (S1), coordinated escalation to 3.5% in NATO and non-NATO blocs (S2), and NATO to 5% with non-NATO to 3.5% (S3)—under a wide sensitivity grid. In the paper’s central case, coordinated escalation substantially increases annual and cumulative military emissions, with the strongest escalation scenario (S3) capable of pushing cumulative military emissions to more than the remaining 1.5°C carbon budget by 2050. fileciteturn2file1L1-L18

## Repository contents

- `military_emissions_model.py` — core simulation script. It runs the full parameter grid and writes three CSV files: `generated_data_2025-2035.csv`, `generated_data_2025-2050.csv`, and `data_budgets_2050.csv`. fileciteturn3file19L1-L10 fileciteturn3file4L19-L39
- `plot_absolute_fig_1_and_fig_s1.py` — generates **Fig_1** and **Fig_S1** from `generated_data_2025-2035.csv`. These correspond to absolute military GHG emissions in 2025–2035. fileciteturn1file5L1-L8 fileciteturn3file17L57-L83
- `plot_share_fig_2_and_fig_s2.py` — generates **Fig_2** and **Fig_S2** from `generated_data_2025-2035.csv`. These show the military share of global GHG emissions in 2025–2035. fileciteturn1file11L1-L8 fileciteturn3file5L15-L40
- `plot_cumulative_fig_3_and_fig_s3.py` — generates **Fig_3** and **Fig_S3** from `generated_data_2025-2050.csv`. These show cumulative military emissions to 2050. fileciteturn3file3L1-L8
- `plot_efficiency_fig_4.py` — generates **Fig_4**, a break-even plot asking whether military-side decarbonisation can offset burden escalation. It saves `Fig_4.pdf` and `Fig_4.png`. fileciteturn3file12L45-L88
- `plot_budget_fig_5_and_fig_s5.py` — generates **Fig_5** and **Fig_S5** from `generated_data_2025-2050.csv`. These show carbon-budget depletion by 2050. fileciteturn1file1L1-L6 fileciteturn3file7L13-L39
- `plot_pathways_fig_6.py` — generates **Fig_6** from `generated_data_2025-2050.csv` and the AR6 scenario database file described below. It overlays selected model pathways on AR6 SSP reference pathways and climate-category bands. fileciteturn3file2L1-L40 fileciteturn3file8L1-L39
- `GHG_militaries_lancet_JSM.pdf` — manuscript version of the paper. fileciteturn2file1L1-L18

## Requirements

The scripts use `numpy`, `pandas`, and `matplotlib`. `plot_pathways_fig_6.py` also reads an Excel file from inside the AR6 archive via `pandas`, so having `openpyxl` installed is recommended.

A safe setup is:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
pip install numpy pandas matplotlib openpyxl
```

**Python 3.10+ is recommended.** Several scripts use modern type-hint syntax such as `list[float] | None`, which is not compatible with older Python 3.8 interpreters. fileciteturn1file5L9-L16 fileciteturn1file11L9-L16

## How to run the model

Run the core simulation script first:

```bash
python military_emissions_model.py
```

This creates three data files in the current directory:

- `generated_data_2025-2035.csv`
- `generated_data_2025-2050.csv`
- `data_budgets_2050.csv` fileciteturn3file19L1-L10 fileciteturn3file4L19-L39

## How to generate the figures

After generating the CSV files above, run the plotting scripts individually:

```bash
python plot_absolute_fig_1_and_fig_s1.py
python plot_share_fig_2_and_fig_s2.py
python plot_cumulative_fig_3_and_fig_s3.py
python plot_efficiency_fig_4.py
python plot_budget_fig_5_and_fig_s5.py
python plot_pathways_fig_6.py
```

Expected figure outputs:

- `plot_absolute_fig_1_and_fig_s1.py` → `Fig_1.pdf`, `Fig_1.png`, `Fig_S1.pdf`, `Fig_S1.png` fileciteturn1file5L1-L8
- `plot_share_fig_2_and_fig_s2.py` → `Fig_2.pdf`, `Fig_2.png`, `Fig_S2.pdf`, `Fig_S2.png` fileciteturn3file5L15-L40
- `plot_cumulative_fig_3_and_fig_s3.py` → `Fig_3.pdf`, `Fig_3.png`, `Fig_S3.pdf`, `Fig_S3.png` fileciteturn3file3L1-L8
- `plot_efficiency_fig_4.py` → `Fig_4.pdf`, `Fig_4.png` fileciteturn3file12L45-L88
- `plot_budget_fig_5_and_fig_s5.py` → `Fig_5.pdf`, `Fig_5.png`, `Fig_S5.pdf`, `Fig_S5.png` fileciteturn3file7L13-L39
- `plot_pathways_fig_6.py` → `Fig_6.pdf`, `Fig_6.png` fileciteturn1file18L42-L62

## Important note for `plot_efficiency_fig_4.py`

`military_emissions_model.py` writes the 2050 summary file as `data_budgets_2050.csv`, but `plot_efficiency_fig_4.py` currently looks for `generated_data_budgets_2050.csv`. fileciteturn3file4L25-L31 fileciteturn3file6L1-L14

So before running `plot_efficiency_fig_4.py`, do **one** of the following:

1. Rename the model output:

```bash
mv data_budgets_2050.csv generated_data_budgets_2050.csv
```

2. Or edit this line inside `plot_efficiency_fig_4.py`:

```python
csv_file = "generated_data_budgets_2050.csv"
```

and change it to:

```python
csv_file = "data_budgets_2050.csv"
```

## Additional data needed for Fig. 6

`plot_pathways_fig_6.py` requires the file:

```text
1668008312256-AR6_Scenarios_Database_World_v1.1.csv.zip
```

This file is **not** generated by the model and should be downloaded separately from the **AR6 Scenario Explorer and Scenarios Database hosted by IIASA**:

- Main site: https://data.ene.iiasa.ac.at/ar6/
- About page: https://data.ene.iiasa.ac.at/ar6/static/About.html

IIASA’s AR6 site states that the dataset is split into multiple downloadable files, including **“Scenario data for the global region”**, and provides the AR6 database citation and download guidance. citeturn717937view1

After downloading, place the ZIP file in the same directory as `plot_pathways_fig_6.py`. If the downloaded filename differs from the exact name expected by the script, rename it to:

```text
1668008312256-AR6_Scenarios_Database_World_v1.1.csv.zip
```

## Central model settings used throughout the figure scripts

Several figure scripts highlight a central case defined by:

- baseline military share in 2025 = **5.5%**
- global GDP growth = **3%/yr**
- military decarbonisation = **−1%/yr**
- rest-of-economy decarbonisation = **−1%/yr**
- elasticity = **1.5%** per +1 percentage-point increase in military burden. fileciteturn2file1L1-L18 fileciteturn2file1L68-L145

## Citation

If you use this code, please cite the associated paper and, for Fig. 6, also cite the AR6 Scenario Explorer / AR6 Scenarios Database hosted by IIASA as requested on the IIASA site. fileciteturn2file1L1-L18 citeturn717937view1
