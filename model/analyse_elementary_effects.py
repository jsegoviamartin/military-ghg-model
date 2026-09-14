"""Generate the elementary-effects and scenario-contrast tables."""

from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[1]
data = pd.read_csv(root / "data/generated/generated_data_2025-2050.csv")
out = root / "output/elementary_effects"

design = ["scenario_code", "s_mil_2025_assumed", "g_world", "d_mil", "d_rest", "epsilon"]
factors = {
    "Baseline military-emissions share": ("s_mil_2025_assumed", False),
    "GDP growth": ("g_world", False),
    "Military decarbonisation, d_m": ("d_mil", False),
    "Rest-of-economy decarbonisation, d_r": ("d_rest", False),
    "Spillover elasticity": ("epsilon", False),
}

annual = data.query("year == 2035")
cumulative = data.groupby(design, as_index=False).E_mil_Gt.sum()
outcomes = {
    "Annual military emissions, 2035": (annual, "E_mil_Gt", 1, "GtCO2e/yr"),
    "Military share, 2035": (annual, "s_mil", 100, "percentage points"),
    "Cumulative military emissions, 2025-2050": (cumulative, "E_mil_Gt", 1, "GtCO2e"),
}


def difference(frame, output, factor, a, b, multiplier=1, per_pp=False):
    """Change one input while all other inputs remain exactly matched."""
    keys = [x for x in design if x != factor]
    left = frame[frame[factor].eq(a)][keys + [output]]
    right = frame[frame[factor].eq(b)][keys + [output]]
    pairs = left.merge(right, on=keys, suffixes=("_a", "_b"))
    step = abs(b-a)*100 if per_pp else 1
    return multiplier * (pairs[f"{output}_b"] - pairs[f"{output}_a"]) / step


# Table 1: signed mean elementary effect (mu) and standard deviation (sigma).
table = pd.DataFrame(index=factors, columns=outcomes)
for label, (factor, reverse) in factors.items():
    for outcome, (frame, output, multiplier, _) in outcomes.items():
        levels = sorted(frame[factor].unique(), reverse=reverse)
        effects = pd.concat([difference(frame, output, factor, a, b, multiplier, True)
                             for a, b in zip(levels[:-1], levels[1:])])
        table.loc[label, outcome] = f"{effects.mean():.3f} ({effects.std():.3f})"

# Table 2: categorical scenarios compared with their exactly matched S0 cases.
rows = []
for outcome, (frame, output, multiplier, _) in outcomes.items():
    for scenario in ("S1", "S2", "S3"):
        effects = difference(frame, output, "scenario_code", "S0", scenario, multiplier)
        rows.append([f"{scenario} vs S0", outcome, f"{effects.mean():.3f} ({effects.std():.3f})"])
scenarios = pd.DataFrame(rows, columns=["contrast", "outcome", "mean_sigma"])
scenarios = scenarios.pivot(index="contrast", columns="outcome", values="mean_sigma").loc[
    ["S1 vs S0", "S2 vs S0", "S3 vs S0"], list(outcomes)]

out.mkdir(parents=True, exist_ok=True)
table.to_csv(out / "elementary_effects_compact_table.csv", index_label="input")
scenarios.to_csv(out / "scenario_contrasts.csv", index_label="contrast")
print(table, "\n\n", scenarios, sep="")
