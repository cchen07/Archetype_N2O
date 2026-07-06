# Repository for manuscript: Archetypal Microbiome Profiles as Indicators of Nitrous Oxide Emission States in Activated Sludge

Preprint on [arXiv](https://arxiv.org/abs/2606.18295)

This repository contains an archetypal analysis workflow for N2O-related microbial genus abundance datasets. The main notebook loads the 2 datasets, fits three archetypes for each dataset, and displays four diagnostic plots per dataset.

## Repository Layout

```text
.
|-- Code/
|   |-- plot.ipynb
|   `-- plot_functions.py
`-- Data/
    |-- X_genus_W.csv
    |-- X_genus_U.csv
    |-- Emission_W.csv
    `-- Emission_U.csv
```

## Data

- `Data/X_genus_W.csv`: Werdhölzli (W) genus abundance matrix, with 37 samples and 2035 genus features.
- `Data/X_genus_U.csv`: Uster (U) genus abundance matrix, with 46 samples and 556 genus features.
- `Data/Emission_W.csv`: Werdhölzli emission metadata with `EF`, `Temp (°C)` and `class` columns.
- `Data/Emission_U.csv`: Uster emission metadata with `Emission`, `Temp (°C)`, and `class` columns.

The W samples are indexed by date. The U samples are indexed by reactor/date labels such as `SBR1_2018-11-09`.

## Main Workflow

Open and run:

```text
Code/plot.ipynb
```

The notebook generates eight plots in total:

1. W archetype profile heatmap
2. W simplex plot
3. W stacked archetype composition over time with EF overlay
4. W archetype coefficients vs temperature
5. U archetype profile heatmap
6. U simplex plot by reactor
7. U stacked archetype composition by reactor with emission overlay
8. U archetype coefficients vs temperature

Each plot is in its own notebook cell so the outputs can be viewed separately.

## Plotting Functions

The reusable plotting and fitting code is in `Code/plot_functions.py`.

Key functions:

- `plot_archetypes`: fit an archetypal analysis model.
- `plot_archetype_profiles_heatmap`: fit archetypes and plot top genus profiles as a heatmap.
- `plot_simplex`: show samples in a three-archetype simplex.
- `plot_simplex_by_reactors`: show samples in a simplex with reactor-specific markers.
- `plot_stack_bar`: plot stacked archetype coefficients over time.
- `plot_stack_bar_by_reactors`: plot stacked coefficients separately by reactor.
- `plot_coefficients_vs_continuous_panels`: compare archetype coefficients with a continuous emission variable.

## Notes

- The default number of archetypes in the notebook is `3`.
- The heatmaps show the top `20` genera by mean archetype abundance.
