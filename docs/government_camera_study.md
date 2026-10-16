# Temporal Validation of Government-Informed Camera Allocation

## Research Question

Given a fixed monitoring-camera inventory and a limited activation budget, can
historical crash demand improve the coverage of subsequent crash locations?
This question links public-sector safety priorities to edge resource allocation.

## Data and Temporal Design

The source archive contains 393,854 deduplicated records from Taiwan's 2024
A1/A2 injury-crash release. New Taipei contributes 46,957 dated A1/A2 records.
Camera selection uses January--September (34,108 records); evaluation uses
October--December (12,849 records). Coordinate validation retains 34,106 and
12,846 records, respectively. The 17-site public-camera inventory remains
fixed across the experiment.

Crash locations are grouped at three decimal degrees. Grid centers are fixed
between partitions. Training and evaluation demand are aggregated independently;
camera selection receives training weights only. The fixed inventory supports
a retrospective allocation experiment without assumptions about installation
dates.

Sources: [National Police Agency crash data](https://data.gov.tw/dataset/172969)
and [New Taipei camera locations](https://data.ntpc.gov.tw/datasets/bb59a616-3572-4e92-9d09-01bf422057a6).

## Objective and Baselines

For camera subset S, maximize the sum of historical crash weights in grid cells
within radius r of at least one selected site, subject to |S| <= k. A1:A2
weights are evaluated at 1:1, 5:1, and 10:1; radii are 500 m, 1 km, and 2 km.
Camera budgets are 1, 3, 5, 10, and 17.

The objective is nonnegative, monotone, and submodular. Selecting the camera
with the largest uncovered marginal weight follows the greedy approximation
studied by [Nemhauser et al.](https://doi.org/10.1007/BF01588971). The comparison
includes crash-count greedy selection, independent risk ranking, and random
selection over seeds 0--199. Each strategy uses the same budget and inventory.

## Results at the Reference Configuration

Reference setting: A1:A2 = 5:1, radius = 1 km.

| Camera budget | Risk greedy | Count greedy | Independent rank | Random mean |
| --- | ---: | ---: | ---: | ---: |
| 3 | 7.31% | 7.31% | 7.31% | 3.79% |
| 5 | 10.69% | 10.69% | 10.69% | 6.31% |
| 10 | 15.49% | 15.49% | 13.69% | 11.38% |
| 17 | 16.39% | 16.39% | 16.39% | 16.39% |

At a ten-camera budget, marginal-gain selection improves held-out coverage by
1.80 percentage points over independent ranking and 4.12 points over random
selection. Risk and count greedy select equally effective sets at this setting;
the marginal-gain objective captures nonredundant coverage. The complete 180-row table
preserves all weight/radius/budget settings.

![Camera budget and held-out risk coverage](assets/government_camera_budget.png)

A paired bootstrap resamples the 29 New Taipei districts (2,000 repetitions,
seed 42). The greedy-minus-independent difference is 1.80 percentage points,
with a 95% percentile interval of -1.01 to 5.64 points. This interval captures
heterogeneity across districts; random-baseline error bars describe one standard
deviation across camera-selection seeds. The two uncertainty measures address
different sources of variation.

[Uncertainty estimates](results/research_uncertainty.json) ·
[Vector figure](assets/government_camera_budget.pdf)

## Interpretation

The endpoint measures geographic proximity coverage of held-out crash demand.
Camera orientation, occlusion, road-network reachability, and actual event
visibility belong to the subsequent site-level assessment. The result
quantifies allocation efficiency and supports prioritizing site audits under a
limited deployment budget. Longitudinal intervention evaluation can then follow
the exposure-aware road-safety principles discussed by
[Persaud and Lyon](https://doi.org/10.1016/j.aap.2006.09.009).

## Reproduction

```powershell
python scripts/evaluate_government_camera_budget.py
python -m pytest tests/test_risk_coverage.py
```

- [Protocol and source hashes](results/government_camera_budget/protocol.json)
- [Complete comparison](results/government_camera_budget/comparison.csv)
- [Selected camera sets](results/government_camera_budget/selections.json)
