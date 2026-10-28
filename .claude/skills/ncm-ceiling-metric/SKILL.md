---
name: ncm-ceiling-metric
description: Estimate the noise ceiling per endpoint from replicate measurements and convert model scores into fraction of ceiling. Use for the analysis stage.
---

# Noise ceiling and fraction-of-ceiling metric

1. For each endpoint estimate inter-document variance of the same molecule from the replicate table.
2. Derive the best achievable RMSE and R² under that noise; bootstrap over molecules for confidence intervals.
3. Fraction of ceiling = (baseline error - model error) / (baseline error - ceiling error), where baseline is the mean predictor. State the exact formula used in the paper. Auxiliary (not headline): RMSE_floor/RMSE and R2/R2_max; see docs/experiment_design.md.
4. Test robustness: filtering thresholds, Ki vs IC50, year ranges.
5. Compare models by fraction of ceiling with paired bootstrap; report where differences are inside the noise.

Do not claim a result before it is computed on data.
