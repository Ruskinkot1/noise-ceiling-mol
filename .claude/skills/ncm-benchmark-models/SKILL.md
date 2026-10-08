---
name: ncm-benchmark-models
description: Run baselines and open pretrained embeddings on the splits and score them. Use for the modelling stage.
---

# Benchmark models

1. Splits: random, scaffold (Murcko), temporal (publication year). Check leakage by InChIKey and scaffold.
2. Baselines: ECFP + GBM/RF, ECFP + MLP, D-MPNN.
3. Pretrained embeddings (inference only, open license, runs locally): pick 3–4, for example ChemBERTa, MolFormer, Uni-Mol, GROVER. Check license, size and ability to run on a laptop first.
4. 5 seeds per model/split/endpoint, report mean and confidence intervals.
5. Ask before installing anything over 1 GB; the laptop disk is nearly full. Clean up afterwards.

Output: results table in `results/` plus the exact command and versions used.
