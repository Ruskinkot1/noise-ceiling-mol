---
name: ncm-decontaminate
description: Remove molecules and endpoints already present in public benchmarks or model pretraining corpora so the dataset is unique. Use after collection and before splitting.
---

# Decontamination

1. Build a registry of seen InChIKeys from MoleculeNet, TDC, Polaris, ADMET sets, and published pretraining corpora of the models being compared. Record each source and date.
2. Drop exact matches (full InChIKey and first 14 characters).
3. Drop near analogs: Tanimoto (ECFP4) above 0.8 to any registry molecule. Pick the threshold on the pilot and write down why.
4. Keep only documents published after each model's training cutoff. If a cutoff is not documented, mark "не проверено" and use the latest possible date.
5. Log counts removed at each step and, per model, the share of test molecules overlapping its training set.
6. Also keep the uncleaned set to measure leakage as the difference in scores.

Pass the registry to the collector with `--exclude`.
