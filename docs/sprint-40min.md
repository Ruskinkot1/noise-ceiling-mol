# 40-minute sprint, 5 participants

Project: **noise-ceiling-mol** (the noise ceiling in molecular property benchmarks).

**Goal:** in a single session, collect the inputs for stages 0–1: novelty check and data selection. Everyone works independently, with a joint summary at the end.

## Timing

| Time | Activity |
|---|---|
| 0–3 min | Task assignment |
| 3–33 min | Independent work |
| 33–40 min | Summary, 1.5 min per person |

## Tasks

| # | Participant | Task | Output |
|---|---|---|---|
| 1 | Novelty A | Find 2022–2026 work on noise ceiling, inter-assay variability, label noise in ChEMBL and TDC | 5–8 papers: title, year, gist, difference from our idea, link |
| 2 | Novelty B | Find work comparing pretrained molecular embeddings with ECFP, and work on activity cliffs | 5–8 papers in the same format and a conclusion: is our formulation unclaimed |
| 3 | ChEMBL data | Write a query or script that finds molecules with repeated measurements of one endpoint from different documents. Run it on 3–5 targets | The query and a "target → number of molecules with ≥2 sources" table |
| 4 | Models | Compile a list of open pretrained embeddings that run locally (ChemBERTa, MolFormer, Uni-Mol, GROVER, etc.). Check license, size, inference method | Table: model, license, size, runs on a laptop or not, link |
| 5 | Metric and splits | Describe the "fraction of ceiling" formula and the evaluation protocol: how to compute the ceiling from replicates, how to build scaffold and temporal splits, how many seeds | A page with formulas and a step-by-step protocol |

## Rules

- Everyone cites a source and marks anything unverified as "not verified".
- Results go into the shared document, in your own section, no later than minute 33.
- If a task is finished early, take the spare one: find out whether ChEMBL contains documented measurement-error data (standard errors, comment fields).

## Session outcome (last 7 minutes)

1. Novelty decision: the idea is unclaimed, partly taken, or taken.
2. Number of available targets with replicates: is the minimum of 15 met.
3. List of models for stage 3.
4. Task assignment for next week.
