# Models (W4)

Checked 2026-10-09. Anything not confirmed by a model card/code is marked "not verified".

## Table

| Model (key in scripts/06) | Repository @ revision | License | Weights size | Training cutoff date | How it is run | Check result |
|---|---|---|---|---|---|---|
| ECFP4 (counts, 2048) + GBM/RF/MLP/Ridge | rdkit + sklearn | BSD (rdkit, sklearn) | — | — | `--models ecfp_gbm ecfp_rf ecfp_mlp` | works (pipeline tests) |
| D-MPNN (`gnn_dmpnn`) | in-house code `src/ncmol/gnn.py`, pure torch | project code | trained from scratch | — | `--models gnn_dmpnn` | pytest green; 200 synthetic molecules run through scripts/04 (3 splits); see below |
| ChemBERTa-77M-MLM (`chemberta-77m-mlm`) | DeepChem/ChemBERTa-77M-MLM @ ed8a5374 | not stated in the HF card/metadata: not verified | 13.7 MB (pytorch_model.bin) | not verified | `06_embed_foundation.py --models chemberta-77m-mlm` -> `emb-chemberta-77m-mlm_gbm` | inference on 20 mol. OK, dim 384, no NaN, 1 invalid SMILES handled (zero vector, `valid=False`) |
| ChemBERTa-77M-MTR (`chemberta-77m-mtr`) | DeepChem/ChemBERTa-77M-MTR @ 66b895cb | not verified (as above) | 14.0 MB | not verified | likewise | inference on 20 mol. OK, dim 384 |
| ChemBERTa-5M-MLM (`chemberta-5m-mlm`) | DeepChem/ChemBERTa-5M-MLM @ 695672a6 | not verified | 13.7 MB | not verified | likewise | inference on 20 and on 200 mol. OK, dim 384 |
| MoLFormer-XL-both-10pct (`molformer-xl-10pct`) | ibm-research/MoLFormer-XL-both-10pct @ 361063d0 | Apache-2.0 (HF metadata and card) | 187 MB (safetensors) | card: 10% ZINC + 10% PubChem, cutoff date not stated: not verified | likewise; `trust_remote_code=True` | code `modeling_molformer.py` (873 lines) and `configuration_molformer.py` read/skimmed with grep: only torch/transformers, no network/files/exec/pickle; standard tokenizer (tokenizer.json). Inference on 20 mol. OK, dim 768 |
| roberta_zinc_480m (`roberta-zinc-480m`) | entropy/roberta_zinc_480m @ 05b00ec3 | MIT (line "License: MIT" in the card; no license field in the HF metadata) | 409 MB (only model.safetensors is downloaded, without the duplicate .bin) | ZINC, date not stated: not verified | likewise | inference on 20 mol. OK, dim 768 |

Not included: Uni-Mol, GROVER (heavy/3D, per the assignment), ChemGPT-1.2B (4.9 GB, >500 MB), seyonec/ChemBERTa-zinc-base-v1 (179 MB, license not stated; duplicates the ChemBERTa family).
Maximum single download: 409 MB (<500 MB). Total HF cache after all checks ~610 MB in `~/.cache/huggingface` (does not go into the repository; delete if disk space runs low).

## How to run
```
pip install -e .            # torch, transformers, rdkit, sklearn are already installed
python scripts/06_embed_foundation.py                      # all models -> data/embeddings/<key>.npz
python scripts/06_embed_foundation.py --models molformer-xl-10pct --smoke 20   # check on 20 molecules
python scripts/04_run_benchmark.py --models ecfp_gbm gnn_dmpnn emb-chemberta-77m-mlm_gbm emb-molformer-xl-10pct_gbm
```
`npz`: `mol_ids`, `X` (n x dim), `valid` (mask of parsed SMILES), `revision`, `repo`. Cache: if the file exists, covers all
mol_id and the revision matches, the model is skipped (`--force` recomputes). Embeddings = mean-pooling of `last_hidden_state`
with the attention mask over canonicalized (RDKit) SMILES, batches by length, truncation at 256 tokens, CPU, 4 threads, models frozen.
The head suffix in the name (`_gbm`, `_rf`, `_mlp`, `_ridge`) selects the sklearn head on top of the embedding.

## GNN interface
`featurize("gnn_dmpnn", smiles, ids)` returns an object array of SMILES (not a matrix), `make_model("gnn_dmpnn", seed)` returns
`DMPNNRegressor` with `fit(X_smiles, y)` / `predict`. scripts/04_run_benchmark.py did not need changes: `X[tr]` works the same way.
D-MPNN: directed edges, depth 3, hidden 200, dropout 0.1, mean-pooling over atoms, FFN; Adam lr 1e-3, batch 64, up to 60 epochs;
10% of train is used as internal validation, early stopping (patience 10) restoring the best weights; the target is standardized on train.
Determinism: `torch.manual_seed(seed)` + `np.random.default_rng(seed)` (test: two trainings with the same seed give identical
predictions; CPU, 1 thread by default). Hyperparameters were not tuned: per `experiment_design.md` (item 2) the tuning budget for the GNN and
the ECFP models must be the same; this remains a task before the final run.

## What is verified and what is not
- Verified: tests `tests/test_models_gnn.py` without network (graph, reverse edges, invalid SMILES, determinism, learnability on 200
  synthetic molecules: correlation > 0.5, early stopping, masked mean-pooling on a tiny random RoBERTa,
  embedding cache/featurize); `pytest -q` — 36 passed.
- Verified: real inference of all 5 models on 20 molecules (with 1 deliberately invalid SMILES); 200 synthetic molecules run
  through scripts/06 (ChemBERTa-5M) and scripts/04 (ecfp_gbm, gnn_dmpnn, embedding+ridge, 3 splits, 1 seed). Synthetic labels
  say nothing about model quality.
- D-MPNN time: 2000 molecules, 10 epochs ≈ 14 s at a machine load of ~15 (other processes on 4 cores); 60 epochs on 8k molecules estimated
  at several minutes per model/seed/split. 5 seeds x 3 splits x N tasks x GNN is the main time cost of the benchmark. Not measured on a clean machine.
- Not verified: licenses of DeepChem/ChemBERTa-*; cutoff dates of all pretraining corpora; overlap of the pretraining corpora
  (ZINC/PubChem/ChEMBL) with our molecules (decontamination is needed, W-decontamination); quality on the project's real data
  (`data/processed` did not exist at the time of the check); the effect of truncation at 256 tokens (on long SMILES); equivalence
  of our D-MPNN to the chemprop implementation (the code was written from the description in Yang et al. 2019, there is no direct comparison).
- Warning: under high CPU load, multi-threaded torch slows down by an order of magnitude; the GNN therefore uses 1 thread by default
  (`n_threads`).
- ChemBERTa: the pooler layer is absent from the checkpoints and is initialized randomly, but is not used (last_hidden_state is taken).
