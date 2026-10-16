#!/usr/bin/env python
"""Раздел Results и приложение статьи из results/full_clean, results/full_raw и data/full_clean.
Выход: paper/sections/results.tex, paper/sections/appendix.tex (не править руками).
Запуск: python scripts/make_paper_tables.py (после 05, 09 и 10)."""
import pandas as pd

R = "results/full_clean"
NAMES = {"ecfp_gbm": "ECFP+GBM", "ecfp_rf": "ECFP+RF", "ecfp_mlp": "ECFP+MLP",
         "emb-chemberta-77m-mlm_gbm": "ChemBERTa-77M (frozen)+GBM", "emb-molformer-xl-10pct_gbm": "MoLFormer-XL-10\\% (frozen)+GBM",
         "gnn_dmpnn": "D-MPNN", "ft-chemberta-77m-mlm": "ChemBERTa-77M (fine-tuned)"}
SHORT = {"ecfp_gbm": "GBM", "ecfp_rf": "RF", "emb-chemberta-77m-mlm_gbm": "ChemB.", "emb-molformer-xl-10pct_gbm": "MoLF."}
ORDER = list(NAMES)
SPL = ["random", "scaffold", "temporal"]
EMB = ["emb-chemberta-77m-mlm_gbm", "emb-molformer-xl-10pct_gbm"]
END = "\\end{table}"

lb = pd.read_csv(f"{R}/leaderboard.csv")
pv = pd.read_csv(f"{R}/paired_vs_baseline.csv")
ce = pd.read_csv(f"{R}/ceilings.csv").sort_values("task").set_index("task")
rb = pd.read_csv(f"{R}/ceiling_robustness.csv").set_index("task")
ptf = pd.read_csv(f"{R}/per_task_fraction.csv")
mde = pd.read_csv(f"{R}/mde.csv").groupby(["model", "split"]).mde_frac.median()
tasks_meta = pd.read_csv("data/full_clean/tasks.csv").set_index("task")
nt = int(lb.n_tasks.max())


def esc(s): return s.replace("_", "\\_")
def pfmt(p): return "$<$0.001" if p < 0.001 else f"{p:.3f}"
def f(m, sp): return lb[(lb.model == m) & (lb.split == sp)].frac_ceiling.iloc[0]
def pv_row(m, sp): return pv[(pv.model == m) & (pv.split == sp) & (pv.n_tasks == nt)].iloc[0]


def rng(a, b, fmt="{:.2f}"):
    a, b = fmt.format(a), fmt.format(b)
    return a if a == b else f"{a}--{b}"


def table(caption, label, cols, header, rows, size="small"):
    return ("\\begin{table}[ht]\\centering\\" + size + "\\setlength{\\tabcolsep}{4pt}\\caption{" + caption + "}\\label{" + label + "}\n\\begin{tabular}{" + cols + "}\\toprule "
            + header + "\\\\\\midrule\n" + "\n".join(r + "\\\\" for r in rows) + "\n\\bottomrule\\end{tabular}" + END)


# --- таблица потолков ---
target_names = {}
for t in ce.index:
    tg = t.rsplit("_", 1)[0]
    target_names[t] = pd.read_csv(f"data/raw/{tg}.csv", usecols=["target_name"]).target_name.iloc[0]
rows = []
for t, r in ce.iterrows():
    q = rb.loc[t]
    rows.append(f"{esc(t)} & {int(r.n_mols_rep)} & {int(q.n_doc_pairs)} & {100 * q.top_pair_share:.0f} & {r.sigma:.2f} & "
                f"[{r.sigma_lo:.2f}, {r.sigma_hi:.2f}] & [{q.sigma_pairs_lo:.2f}, {q.sigma_pairs_hi:.2f}] & {r.rmse_floor:.2f} & {r.r2_max:.2f}")
t_ceil = table("Noise floors per task (decontaminated data). $n_{\\text{rep}}$: replicated molecules after trimming; doc.\\ pairs: distinct pairs of documents "
               "among the replicates; top \\%: share of molecule pairs from the most frequent document pair. 95\\% intervals for $\\hat s$ from a bootstrap over "
               "molecules and over document pairs (1{,}000 resamples); the latter is for the pairwise form of the estimator, which differs slightly from $\\hat s$ when molecules have more than two documents. $\\mathrm{RMSE}_{\\mathrm{floor}}$ and $R^2_{\\max}$ over all molecules of the task.",
               "tab:ceil", "lrrrcccccc".replace("cccccc", "ccccc"),
               "Task & $n_{\\text{rep}}$ & doc.\\ pairs & top \\% & $\\hat s$ & CI molecules & CI doc.\\ pairs & $\\mathrm{RMSE}_{\\mathrm{floor}}$ & $R^2_{\\max}$", rows, "footnotesize")

rows = [f"{esc(t)} & {q.sigma_base:.2f} & {100 * q.frac_identical:.0f} & {q.sigma_no_identical:.2f} & {q.sigma_stereo:.2f} & {q.sigma_one_assay:.2f} & {100 * q.frac_multi_assay:.0f}"
        for t, q in rb.iterrows()]
t_rob = table("Sensitivity of $\\hat s$ to curation choices. ident.\\ \\%: share of between-document pairs with identical values; no dup.: identical values "
              "from different documents counted once; stereo: molecules defined by stereo-aware SMILES; one assay: only the largest assay of each document; "
              "multi-assay \\%: molecule--document pairs with values from more than one assay.",
              "tab:robust", "lcccccc", "Task & $\\hat s$ & ident.\\ \\% & no dup. & stereo & one assay & multi-assay \\%", rows)

# --- f по моделям ---
rows = []
for m in ORDER:
    s = lb[lb.model == m].set_index("split")
    if s.empty:
        continue
    rows.append(f"{NAMES[m]} & {int(s.n_tasks.iloc[0])} & " + " & ".join(f"{s.loc[x, 'frac_ceiling']:.2f}" if x in s.index else "--" for x in SPL))
t_frac = table("Mean fraction of ceiling $f$ over tasks ($n$ = number of tasks), decontaminated data. Models with $n<" + str(nt) +
               "$ were run on a subset of tasks and are not comparable with the rest (see Table~\\ref{tab:subset}).",
               "tab:frac", "lcccc", "Model & $n$ & Random & Scaffold & Temporal", rows)

rows = []
for m in ORDER:
    if m == "ecfp_gbm":
        continue
    for sp in SPL:
        q = pv[(pv.model == m) & (pv.split == sp) & (pv.n_tasks == nt)]
        if q.empty:
            continue
        q = q.iloc[0]
        rows.append(f"{NAMES[m]} & {sp} & {q.mean_diff_frac_ceiling:+.3f} & {int(q.n_better)}/{int(q.n_tasks)} & {pfmt(q.p_holm)}")
t_paired = table("Paired difference in $f$ versus ECFP+GBM over " + str(nt) + " tasks (positive = better than ECFP+GBM); Wilcoxon signed-rank test, "
                 "Holm-adjusted $p$.", "tab:paired", "llccc", "Model & Split & mean diff. & better on & $p_{\\text{Holm}}$", rows)

sub = sorted(set(ptf[ptf.model == "gnn_dmpnn"].task) & set(ptf[ptf.model == "ft-chemberta-77m-mlm"].task))
sv = ptf[ptf.task.isin(sub)].groupby(["model", "split"]).frac_ceiling.mean()
rows = [f"{NAMES[m]} & " + " & ".join(f"{sv.get((m, x), float('nan')):.2f}" for x in SPL) for m in ORDER if (m, "random") in sv]
t_sub = table("Mean $f$ on the " + str(len(sub)) + " tasks where D-MPNN (3 seeds) and fine-tuned ChemBERTa (3--5 seeds) were run (" +
              ", ".join(esc(t) for t in sub) + "); all models on the same tasks.", "tab:subset", "lccc", "Model & Random & Scaffold & Temporal", rows)

# --- очищенные против неочищенных ---
rw = pd.read_csv("results/full_raw/per_task_fraction.csv")
cm = ptf.merge(rw, on=["task", "split", "model"], suffixes=("_c", "_r"))
cm = cm[cm.model.isin(["ecfp_gbm", "ecfp_rf", "ecfp_mlp"] + EMB)]
cm["df"], cm["dr"] = cm.frac_ceiling_r - cm.frac_ceiling_c, cm.rmse_r - cm.rmse_c
lk = cm.groupby(["model", "split"]).agg(fc=("frac_ceiling_c", "mean"), fr=("frac_ceiling_r", "mean"), dr=("dr", "mean"))
rows = [f"{NAMES[m]} & {sp} & {lk.loc[(m, sp), 'fc']:.2f} & {lk.loc[(m, sp), 'fr']:.2f} & {lk.loc[(m, sp), 'dr']:+.3f}" for m in ORDER for sp in SPL if (m, sp) in lk.index]
t_leak = table("Effect of decontamination: mean $f$ on decontaminated and uncleaned data and mean change in test RMSE (uncleaned $-$ decontaminated) over " + str(nt) + " tasks.",
               "tab:leak", "llccc", "Model & Split & $f$ clean & $f$ uncleaned & $\\Delta$RMSE", rows)
fl_r = pd.read_csv("results/full_raw/ceilings.csv").set_index("task").rmse_floor
dfl = (fl_r - ce.rmse_floor).dropna()
dr_main = cm[cm.model != "ecfp_mlp"].groupby(["model", "split"]).dr.mean().abs().max()
dr_mlp = cm[cm.model == "ecfp_mlp"].groupby("split").dr.mean().abs().max()
df_main = cm[cm.model != "ecfp_mlp"].df.mean()

# --- числа для текста ---
q_rf = ptf[ptf.model == "ecfp_rf"]
above = q_rf[q_rf.frac_ceiling > 1]
above_tasks = set(above.task)
rf_stats = {}
for sp in SPL:
    r = lb[(lb.model == "ecfp_rf") & (lb.split == sp)].iloc[0]
    q = q_rf[q_rf.split == sp]
    rf_stats[sp] = (q.frac_ceiling.median(), q[~q.task.isin(above_tasks)].frac_ceiling.mean(), r.frac_ceiling_at_sigma_lo, r.frac_ceiling_at_sigma_hi)
above_txt = ", ".join(f"{esc(r.task)} ({r.split})" for r in above.itertuples())
ki = ce[ce.index.str.endswith("_Ki")].sigma.median()
ic = ce[ce.index.str.endswith("_IC50")].sigma.median()
gap_rs = [pv_row(m, sp).mean_diff_frac_ceiling for m in EMB for sp in ("random", "scaffold")]
gap_t = [pv_row(m, "temporal").mean_diff_frac_ceiling for m in EMB]
better_t = [int(pv_row(m, "temporal").n_better) for m in EMB]
p_emb = max(pv_row(m, sp).p_holm for m in EMB for sp in SPL)
rf_d = [pv_row("ecfp_rf", sp).mean_diff_frac_ceiling for sp in SPL]
rf_b = [int(pv_row("ecfp_rf", sp).n_better) for sp in SPL]
mde_emb = mde.xs("random", level="split").drop("ecfp_rf").median()
tp = ptf[(ptf.split == "temporal") & ptf.task.isin(sub)].pivot_table(index="task", columns="model", values="frac_ceiling")
ft_lead = tp["ft-chemberta-77m-mlm"] - tp["ecfp_rf"]
lead_task = ft_lead.idxmax()
sv_t = sv.xs("temporal", level="split")
ft_wo = tp.drop(lead_task)[["ft-chemberta-77m-mlm", "ecfp_rf"]].mean()
wide = rb.sigma_pairs_hi - rb.sigma_pairs_lo
ratio = wide / (ce.sigma_hi - ce.sigma_lo)
n_wide = int((ratio > 1.2).sum())
r2971 = rb.loc["CHEMBL2971_IC50"]
r237 = rb.loc["CHEMBL237_IC50"]
d_noid = (rb.sigma_no_identical - rb.sigma_base)
d_ster = (rb.sigma_stereo - rb.sigma_base).abs()

txt = r"""% Generated by scripts/make_paper_tables.py -- do not edit by hand.
\section{Results}\label{sec:results}
Unless noted, results are averages over the """ + str(nt) + r""" decontaminated tasks and five seeds.

\paragraph{Noise floors.} Table~\ref{tab:ceil} lists the estimates. $\mathrm{RMSE}_{\mathrm{floor}}$ ranges from """ + f"{ce.rmse_floor.min():.2f} to {ce.rmse_floor.max():.2f}" + r""" pChEMBL units and $R^2_{\max}$ from """ + f"{ce.r2_max.min():.2f} to {ce.r2_max.max():.2f}" + r""". The median $\hat s$ is """ + f"{ki:.2f}" + r""" for $K_i$ tasks and """ + f"{ic:.2f}" + r""" for $\mathrm{IC}_{50}$ tasks, of the same order as earlier ChEMBL-wide estimates for $K_i$ (0.54)~\cite{kramer2012experimental} and the reported 25\% larger spread of $\mathrm{IC}_{50}$ data~\cite{kalliokoski2013comparability}; in our set the measurement type is confounded with target class ($K_i$ tasks are mostly GPCRs, $\mathrm{IC}_{50}$ tasks mostly kinases), so we do not attribute the difference to the measurement type. Replicates are often concentrated in few document pairs: CHEMBL237\_IC50 rests on """ + f"{int(r237.n_doc_pairs)}" + r""" document pairs, one of which contributes """ + f"{100 * r237.top_pair_share:.0f}" + r"""\% of molecule pairs. Accordingly, the bootstrap over document pairs gives intervals more than 20\% wider than the bootstrap over molecules for """ + str(n_wide) + r""" of """ + str(nt) + r""" tasks, and """ + f"{ratio['CHEMBL237_IC50']:.0f}" + r""" and """ + f"{ratio['CHEMBL2971_IC50']:.0f}" + r""" times as wide for CHEMBL237\_IC50 and CHEMBL2971\_IC50 (for the latter, """ + f"{r2971.sigma_pairs_lo:.2f}--{r2971.sigma_pairs_hi:.2f}" + r"""); it is narrower only for tasks with many document pairs. Table~\ref{tab:robust} shows the sensitivity to curation: counting identical values from different documents once raises $\hat s$ by up to """ + f"{d_noid.max():.2f}" + r""" (CHEMBL237\_Ki), and a stereo-aware molecule definition changes it by more than 0.05 for """ + str(int((d_ster > 0.05).sum())) + r""" tasks. These are the uncertainties of the reference point against which all models below are scored.

\paragraph{Fraction of ceiling.} Table~\ref{tab:frac} gives mean $f$ by model and split. ECFP+RF is the best model on every split. On random splits its mean $f$ is """ + f"{f('ecfp_rf', 'random'):.2f}" + r""", but the median is """ + f"{rf_stats['random'][0]:.2f}" + r""", the mean without the tasks where it exceeds 1 is """ + f"{rf_stats['random'][1]:.2f}" + r""", and computing the floor at the ends of the confidence interval of $\hat s$ moves the mean between """ + f"{rf_stats['random'][2]:.2f} and {rf_stats['random'][3]:.2f}" + r""". On scaffold and temporal splits the means are """ + f"{f('ecfp_rf', 'scaffold'):.2f} and {f('ecfp_rf', 'temporal'):.2f}" + r""" (medians """ + f"{rf_stats['scaffold'][0]:.2f} and {rf_stats['temporal'][0]:.2f}" + r"""). Thus most of the reducible error is captured on random splits, whereas on temporal splits about two thirds of it remains; improvements are measurable mainly on the harder splits.

\paragraph{ECFP versus frozen embeddings.} Frozen ChemBERTa-77M and MoLFormer-XL-10\% embeddings with a GBM head are below ECFP+GBM on every split (Table~\ref{tab:paired}, Figure~\ref{fig:paired}): by """ + rng(-max(gap_rs), -min(gap_rs)) + r""" in $f$ on random and scaffold splits, where they are worse on all """ + str(nt) + r""" tasks, and by """ + rng(-max(gap_t), -min(gap_t)) + r""" on temporal splits, where they are better on only """ + rng(min(better_t), max(better_t), "{}") + r""" of """ + str(nt) + r""" tasks (Holm-adjusted $p\le""" + f"{p_emb:.3f}" + r"""$ in all cases). ECFP+RF is slightly better than ECFP+GBM (""" + rng(min(rf_d), max(rf_d)) + r""" in $f$; better on """ + rng(min(rf_b), max(rf_b), "{}") + r""" of """ + str(nt) + r""" tasks). In a three-task pilot (CHEMBL237\_IC50, CHEMBL206\_IC50, CHEMBL204\_Ki) frozen embeddings had appeared better than ECFP on temporal splits; on """ + str(nt) + r""" tasks this did not hold. The conclusion concerns two small frozen models with a tree-based head, not pretrained models in general.

\paragraph{D-MPNN and fine-tuning.} Table~\ref{tab:subset} compares all models on the """ + str(len(sub)) + r""" tasks where the D-MPNN and fine-tuned ChemBERTa were run. On random and scaffold splits neither reaches ECFP, and fine-tuning does not improve on the frozen embedding. On temporal splits the fine-tuned model has the highest mean (""" + f"{sv_t['ft-chemberta-77m-mlm']:.2f}" + r""" versus """ + f"{sv_t['ecfp_rf']:.2f}" + r""" for ECFP+RF), but this comes from a single task, """ + esc(lead_task) + r""" (""" + f"{tp.loc[lead_task, 'ft-chemberta-77m-mlm']:.2f} versus {tp.loc[lead_task, 'ecfp_rf']:.2f}" + r"""); on the other """ + str(len(sub) - 1) + r""" tasks the means are """ + f"{ft_wo['ft-chemberta-77m-mlm']:.2f} and {ft_wo['ecfp_rf']:.2f}" + r""". With """ + str(len(sub)) + r""" tasks we do not test these differences.

\paragraph{Models below the noise floor.} On """ + str(len(above)) + r""" task--split combinations ECFP+RF scores $f>1$, i.e.\ its test RMSE is below the estimated floor: """ + above_txt + r""" (Figure~\ref{fig:floor}). These tasks are among those with the largest estimated floors (0.67--0.79), and all but one are random splits. If between-document noise were independent of what a model can learn, this would be impossible in expectation~\cite{kolmar2021noise}; models reaching or exceeding noise-based bounds were also reported by Crusius et al.~\cite{crusius2025fitting}. Two explanations are possible. (i) The floor is over-estimated for these tasks: it is based on molecules that are not representative of the test set, and for CHEMBL2971\_IC50 its interval over document pairs is very wide (""" + f"{r2971.sigma_pairs_lo:.2f}--{r2971.sigma_pairs_hi:.2f}" + r"""), and two of its document pairs differ by a nearly constant 2.4 and 3.0 log units (standard deviations 0.41 and 0.13 over 29 and 26 shared molecules), which suggests unit errors that the 3-unit trimming does not remove. (ii) Under random splits, molecules from the same document share assay-specific offsets that a model can learn from their neighbours in the training set, so part of the between-document ``noise'' is predictable; models are known to exploit publication-level signals in ChEMBL~\cite{blevins2025cleverhans}. CHEMBL2971\_IC50 also exceeds 1 on the scaffold split, which (ii) explains less easily. A document-grouped split would separate the two explanations and is left for the next version. In either case $f>1$ signals a problem with the floor or the split, not accuracy beyond the measurement limit.

\paragraph{Effect of decontamination.} Repeating the runs on the uncleaned data (Table~\ref{tab:leak}) changes the mean test RMSE by at most """ + f"{dr_main:.3f}" + r""" pChEMBL units for ECFP+GBM, ECFP+RF and the embedding models, and by up to """ + f"{dr_mlp:.3f}" + r""" for the unstable ECFP+MLP. Mean $f$ is """ + f"{df_main:.2f}" + r""" higher on the uncleaned data for the former models, mostly because the floors are slightly larger there (by """ + f"{dfl.mean():.3f}" + r""" on average), not because the errors are smaller. Removing molecules that occur in MoleculeNet or TDC therefore has little effect on these tasks, and the model ranking is unchanged. This is not a pure leakage test: the uncleaned test sets contain 6.7\% more molecules, decontamination removed 24\% of the replicated molecules, and overlap with pretraining corpora is not covered.

\paragraph{Minimum detectable effect.} The median single-test-set MDE for a paired comparison with ECFP+GBM is """ + f"{mde.loc[('ecfp_rf', 'random')]:.3f}, {mde.loc[('ecfp_rf', 'scaffold')]:.3f} and {mde.loc[('ecfp_rf', 'temporal')]:.3f}" + r""" in $f$ for ECFP+RF on random, scaffold and temporal splits, and about """ + f"{mde_emb:.2f}" + r""" for the embedding models, whose errors are less correlated with the baseline. These values assume independent test molecules and are therefore optimistic. The ECFP+RF advantage is below the MDE of a single test set and is detectable only when pooled over tasks, while the embedding gaps on random and scaffold splits are two to three times the MDE.

""" + t_ceil + "\n\n" + t_rob + "\n\n" + r"""\begin{figure}[ht]\centering\includegraphics[width=\linewidth]{figures/paired_diff.pdf}
\caption{Per-task difference in fraction of ceiling relative to ECFP+GBM (points: tasks; bar: mean over tasks; grey band: $\pm$ median single-test-set MDE of each model).}\label{fig:paired}\end{figure}

\begin{figure}[ht]\centering\includegraphics[width=0.5\linewidth]{figures/floor_vs_rmse.pdf}
\caption{Estimated noise floor versus test RMSE of ECFP+RF per task and split. Points below the dashed line have RMSE below the estimated floor ($f>1$).}\label{fig:floor}\end{figure}

""" + t_frac + "\n\n" + t_paired + "\n\n" + t_sub + "\n\n" + t_leak + "\n"
open("paper/sections/results.tex", "w").write(txt)

# --- приложение: воспроизводимость и f по задачам ---
rows = []
for t in ce.index:
    vals = []
    for sp in SPL:
        for m in ["ecfp_rf", "ecfp_gbm"] + EMB:
            v = ptf[(ptf.task == t) & (ptf.split == sp) & (ptf.model == m)].frac_ceiling
            vals.append(f"{v.iloc[0]:.2f}" if len(v) else "--")
    rows.append(esc(t) + " & " + " & ".join(vals))
hdr = "Task & " + " & ".join(SHORT[m] for m in ["ecfp_rf", "ecfp_gbm"] + EMB) * 1
hdr = ("Task & \\multicolumn{4}{c}{Random} & \\multicolumn{4}{c}{Scaffold} & \\multicolumn{4}{c}{Temporal}\\\\ & "
       + " & ".join([" & ".join(SHORT[m] for m in ["ecfp_rf", "ecfp_gbm"] + EMB)] * 3))
t_pertask = ("\\begin{table}[ht]\\centering\\scriptsize\\setlength{\\tabcolsep}{2.5pt}\\caption{Fraction of ceiling per task (mean over seeds). RF, GBM: ECFP models; ChemB., MoLF.: frozen "
             "embeddings with GBM head.}\\label{tab:pertask}\n\\begin{tabular}{lcccccccccccc}\\toprule " + hdr + "\\\\\\midrule\n"
             + "\n".join(r + "\\\\" for r in rows) + "\n\\bottomrule\\end{tabular}" + END)
rows = [f"{esc(t)} & {esc(target_names[t])} & {int(tasks_meta.loc[t, 'n_mols'])} & {int(tasks_meta.loc[t, 'n_docs'])}" for t in ce.index]
t_tasks = table("The 17 tasks: ChEMBL target, number of molecules and documents after decontamination.", "tab:tasks", "lp{7cm}rr",
                "Task & Target & molecules & documents", rows)
CANDS = [c["id"] for c in __import__("yaml").safe_load(open("configs/endpoints.yaml"))["chembl_candidates"]]
app = r"""% Generated by scripts/make_paper_tables.py -- do not edit by hand.
\appendix
\section{Reproducibility details}\label{app:repro}
{\sloppy\paragraph{Targets.} We queried """ + str(len(CANDS)) + r""" single-protein targets: """ + esc(", ".join(CANDS)) + r""". Splitting by measurement type gives 58 tasks, of which the 17 in Table~\ref{tab:tasks} have at least 100 replicated molecules after decontamination.\par}

\paragraph{Hyperparameter grids.} Three configurations are sampled per model from: GBM (scikit-learn \texttt{HistGradientBoostingRegressor}, early stopping): learning rate $\{0.03, 0.05, 0.1\}$, maximum iterations $\{200, 400\}$, maximum leaf nodes $\{15, 31, 63\}$, L2 regularisation $\{0, 1\}$. RF (300 trees): maximum features $\{\sqrt{p}, 0.2, 0.5\}$, minimum samples per leaf $\{1, 2, 5\}$. MLP (standardised inputs, early stopping): hidden layers $\{(256), (512, 128), (1024, 256)\}$, $\alpha\in\{10^{-4}, 10^{-3}, 10^{-2}\}$, learning rate $\{10^{-3}, 3\cdot10^{-4}\}$. D-MPNN: hidden size $\{128, 200, 300\}$, depth $\{2, 3, 4\}$, dropout $\{0, 0.1, 0.2\}$, learning rate $\{10^{-3}, 5\cdot10^{-4}\}$. The selected configurations are stored with every result.

\paragraph{Software and hardware.} Python 3.11, RDKit 2026.03.4, scikit-learn 1.6.1, PyTorch 2.4.1, transformers 4.51.0, pandas 2.3.3, NumPy 1.26.4, SciPy 1.13.1. All runs used one Apple M1 laptop (8 CPU cores; embeddings and fine-tuning on the integrated GPU via PyTorch MPS, everything else on CPU). The ECFP and embedding benchmark on 17 tasks took about one and a half days with five to six parallel jobs.

""" + t_tasks + "\n\n" + t_pertask + "\n"
open("paper/sections/appendix.tex", "w").write(app)
print("ok", nt, len(above), len(sub), lead_task)
