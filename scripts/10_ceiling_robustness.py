#!/usr/bin/env python
"""Шаг 10. Устойчивость оценки шума между документами (по замечаниям рецензента).
Для каждой задачи data/<processed>/<task>.measurements.csv:
  sigma_base        — как в 03 (min_docs=2, обрезка по размаху 3 ед.);
  n_doc_pairs, top_pair_share — число различных пар документов среди повторов и доля пар молекул у самой частой;
  sigma_pairs [lo, hi] — парная оценка sqrt(mean(Δ²/2)) и бутстрэп по ПАРАМ ДОКУМЕНТОВ (кластерный);
  frac_identical    — доля межисточниковых пар с одинаковым значением (|Δ| < 0.005);
  sigma_no_identical — одинаковые значения разных документов считаются одним измерением;
  sigma_stereo      — молекула = полный SMILES со стереохимией (а не блок связности InChIKey);
  sigma_one_assay   — из каждого документа берётся только его самый крупный ассей;
  frac_multi_assay  — доля пар (молекула, документ) с >1 ассеем; frac_trimmed — доля повторных молекул, снятых обрезкой.
Выход: results/<out>/ceiling_robustness.csv"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from ncmol.ceiling import doc_level, pooled_variance

ap = argparse.ArgumentParser()
ap.add_argument("--processed", default="data/full_clean")
ap.add_argument("--out", default="results/full_clean")
ap.add_argument("--n-boot", type=int, default=1000)
ap.add_argument("--max-pair-diff", type=float, default=3.0)
a = ap.parse_args()
MAX = a.max_pair_diff


def sig(d, key="mol_id"):
    s2, n, _ = pooled_variance(d.rename(columns={key: "mol_id"}) if key != "mol_id" else d, 2, MAX)
    return float(np.sqrt(s2)), n


rows = []
for t in pd.read_csv(Path(a.processed) / "tasks.csv").task:
    m = pd.read_csv(Path(a.processed) / f"{t}.measurements.csv")
    d = doc_level(m)
    base, n_rep = sig(d)
    # пары документов среди повторных молекул (после обрезки по размаху)
    g = d.groupby("mol_id")["y"]
    rep = d[(g.transform("size") >= 2) & (g.transform(lambda s: s.max() - s.min()) <= MAX)]
    pairs = []
    for mol, x in rep.groupby("mol_id"):
        for (da, ya), (db, yb) in combinations(sorted(zip(x.doc_id, x.y)), 2):
            pairs.append((da + "|" + db, ya - yb))
    pr = pd.DataFrame(pairs, columns=["pair", "delta"])
    cnt = pr.pair.value_counts()
    agg = pr.assign(h=pr.delta ** 2 / 2).groupby("pair").agg(h=("h", "sum"), n=("h", "size"))
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(agg), size=(a.n_boot, len(agg)))
    boot = np.sqrt(agg.h.to_numpy()[idx].sum(1) / agg.n.to_numpy()[idx].sum(1))
    # одинаковые значения в разных документах -> одно измерение
    dn = d.assign(yr=d.y.round(2)).drop_duplicates(["mol_id", "yr"]).drop(columns="yr")
    # стереохимия: молекула = полный SMILES
    ds = m.groupby(["smiles", "doc_id"], as_index=False)["y"].mean()
    # один (самый крупный) ассей на документ
    big = m.groupby(["doc_id", "assay_id"]).size().reset_index(name="k").sort_values("k").drop_duplicates("doc_id", keep="last")
    d1 = doc_level(m[m.assay_id.isin(big.assay_id)])
    ma = m.groupby(["mol_id", "doc_id"]).assay_id.nunique()
    n_rep_untrim = int((d.groupby("mol_id").size() >= 2).sum())
    rows.append({"task": t, "n_rep": n_rep, "sigma_base": base, "n_doc_pairs": len(cnt),
                 "top_pair_share": float(cnt.iloc[0] / cnt.sum()), "sigma_pairs": float(np.sqrt(agg.h.sum() / agg.n.sum())),
                 "sigma_pairs_lo": float(np.percentile(boot, 2.5)), "sigma_pairs_hi": float(np.percentile(boot, 97.5)),
                 "frac_identical": float((pr.delta.abs() < 0.005).mean()), "sigma_no_identical": sig(dn)[0],
                 "sigma_stereo": sig(ds, "smiles")[0], "n_rep_stereo": sig(ds, "smiles")[1],
                 "sigma_one_assay": sig(d1)[0], "frac_multi_assay": float((ma > 1).mean()),
                 "frac_trimmed": 1 - n_rep / n_rep_untrim, "frac_rep": n_rep_untrim / d.mol_id.nunique()})
out = pd.DataFrame(rows)
out.to_csv(Path(a.out) / "ceiling_robustness.csv", index=False)
pd.set_option("display.width", 250)
print(out.round(3).to_string(index=False))
