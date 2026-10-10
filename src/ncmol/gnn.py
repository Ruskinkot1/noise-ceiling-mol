"""D-MPNN (Yang et al. 2019, направленные рёбра) на чистом torch. Регрессия pActivity, CPU.

Интерфейс sklearn: fit(X, y), predict(X), где X — массив SMILES (а не матрица признаков).
Детерминизм: сид -> torch.manual_seed + фиксированные перестановки (np.random.default_rng(seed)).
Ранняя остановка по валидации (val_frac от train), лучшие веса восстанавливаются.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from rdkit import Chem, RDLogger
from torch import nn

from .device import resolve_device

RDLogger.DisableLog("rdApp.*")

_ELEMS = [6, 7, 8, 9, 15, 16, 17, 35, 53, 5, 14, 34]  # остальные -> "other"
_HYB = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3]
_BT = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]


def _onehot(v, choices):
    out = [0.0] * (len(choices) + 1)
    out[choices.index(v) if v in choices else len(choices)] = 1.0
    return out


def atom_features(a) -> list:
    return (_onehot(a.GetAtomicNum(), _ELEMS) + _onehot(a.GetTotalDegree(), [1, 2, 3, 4])
            + _onehot(a.GetFormalCharge(), [-1, 0, 1]) + _onehot(a.GetTotalNumHs(), [0, 1, 2, 3])
            + _onehot(a.GetHybridization(), _HYB)
            + [float(a.GetIsAromatic()), float(a.IsInRing()), a.GetMass() * 0.01])


def bond_features(b) -> list:
    return _onehot(b.GetBondType(), _BT) + [float(b.GetIsConjugated()), float(b.IsInRing())]


ATOM_DIM = len(atom_features(Chem.MolFromSmiles("C").GetAtomWithIdx(0)))
BOND_DIM = len(bond_features(Chem.MolFromSmiles("CC").GetBondWithIdx(0)))


def mol_graph(smiles: str):
    """-> (x[n,ATOM_DIM], edge_src[e], edge_dst[e], edge_attr[e,BOND_DIM], rev[e]); e = 2*число связей.
    Невалидный SMILES / пустая молекула -> один нулевой атом без рёбер."""
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None or mol.GetNumAtoms() == 0:
        return (np.zeros((1, ATOM_DIM), np.float32), np.zeros(0, np.int64), np.zeros(0, np.int64),
                np.zeros((0, BOND_DIM), np.float32), np.zeros(0, np.int64))
    x = np.array([atom_features(a) for a in mol.GetAtoms()], np.float32)
    src, dst, attr = [], [], []
    for b in mol.GetBonds():
        i, j, f = b.GetBeginAtomIdx(), b.GetEndAtomIdx(), bond_features(b)
        src += [i, j]; dst += [j, i]; attr += [f, f]
    # рёбра добавлены парами (2k, 2k+1): обратное ребро = индекс ^ 1
    rev = np.arange(len(src), dtype=np.int64) ^ 1
    return (x, np.array(src, np.int64), np.array(dst, np.int64),
            np.array(attr, np.float32).reshape(-1, BOND_DIM), rev)


def collate(graphs):
    xs, src, dst, attr, rev, batch = [], [], [], [], [], []
    off = eoff = 0
    for k, (x, s, d, a, r) in enumerate(graphs):
        xs.append(x); src.append(s + off); dst.append(d + off); attr.append(a); rev.append(r + eoff)
        batch.append(np.full(len(x), k, np.int64))
        off += len(x); eoff += len(s)
    cat = np.concatenate
    return (torch.from_numpy(cat(xs)), torch.from_numpy(cat(src)), torch.from_numpy(cat(dst)),
            torch.from_numpy(cat(attr)), torch.from_numpy(cat(rev)), torch.from_numpy(cat(batch)), len(graphs))


def _to(batch, dev):
    return tuple(t.to(dev) if torch.is_tensor(t) else t for t in batch)


class DMPNN(nn.Module):
    def __init__(self, hidden=200, depth=3, dropout=0.1, ffn_hidden=200):
        super().__init__()
        self.depth = depth
        self.w_i = nn.Linear(ATOM_DIM + BOND_DIM, hidden, bias=False)
        self.w_h = nn.Linear(hidden, hidden, bias=False)
        self.w_o = nn.Linear(ATOM_DIM + hidden, hidden)
        self.drop = nn.Dropout(dropout)
        self.ffn = nn.Sequential(nn.Linear(hidden, ffn_hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(ffn_hidden, 1))

    def forward(self, x, src, dst, attr, rev, batch, n_mol):
        n = x.shape[0]
        h0 = torch.relu(self.w_i(torch.cat([x[src], attr], 1)))
        h = h0
        for _ in range(self.depth - 1):
            a = torch.zeros(n, h.shape[1]).index_add_(0, dst, h)   # сумма входящих сообщений в атом
            m = a[src] - h[rev]                                     # без обратного ребра
            h = self.drop(torch.relu(h0 + self.w_h(m)))
        a = torch.zeros(n, h.shape[1]).index_add_(0, dst, h)
        hv = self.drop(torch.relu(self.w_o(torch.cat([x, a], 1))))
        cnt = torch.zeros(n_mol).index_add_(0, batch, torch.ones(n)).clamp(min=1).unsqueeze(1)
        mol = torch.zeros(n_mol, hv.shape[1]).index_add_(0, batch, hv) / cnt   # mean-pooling
        return self.ffn(mol).squeeze(1)


class DMPNNRegressor:
    """sklearn-подобный регрессор. X — SMILES (список/массив строк)."""

    def __init__(self, seed=0, hidden=200, depth=3, dropout=0.1, lr=1e-3, batch_size=64,
                 max_epochs=60, patience=10, val_frac=0.1, n_threads=1, device=None):
        self.seed, self.hidden, self.depth, self.dropout = seed, hidden, depth, dropout
        self.lr, self.batch_size, self.max_epochs, self.patience = lr, batch_size, max_epochs, patience
        self.val_frac, self.n_threads = val_frac, n_threads
        self.device = device  # None -> NCMOL_DEVICE или cpu; см. ncmol.device
        self.n_epochs_ = None

    def _dev(self):
        return resolve_device(self.device)

    def _graphs(self, X):
        return [mol_graph(s) for s in np.asarray(X, dtype=object).ravel()]

    def _predict_graphs(self, graphs, bs=256):
        self.net_.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(graphs), bs):
                out.append(self.net_(*_to(collate(graphs[i:i + bs]), self._dev())).cpu().numpy())
        return np.concatenate(out) * self.y_std_ + self.y_mean_ if out else np.zeros(0)

    def fit(self, X, y):
        if self.n_threads:
            torch.set_num_threads(self.n_threads)
        y = np.asarray(y, dtype=np.float64)
        graphs = self._graphs(X)
        rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        n = len(graphs)
        perm = rng.permutation(n)
        n_val = int(round(n * self.val_frac)) if n >= 20 else 0
        va, tr = perm[:n_val], perm[n_val:]
        self.y_mean_, self.y_std_ = float(y[tr].mean()), float(y[tr].std()) or 1.0
        yt = ((y - self.y_mean_) / self.y_std_).astype(np.float32)
        dev = self._dev()
        self.net_ = DMPNN(self.hidden, self.depth, self.dropout).to(dev)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        best, best_state, bad = np.inf, None, 0
        for ep in range(self.max_epochs):
            self.net_.train()
            order = rng.permutation(tr)
            for i in range(0, len(order), self.batch_size):
                idx = order[i:i + self.batch_size]
                if len(idx) < 2:
                    continue  # BatchNorm нет, но шаг по одному примеру шумен
                loss = nn.functional.mse_loss(self.net_(*_to(collate([graphs[j] for j in idx]), dev)),
                                              torch.from_numpy(yt[idx]).to(dev))
                opt.zero_grad(); loss.backward(); opt.step()
            if n_val:
                self.net_.eval()
                with torch.no_grad():
                    pv = np.concatenate([self.net_(*_to(collate([graphs[j] for j in va[k:k + 256]]), dev)).cpu().numpy()
                                         for k in range(0, n_val, 256)])
                score = float(np.mean((pv - yt[va]) ** 2))
                if score < best - 1e-6:
                    best, bad, best_state, self.n_epochs_ = score, 0, copy.deepcopy(self.net_.state_dict()), ep + 1
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
            else:
                self.n_epochs_ = ep + 1
        if best_state is not None:
            self.net_.load_state_dict(best_state)
        return self

    def predict(self, X):
        return self._predict_graphs(self._graphs(X))
