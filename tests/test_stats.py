"""Тесты stats.py / metrics.py на синтетике с известными ответами."""
import numpy as np
import pytest

from ncmol import metrics, stats


def _data(n=600, sigma=0.5, seed=0, model_err=0.3, docs=2):
    rng = np.random.default_rng(seed)
    t = rng.normal(0, 1, n)
    nd = np.full(n, docs)
    y = t + rng.normal(0, sigma / np.sqrt(docs), n)
    return rng, t, y, nd


def test_holm_and_bh_known_values():
    p = [0.01, 0.04, 0.03, 0.005]
    assert np.allclose(stats.holm(p), [0.03, 0.06, 0.06, 0.02])
    assert np.allclose(stats.benjamini_hochberg(p), [0.02, 0.04, 0.04, 0.02])
    h = stats.holm([0.01, np.nan, 0.02])
    assert np.isnan(h[1]) and np.allclose(h[[0, 2]], [0.02, 0.02])


def test_cluster_bootstrap_ci_covers_and_clusters_matter():
    rng = np.random.default_rng(1)
    # 100 молекул x 5 строк с общим смещением внутри молекулы: кластерный ДИ шире наивного
    mu = rng.normal(0, 1, 100)
    ids = np.repeat(np.arange(100), 5)
    x = mu[ids] + rng.normal(0, 0.1, 500)
    d = stats.cluster_bootstrap(ids, lambda i: x[i].mean(), 500, seed=0)[:, 0]
    lo, hi = stats.percentile_ci(d)
    assert lo < x.mean() < hi
    naive = x.std() / np.sqrt(len(x))
    assert d.std() > 3 * naive  # истинная SE ≈ 0.1, наивная ≈ 0.045


def test_metrics_ci_known_rmse():
    rng = np.random.default_rng(2)
    n = 800
    y = rng.normal(0, 1, n)
    p = y + rng.normal(0, 0.4, n)
    r = stats.metrics_ci(y, p, np.arange(n), n_boot=300)
    est, lo, hi = r["rmse"]
    assert lo < est < hi and abs(est - 0.4) < 0.03
    assert abs(r["mae"][0] - 0.4 * np.sqrt(2 / np.pi)) < 0.03
    assert r["spearman"][0] > 0.9


def test_frac_ceiling_known_values():
    # rmse_mean=1, floor=0.2, rmse=0.6 -> (1-0.6)/(1-0.2)=0.5
    assert metrics.frac_ceiling(0.6, 1.0, 0.2) == pytest.approx(0.5)
    assert metrics.frac_ceiling(0.2, 1.0, 0.2) == pytest.approx(1.0)
    assert metrics.frac_ceiling(1.0, 1.0, 0.2) == pytest.approx(0.0)
    assert metrics.frac_ceiling(1.2, 1.0, 0.2) < 0


def test_ideal_model_hits_ceiling_and_ci_contains_it():
    """Модель = истинное значение: RMSE на шумной метке ≈ floor, доля потолка ≈ 1."""
    sigma, docs = 0.6, 2
    rng, t, y, nd = _data(n=3000, sigma=sigma, docs=docs)
    ytm = np.zeros_like(y)  # «среднее по train» = истинное среднее 0
    est, lo, hi, _ = stats.frac_ceiling_ci(y, t, ytm, nd, np.arange(len(y)), sigma, n_boot=300)
    assert abs(est - 1) < 0.05 and lo < 1.0 < hi + 0.05
    # предсказание = среднее -> 0
    est0, *_ = stats.frac_ceiling_ci(y, ytm, ytm, nd, np.arange(len(y)), sigma, n_boot=50)
    assert abs(est0) < 1e-9


def test_attenuation_correction_recovers_true_error():
    """Шум метки завышает RMSE: sqrt(rmse_obs^2 - floor^2) ≈ истинная ошибка модели."""
    sigma, docs, n = 0.8, 1, 20000
    rng = np.random.default_rng(3)
    t = rng.normal(0, 1, n)
    y = t + rng.normal(0, sigma, n)
    pred = t + rng.normal(0, 0.3, n)  # истинная ошибка 0.3
    rmse = np.sqrt(np.mean((y - pred) ** 2))
    mean_rmse = np.sqrt(np.mean((y - 0) ** 2))
    floor = metrics.rmse_floor(sigma, 1.0 / docs)
    a = metrics.attenuation_corrected(rmse, mean_rmse, floor)
    assert abs(a["rmse_true"] - 0.3) < 0.02
    assert abs(a["rmse_mean_true"] - 1.0) < 0.03  # sd истинных значений = 1
    assert not a["below_floor"]
    # надёжность и дезаттенюация: corr(y,t) = 1/sqrt(1+0.64) -> r_true = 1
    rel = metrics.reliability(np.var(y), floor**2)
    assert rel == pytest.approx(1 / 1.64, abs=0.03)
    r_obs = np.corrcoef(y, t)[0, 1]
    assert metrics.disattenuated_corr(r_obs, rel) == pytest.approx(1.0, abs=0.03)
    # R2 относительно истинных: 1 - 0.09/1
    mse = np.mean((y - pred) ** 2)
    assert metrics.r2_corrected(mse, np.var(y), floor) == pytest.approx(1 - 0.09, abs=0.03)


def test_attenuation_flags_below_floor():
    assert metrics.attenuation_corrected(0.1, 1.0, 0.3)["below_floor"]


def test_paired_bootstrap_detects_real_difference_and_null():
    rng = np.random.default_rng(4)
    n = 1500
    t = rng.normal(0, 1, n)
    y = t + rng.normal(0, 0.3, n)
    a = t + rng.normal(0, 0.2, n)      # лучше
    b = t + rng.normal(0, 0.5, n)      # хуже
    c = t + rng.normal(0, 0.5, n)      # такой же, как b, другая реализация
    ids = np.arange(n)
    rm = lambda p: (lambda i: np.sqrt(np.mean((y[i] - p[i]) ** 2)))
    r = stats.paired_diff_ci(rm(a), rm(b), ids, n_boot=300)
    assert r["hi"] < 0 and not r["inside_noise"] and stats.slice_flag(r["lo"], r["hi"]) == "модель лучше"
    r0 = stats.paired_diff_ci(rm(b), rm(c), ids, n_boot=300)
    assert r0["inside_noise"] and stats.slice_flag(r0["lo"], r0["hi"]) == "внутри шума"
    # парность: разность на одинаковых ресэмплах -> ДИ уже, чем у разности независимых ресэмплов
    assert (r0["hi"] - r0["lo"]) < 0.1


def test_type_i_error_of_paired_bootstrap_roughly_nominal():
    rng = np.random.default_rng(5)
    n, rej, reps = 200, 0, 60
    for k in range(reps):
        y = rng.normal(0, 1, n)
        a, b = y + rng.normal(0, 0.5, n), y + rng.normal(0, 0.5, n)
        ids = np.arange(n)
        r = stats.paired_diff_ci(lambda i: np.sqrt(np.mean((y[i] - a[i]) ** 2)),
                                 lambda i: np.sqrt(np.mean((y[i] - b[i]) ** 2)), ids, n_boot=150, seed=k)
        rej += not r["inside_noise"]
    assert rej / reps < 0.2  # номинально 0.05; широкий допуск из-за малого reps


def test_wilcoxon_tasks():
    d = np.array([-0.3, -0.2, -0.25, -0.1, -0.4, -0.15, -0.05, -0.35])
    r = stats.wilcoxon_tasks(d)
    assert r["n_tasks"] == 8 and r["n_negative"] == 8 and r["p"] == pytest.approx(2 / 2**8)
    assert np.isnan(stats.wilcoxon_tasks(d[:5])["p"])  # n=5: min p = 0.0625
    assert np.isnan(stats.wilcoxon_tasks(np.zeros(8))["p"])


def test_mde_model_matches_simulation():
    """Гауссова формула MDE: SE разности RMSE из симуляции ≈ теоретическая."""
    rng = np.random.default_rng(6)
    n, v, rho, s = 400, 0.25, 0.7, 0.5
    cov = v * np.array([[1, rho], [rho, 1]])
    diffs = []
    for _ in range(1500):
        u = rng.multivariate_normal([0, 0], cov, n)
        eps = rng.normal(0, s, n)
        ea, eb = u[:, 0] + eps, u[:, 1] + eps
        diffs.append(np.sqrt(np.mean(ea**2)) - np.sqrt(np.mean(eb**2)))
    se_sim = np.std(diffs)
    mde = stats.mde_rmse_gaussian(n, s, v, rho)
    assert mde / (1.96 + 0.8416) == pytest.approx(se_sim, rel=0.12)


def test_mde_monotone_in_noise_and_n():
    m = stats.mde_rmse_gaussian
    assert m(400, 0.6, 0.25) > m(400, 0.3, 0.25)
    assert m(100, 0.4, 0.25) > m(1000, 0.4, 0.25)
    assert m(1000, 0.4, 0.25) == pytest.approx(m(100, 0.4, 0.25) / np.sqrt(10))
    assert stats.mde_from_se(1.0) == pytest.approx(1.96 + 0.8416, abs=1e-3)
    assert stats.mde_frac_units(0.1, 1.0, 0.2) == pytest.approx(0.125)


def test_tanimoto_and_cliffs():
    fp = np.array([[1, 1, 1, 1, 0, 0],
                   [1, 1, 1, 0, 0, 0],   # T(0,1)=3/4
                   [0, 0, 0, 0, 1, 1],   # T=0 с остальными
                   [1, 1, 1, 1, 0, 0]], float)
    S = stats.tanimoto_matrix(fp, fp)
    assert S[0, 1] == pytest.approx(0.75) and S[0, 2] == 0 and S[0, 3] == 1
    y = np.array([5.0, 8.0, 5.0, 5.2])
    # (0,1): sim .75, |Δ|=3 ; (1,3): sim .75, |Δ|=2.8 ; (0,3): sim 1, Δ=.2 -> не обрыв
    pr = stats.cliff_pairs(fp, y, sim_min=0.7, delta_min=1.5)
    assert {tuple(x) for x in pr} == {(0, 1), (1, 3)}
    assert len(stats.cliff_pairs(fp, y, sim_min=0.8, delta_min=1.5)) == 0
    mt = stats.max_tanimoto_to_train(fp[[1]], fp[[0, 2]])
    assert mt[0] == pytest.approx(0.75)
    mask = stats.cliff_mask_vs_train(fp[[1, 2]], y[[1, 2]], fp[[0, 3]], y[[0, 3]], 0.7, 1.5)
    assert mask.tolist() == [True, False]


def test_cliff_pair_metrics():
    y = np.array([5.0, 8.0, 5.0, 8.0])
    pairs = np.array([[0, 1], [2, 3]])
    good = stats.cliff_pair_metrics(pairs, y, y)
    assert good["sign_acc"] == 1 and good["rmse_delta"] == 0
    flat = stats.cliff_pair_metrics(pairs, y, np.full(4, 6.5))
    assert flat["rmse_delta"] == pytest.approx(3.0)  # плоская модель не видит обрыва
    assert stats.cliff_pair_metrics(np.empty((0, 2), int), y, y)["n_pairs"] == 0


def test_p_noise_exceeds():
    # sigma=1, n=1 обоим: sd разности = sqrt2; P(|N|>=1.96*sqrt2) = 0.05
    assert stats.p_noise_exceeds(1.96 * np.sqrt(2), 1.0) == pytest.approx(0.05, abs=1e-3)
    assert stats.p_noise_exceeds(2.0, 0.5, 4, 4) < stats.p_noise_exceeds(2.0, 0.5, 1, 1)
