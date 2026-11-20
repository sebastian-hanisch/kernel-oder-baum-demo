"""Erzeugt `kob_vormessung.json`: die eingefrorene Studie (scikit-learn-Gauß-SVM, abgestimmter Wald, abgestimmtes Boosting, kleines Netz) auf den Welten der App.

Die Welten kommen aus `kob_worlds.py` (SplitMix64, `kob_rng.py`), nicht aus numpy-Zufall. Das ist **Neumessung mit den Strömen der App**: die Vormessungen des Plans
liefen mit `numpy.default_rng` und gelten nur als Erwartung; die Zahlen der README stammen aus dieser Datei und sind Bänder.

Zufalls-Seeds sind getrennt (DEMO-PLAYBOOK §4): Daten (Training 1000 + s, Test 700000 + s), Aufteilung der Kreuzvalidierung s, Modell der Bäume und des Netzes s + 50000
(Netz mit anderen Startwerten: s + 60000), Drehung 9000 + s. Je Welt 20 Zufalls-Seeds (A, B, D, P zusätzlich frische 20 bis 39), je Zufalls-Seed 4000 frische Testpunkte.

Aufruf (Test-venv, OMP_NUM_THREADS=1):
    python tools/kob_make_vormessung.py teil haupt|netz|skalierung|ringe|lernkurve|live [--jobs 12]   # rechnet einen Teil, schreibt ihn nach --teile
    python tools/kob_make_vormessung.py zusammen                                                      # fügt die Teile zu kob_vormessung.json zusammen
    python tools/kob_make_vormessung.py --check                                                       # 3 Zufalls-Seeds je Welt (A, E, P) neu rechnen und als Band gegen die Datei prüfen
Laufzeiten sind nur Orientierung und kein Befund. scikit-learn, joblib und scipy gehören nur in dieses Werkzeug und in die Tests, nicht zur Laufzeit der App.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import kob_models as M  # noqa: E402
import kob_worlds as W  # noqa: E402

OUT_FILE = ROOT / "kob_vormessung.json"
TEILE_DIR = Path(tempfile.gettempdir()) / "sh-tests" / "kob_teile"
NTEST = 4000
CV_FOLDS = 5
RF_GRID = {"max_features": ["sqrt", 1.0], "min_samples_leaf": [1, 3, 10]}
GB_CELLS = [(3, 0.1), (1, 0.1), (2, 0.1), (5, 0.1), (3, 0.3), (2, 0.3)]     # (Tiefe, Lernrate)
GB_CHECK = (25, 50, 100, 200)                                                  # Zahl der Bäume (Zwischenstände von 200)
SVM_CS = [0.1, 1, 10, 100]
LIN_CS = [0.01, 0.1, 1, 10]
MLP_GRID = {"hidden_layer_sizes": [(16,), (64,)], "alpha": [1e-3, 1e-1]}
HAUPT_WELTEN = ("A", "B", "C", "C2", "E", "D", "F", "P", "F8", "Er", "Br")
FRISCH_WELTEN = ("A", "B", "D", "P")                                           # zusätzlich Zufalls-Seeds 20 bis 39
NETZ_WELTEN = ("A", "B", "E", "D", "F", "P")
LIVE_WELTEN = ("A", "B", "E", "Er", "D", "F", "F8", "P")
LERN_WELTEN = ("A", "B", "E", "D", "F", "P")
LERN_N = (50, 100, 200, 400, 800, 1600)
RING_GITTER = {
    "g=0,01 bis 100": [10 ** (k / 2) for k in range(-4, 5)],
    "g=0,1 bis 10": [10 ** (k / 2) for k in range(-2, 3)],
    "g=0,2 bis 5": [0.2, 0.5, 1.0, 2.0, 5.0],
    "g=0,5 bis 2": [0.5, 1.0, 2.0],
}
SVM_GAMMA_FACS = list(M.GAMMA_FACTORS)


def rnd(v):
    if isinstance(v, (list, tuple)):
        return [rnd(x) for x in v]
    if isinstance(v, (float, np.floating)):
        return round(float(v), 6)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def data_for(world, s, ntrain=None, ntest=NTEST):
    n = W.N_TRAIN[world] if ntrain is None else ntrain
    rs = 9000 + s
    tr = W.generate(world, n, 1000 + s, rot_seed=rs)
    te = W.generate(world, ntest, 700000 + s, rot_seed=rs)
    return tr, te


def prepared(tr, te):
    S, T = M.scale_features(tr.X, te.X, "standardisiert")
    return np.nan_to_num(S), np.nan_to_num(T)


def gm_of(S):
    return M.gamma_from_scale(M.median_sigma(S), 1.0)


# ---------------------------------------------------------------- Teil: Hauptreihe
def one_haupt(args):
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold
    from sklearn.svm import SVC
    warnings.filterwarnings("ignore")
    world, s = args
    tr, te = data_for(world, s)
    Xtr, ytr, Xte, yte = tr.X, tr.y, te.X, te.y
    Str, Ste = prepared(tr, te)
    out = {"world": world, "seed": s, "bayes": te.bayes_error_empirical(), "bayes_exp": te.bayes_error_expected}
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=s)
    gm = gm_of(Str)
    gl = GridSearchCV(SVC(kernel="linear"), {"C": LIN_CS}, cv=cv).fit(Str, ytr)
    out["lin"] = float(np.mean(gl.predict(Ste) != yte))
    gr = GridSearchCV(SVC(kernel="rbf"), {"gamma": [gm * f for f in SVM_GAMMA_FACS], "C": SVM_CS}, cv=cv).fit(Str, ytr)
    out["rbf"] = float(np.mean(gr.predict(Ste) != yte))
    out["rbf_gfac"] = float(gr.best_params_["gamma"] / gm)
    out["rbf_C"] = float(gr.best_params_["C"])
    gmh = GridSearchCV(SVC(kernel="rbf", gamma=gm), {"C": SVM_CS}, cv=cv).fit(Str, ytr)
    out["rbf_med"] = float(np.mean(gmh.predict(Ste) != yte))
    best_svm = 1.0
    for f in SVM_GAMMA_FACS:
        for C in SVM_CS:
            m = SVC(kernel="rbf", gamma=gm * f, C=C).fit(Str, ytr)
            best_svm = min(best_svm, float(np.mean(m.predict(Ste) != yte)))
    out["rbf_orakel"] = best_svm
    rf = RandomForestClassifier(n_estimators=200, random_state=s, n_jobs=1).fit(Xtr, ytr)
    out["rf_std"] = float(np.mean(rf.predict(Xte) != yte))
    gb = GradientBoostingClassifier(random_state=s).fit(Xtr, ytr)
    out["gb_std"] = float(np.mean(gb.predict(Xte) != yte))
    rs = s + 50000
    gs = GridSearchCV(RandomForestClassifier(n_estimators=200, random_state=rs, n_jobs=1), RF_GRID, cv=cv).fit(Xtr, ytr)
    out["rf_t"] = float(np.mean(gs.predict(Xte) != yte))
    out["rf_msl"] = int(gs.best_params_["min_samples_leaf"])
    out["rf_mf"] = str(gs.best_params_["max_features"])
    best_rf = 1.0
    for prm in ParameterGrid(RF_GRID):
        m = RandomForestClassifier(n_estimators=200, random_state=rs, n_jobs=1, **prm).fit(Xtr, ytr)
        best_rf = min(best_rf, float(np.mean(m.predict(Xte) != yte)))
    out["rf_orakel"] = best_rf
    acc = np.zeros((len(GB_CELLS), len(GB_CHECK)))
    for tr_i, va_i in cv.split(Xtr, ytr):
        for ci, (d, lr) in enumerate(GB_CELLS):
            m = GradientBoostingClassifier(n_estimators=max(GB_CHECK), max_depth=d, learning_rate=lr, random_state=rs).fit(Xtr[tr_i], ytr[tr_i])
            for k, pred in enumerate(m.staged_predict(Xtr[va_i])):
                if (k + 1) in GB_CHECK:
                    acc[ci, GB_CHECK.index(k + 1)] += float(np.sum(pred != ytr[va_i]))
    ci, ki = np.unravel_index(int(np.argmin(acc)), acc.shape)
    d, lr = GB_CELLS[ci]
    m = GradientBoostingClassifier(n_estimators=GB_CHECK[ki], max_depth=d, learning_rate=lr, random_state=rs).fit(Xtr, ytr)
    out["gb_t"] = float(np.mean(m.predict(Xte) != yte))
    out["gb_depth"], out["gb_lr"], out["gb_ntrees"] = int(d), float(lr), int(GB_CHECK[ki])
    best_gb = 1.0
    for (d2, lr2) in GB_CELLS:
        m2 = GradientBoostingClassifier(n_estimators=max(GB_CHECK), max_depth=d2, learning_rate=lr2, random_state=rs).fit(Xtr, ytr)
        for k, pred in enumerate(m2.staged_predict(Xte)):
            if (k + 1) in GB_CHECK:
                best_gb = min(best_gb, float(np.mean(pred != yte)))
    out["gb_orakel"] = best_gb
    return out


HAUPT_KEYS = ("bayes", "bayes_exp", "lin", "rbf", "rbf_med", "rbf_orakel", "rf_std", "gb_std", "rf_t", "gb_t", "rf_orakel", "gb_orakel",
              "rbf_gfac", "rbf_C", "rf_msl", "rf_mf", "gb_depth", "gb_lr", "gb_ntrees")


def teil_haupt(jobs):
    from joblib import Parallel, delayed
    tasks = [(w, s) for w in HAUPT_WELTEN for s in range(40 if w in FRISCH_WELTEN else 20)]
    res = Parallel(n_jobs=jobs)(delayed(one_haupt)(t) for t in tasks)
    out = {}
    for w in HAUPT_WELTEN:
        rr = sorted((r for r in res if r["world"] == w), key=lambda r: r["seed"])
        out[w] = {"seeds": [r["seed"] for r in rr], **{k: rnd([r[k] for r in rr]) for k in HAUPT_KEYS}}
    return out


# ---------------------------------------------------------------- Teil: Skalierung (Lieferungen in natürlichen Einheiten)
def one_skalierung(s):
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.svm import SVC
    warnings.filterwarnings("ignore")
    tr, te = data_for("F", s)
    Xtr, ytr, Xte, yte = tr.X, tr.y, te.X, te.y
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=s)

    def med_c(A, B):
        sm = M.median_sigma(A)
        m = GridSearchCV(SVC(kernel="rbf", gamma=M.gamma_from_scale(sm, 1.0)), {"C": SVM_CS}, cv=cv).fit(A, ytr)
        return float(np.mean(m.predict(B) != yte)), sm

    out = {"seed": s, "bayes": te.bayes_error_empirical()}
    out["roh"], out["sigma_roh"] = med_c(Xtr, Xte)
    gmr = M.gamma_from_scale(M.median_sigma(Xtr), 1.0)
    m = GridSearchCV(SVC(kernel="rbf"), {"gamma": [gmr * f for f in SVM_GAMMA_FACS], "C": SVM_CS}, cv=cv).fit(Xtr, ytr)
    out["roh_cv"] = float(np.mean(m.predict(Xte) != yte))
    S, T = M.scale_features(Xtr, Xte, "standardisiert")
    out["std"], out["sigma_std"] = med_c(np.nan_to_num(S), np.nan_to_num(T))
    S2, T2 = M.scale_features(Xtr, Xte, "[0, 1]")
    out["minmax"], out["sigma_mm"] = med_c(S2, T2)
    d2 = ((Xtr[:, None, :] - Xtr[None, :, :]) ** 2).mean((0, 1))
    out["anteil"] = (d2 / d2.sum()).tolist()
    return out


def teil_skalierung(jobs):
    from joblib import Parallel, delayed
    res = Parallel(n_jobs=jobs)(delayed(one_skalierung)(s) for s in range(20))
    return {"seeds": list(range(20)), **{k: rnd([r[k] for r in res]) for k in ("bayes", "roh", "sigma_roh", "roh_cv", "std", "sigma_std", "minmax", "sigma_mm")},
            "anteil": rnd([r["anteil"] for r in res])}


# ---------------------------------------------------------------- Teil: Ringe (Median-Heuristik gegen Kreuzvalidierung je Gitter)
def one_ringe(args):
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.svm import SVC
    warnings.filterwarnings("ignore")
    s, stretch = args
    tr = W.generate("Ri", 100, s)
    te = W.generate("Ri", NTEST, 900000 + s)
    Xtr, Xte = tr.X.copy(), te.X.copy()
    if stretch != 1.0:
        Xtr[:, 1] *= stretch
        Xte[:, 1] *= stretch
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=s)
    sm = M.median_sigma(Xtr)
    gm = M.gamma_from_scale(sm, 1.0)
    mh = GridSearchCV(SVC(kernel="rbf", gamma=gm), {"C": SVM_CS}, cv=cv).fit(Xtr, tr.y)
    out = {"seed": s, "bayes_exp": te.bayes_error_expected, "med": float(np.mean(mh.predict(Xte) != te.y))}
    for name, facs in RING_GITTER.items():
        g = GridSearchCV(SVC(kernel="rbf"), {"gamma": [gm * f for f in facs], "C": SVM_CS}, cv=cv).fit(Xtr, tr.y)
        out[name] = float(np.mean(g.predict(Xte) != te.y))
    if stretch != 1.0:
        S, T = M.scale_features(Xtr, Xte, "standardisiert")
        gs = M.gamma_from_scale(M.median_sigma(S), 1.0)
        ms = GridSearchCV(SVC(kernel="rbf", gamma=gs), {"C": SVM_CS}, cv=cv).fit(S, tr.y)
        out["std"] = float(np.mean(ms.predict(T) != te.y))
    return out


def teil_ringe(jobs):
    from joblib import Parallel, delayed
    out = {}
    for name, seeds in (("reihe1", list(range(20))), ("frisch", list(range(100, 140)))):
        res = Parallel(n_jobs=jobs)(delayed(one_ringe)((s, 1.0)) for s in seeds)
        out[name] = {"seeds": seeds, "med": rnd([r["med"] for r in res]), "bayes_exp": rnd([r["bayes_exp"] for r in res]),
                     **{g: rnd([r[g] for r in res]) for g in RING_GITTER}}
    res = Parallel(n_jobs=jobs)(delayed(one_ringe)((s, 10.0)) for s in range(20))
    out["mal10"] = {"seeds": list(range(20)), "med": rnd([r["med"] for r in res]), "std": rnd([r["std"] for r in res]),
                    "cv_g=0,01 bis 100": rnd([r["g=0,01 bis 100"] for r in res])}
    return out


# ---------------------------------------------------------------- Teil: Netz
def fit_net(Str, ytr, Ste, yte, cv, rs, max_iter):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.model_selection import GridSearchCV
    from sklearn.neural_network import MLPClassifier
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        mg = GridSearchCV(MLPClassifier(max_iter=max_iter, random_state=rs), MLP_GRID, cv=cv, refit=True).fit(Str, ytr)
    n_msg = sum(1 for w in ws if issubclass(w.category, ConvergenceWarning))
    best = mg.best_estimator_
    return {"err": float(np.mean(mg.predict(Ste) != yte)), "unconv": int(best.n_iter_ >= max_iter), "n_iter": int(best.n_iter_), "n_msg": int(n_msg),
            "h": int(mg.best_params_["hidden_layer_sizes"][0]), "alpha": float(mg.best_params_["alpha"])}


def one_netz(args):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.svm import SVC
    warnings.filterwarnings("ignore")
    world, s = args
    tr, te = data_for(world, s)
    Str, Ste = prepared(tr, te)
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=s)
    gm = gm_of(Str)
    gr = GridSearchCV(SVC(kernel="rbf"), {"gamma": [gm * f for f in SVM_GAMMA_FACS], "C": SVM_CS}, cv=cv).fit(Str, tr.y)
    gs = GridSearchCV(RandomForestClassifier(n_estimators=200, random_state=s + 50000, n_jobs=1), RF_GRID, cv=cv).fit(tr.X, tr.y)
    n5 = fit_net(Str, tr.y, Ste, te.y, cv, s + 50000, 5000)
    n3 = fit_net(Str, tr.y, Ste, te.y, cv, s + 50000, 300)
    n5b = fit_net(Str, tr.y, Ste, te.y, cv, s + 60000, 5000)
    return {"world": world, "seed": s, "rbf": float(np.mean(gr.predict(Ste) != te.y)), "rf_t": float(np.mean(gs.predict(te.X) != te.y)),
            "net": n5["err"], "net_unconv": n5["unconv"], "net_iter": n5["n_iter"], "net_msg": n5["n_msg"], "net_h": n5["h"], "net_alpha": n5["alpha"],
            "net300": n3["err"], "net300_unconv": n3["unconv"], "net300_msg": n3["n_msg"],
            "net_b": n5b["err"], "net_b_unconv": n5b["unconv"], "net_b_msg": n5b["n_msg"]}


def teil_netz(jobs):
    from joblib import Parallel, delayed
    res = Parallel(n_jobs=jobs)(delayed(one_netz)((w, s)) for w in NETZ_WELTEN for s in range(20))
    keys = ("rbf", "rf_t", "net", "net_unconv", "net_iter", "net_msg", "net_h", "net_alpha", "net300", "net300_unconv", "net300_msg", "net_b", "net_b_unconv", "net_b_msg")
    out = {}
    for w in NETZ_WELTEN:
        rr = sorted((r for r in res if r["world"] == w), key=lambda r: r["seed"])
        out[w] = {"seeds": [r["seed"] for r in rr], **{k: rnd([r[k] for r in rr]) for k in keys}}
    return out


# ---------------------------------------------------------------- Teil: Lernkurve
def one_lern(args):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.svm import SVC
    warnings.filterwarnings("ignore")
    world, s = args
    res = {}
    for n in LERN_N:
        tr, te = data_for(world, s, ntrain=n)
        Str, Ste = prepared(tr, te)
        cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=s)
        gm = gm_of(Str)
        gr = GridSearchCV(SVC(kernel="rbf"), {"gamma": [gm * f for f in SVM_GAMMA_FACS], "C": SVM_CS}, cv=cv).fit(Str, tr.y)
        gs = GridSearchCV(RandomForestClassifier(n_estimators=200, random_state=s + 50000, n_jobs=1), RF_GRID, cv=cv).fit(tr.X, tr.y)
        res[n] = (float(np.mean(gr.predict(Ste) != te.y)), float(np.mean(gs.predict(te.X) != te.y)), te.bayes_error_empirical())
    return world, s, res


def teil_lernkurve(jobs):
    from joblib import Parallel, delayed
    res = Parallel(n_jobs=jobs)(delayed(one_lern)((w, s)) for w in LERN_WELTEN for s in range(10))
    out = {"n": list(LERN_N)}
    for w in LERN_WELTEN:
        rr = sorted((r for r in res if r[0] == w), key=lambda r: r[1])
        out[w] = {"seeds": [r[1] for r in rr], "rbf": [[rnd(r[2][n][0]) for r in rr] for n in LERN_N], "rf_t": [[rnd(r[2][n][1]) for r in rr] for n in LERN_N],
                  "bayes": [[rnd(r[2][n][2]) for r in rr] for n in LERN_N]}
    return out


# ---------------------------------------------------------------- Teil: Live-Modell gegen Studie
def one_live(args):
    warnings.filterwarnings("ignore")
    world, s = args
    tr, te = data_for(world, s)
    r = M.run_live(tr, te, "standardisiert", seed=s)
    return {"world": world, "seed": s, "krr": r.err_krr, "cart": r.err_cart, "full": r.err_cart_full, "f": r.f, "lam": r.lam, "leaf": r.leaf, "leaves": r.leaves_cart, "leaves_full": r.leaves_full,
            "bayes": r.bayes}


def teil_live(jobs):
    from joblib import Parallel, delayed
    res = Parallel(n_jobs=jobs)(delayed(one_live)((w, s)) for w in LIVE_WELTEN for s in range(20))
    keys = ("krr", "cart", "full", "f", "lam", "leaf", "leaves", "leaves_full", "bayes")
    out = {}
    for w in LIVE_WELTEN:
        rr = sorted((r for r in res if r["world"] == w), key=lambda r: r["seed"])
        out[w] = {"seeds": [r["seed"] for r in rr], **{k: rnd([r[k] for r in rr]) for k in keys}}
    return out


TEILE = {"haupt": teil_haupt, "netz": teil_netz, "skalierung": teil_skalierung, "ringe": teil_ringe, "lernkurve": teil_lernkurve, "live": teil_live}


def meta():
    import platform

    import scipy
    import sklearn
    return {
        "zufallsquelle": "SplitMix64 (kob_rng.py); Normalzahlen als NormalDist().inv_cdf(random())",
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "scikit-learn": sklearn.__version__, "system": platform.system(),
        "testpunkte_je_zufalls_seed": NTEST, "zufalls_seeds": "0 bis 19 (A, B, D, P zusätzlich frisch 20 bis 39)",
        "gitter": {
            "lineare SVM C": LIN_CS, "Gauß-SVM Faktor g auf gamma": SVM_GAMMA_FACS, "Gauß-SVM C": SVM_CS, "Wald": {k: [str(x) for x in v] for k, v in RF_GRID.items()},
            "Boosting (Tiefe, Lernrate)": GB_CELLS, "Boosting Zahl der Bäume": list(GB_CHECK), "Netz": {"verdeckte Einheiten": [16, 64], "alpha": [1e-3, 1e-1], "Iterationsgrenze": 5000},
        },
    }


def zusammen(teile_dir):
    data = {"meta": meta()}
    for name in TEILE:
        p = teile_dir / f"{name}.json"
        if not p.exists():
            print("fehlt:", p)
            return 1
        data[name] = json.loads(p.read_text(encoding="utf-8"))
    OUT_FILE.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print("geschrieben:", OUT_FILE, f"{OUT_FILE.stat().st_size / 1024:.0f} KiB")
    return 0


def check(welten=("A", "E", "P"), seeds=(0, 1, 2), tol=0.02):
    """Rechnet `seeds` je Welt neu und vergleicht Mittel über diese Zufalls-Seeds als Band (tol) mit der Datei; Wald und Boosting hängen an Version und Plattform."""
    from joblib import Parallel, delayed
    stored = json.loads(OUT_FILE.read_text(encoding="utf-8"))["haupt"]
    res = Parallel(n_jobs=min(8, len(welten) * len(seeds)))(delayed(one_haupt)((w, s)) for w in welten for s in seeds)
    bad = 0
    for w in welten:
        for key in ("bayes", "lin", "rbf", "rf_t", "gb_t"):
            new = np.mean([r[key] for r in res if r["world"] == w])
            old = np.mean([stored[w][key][stored[w]["seeds"].index(s)] for s in seeds])
            ok = abs(new - old) <= tol
            bad += not ok
            print(f"{w:3s} {key:5s} neu {new:.4f} Datei {old:.4f} {'ok' if ok else 'ABWEICHUNG'}")
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="3 Zufalls-Seeds je Welt (A, E, P) neu rechnen und als Band gegen die Datei prüfen")
    ap.add_argument("--teile", default=str(TEILE_DIR), help="Ordner für die Teilergebnisse")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("befehl", nargs="*", help="teil <name> | zusammen")
    a = ap.parse_args(argv)
    if a.check:
        return check()
    teile = Path(a.teile)
    teile.mkdir(parents=True, exist_ok=True)
    if a.befehl[:1] == ["zusammen"]:
        return zusammen(teile)
    if a.befehl[:1] == ["teil"] and len(a.befehl) == 2 and a.befehl[1] in TEILE:
        t0 = time.perf_counter()
        res = TEILE[a.befehl[1]](a.jobs)
        (teile / f"{a.befehl[1]}.json").write_text(json.dumps(res, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        print(f"Teil {a.befehl[1]} geschrieben nach {teile} (Laufzeit {time.perf_counter() - t0:.0f} s, nur Orientierung)")
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
