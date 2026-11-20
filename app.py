"""Kernel oder Baum oder Netz: welches Verfahren, auf welcher Welt, gegen welchen Gegner - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Zwei verschiedene Dinge, getrennt gezeigt: die **Studie** (eingefroren, `kob_vormessung.json`: scikit-learn-Gauß-SVM gegen abgestimmten Wald und abgestimmtes Boosting, kleines Netz,
20 Zufalls-Seeds je Welt) und das **Live-Modell** (numpy: Kernel-Ridge-Klassifikator gegen eine einzelne CART auf einer Instanz mit bis zu 600 Zeilen).

Aufbau und Regeln stehen im DEMO-PLAYBOOK.md (§2). Lauffähig mit: streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st

import kob_constants as C
import kob_decision as DEC
import kob_evaluation as E
import kob_frozen as FZ
import kob_models as M
import kob_texts as TX
import kob_tree as T
import kob_visualization as V
import kob_worlds as W
from kob_layout import dunkles_design, kopf_css
from kob_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from kob_rng import SplitMix64

st.set_page_config(page_title="Kernel oder Baum oder Netz – Sebastian Hanisch", layout="wide")

ANGLES_CURVE = (0, 15, 30, 45)


# ---------------------------------------------------------------- Live-Rechnungen (zwischengespeichert; je Eintrag unter etwa 2 s)
@st.cache_data(show_spinner=False)
def _live_instance(world, n, seed, scaling):
    """Eine Instanz der Welt (Training: Zufalls-Seed der App, Test: 4 000 frische Punkte) und das Live-Paar Kernel Ridge gegen CART."""
    rs = 9000 + seed
    tr = W.generate(world, n, seed, rot_seed=rs)
    te = W.generate(world, C.N_TEST_LIVE, 700000 + seed, rot_seed=rs)
    return M.run_live(tr, te, scaling, seed)


@st.cache_data(show_spinner=False)
def _boundary(n, seed, world, deg):
    """Entscheidungsflächen von Kernel Ridge und CART (Blattgröße per Kreuzvalidierung) auf einer 2D-Welt; Gitter 60 x 60."""
    tr = W.generate(world, n, seed, rot_deg=float(deg))
    Str, Ste = M.scale_features(tr.X, tr.X, "standardisiert")
    krr = M.select_krr(Str, tr.y, SplitMix64(seed + 31))
    tree, leaf = M.fit_cart(tr.X, tr.y, SplitMix64(seed + 77))
    lo, hi = tr.X.min(0) - 0.1, tr.X.max(0) + 0.1
    g = C.BOUNDARY_GRID
    xs, ys = np.linspace(lo[0], hi[0], g), np.linspace(lo[1], hi[1], g)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel()])
    m_, s_ = tr.X.mean(0), tr.X.std(0)
    krr_pred = krr.predict((pts - m_) / s_).reshape(g, g)
    cart_pred = T.predict(tree, pts).reshape(g, g)
    return tr.X, tr.y, (lo[0], hi[0], lo[1], hi[1]), krr_pred, cart_pred, krr.f, leaf, tree.n_leaves


@st.cache_data(show_spinner=False)
def _rotation_run(n, seed, deg):
    """Live-Paar auf der 2D-Schwellenregel (E2), die Merkmale um `deg` Grad gedreht: Fehler von Kernel Ridge, CART (Kreuzvalidierung), voll gewachsener CART und deren Blätterzahl."""
    tr = W.generate("E2", n, seed, rot_deg=float(deg))
    te = W.generate("E2", C.N_TEST_LIVE, 700000 + seed, rot_deg=float(deg))
    r = M.run_live(tr, te, "standardisiert", seed)
    return r.err_krr, r.err_cart, r.err_cart_full, r.leaves_full, r.bayes


@st.cache_data(show_spinner=False)
def _gram_invariance(n, seed, deg):
    """Größter Unterschied der Gauß-Gram-Matrizen der 2D-Schwellenregel vor und nach der Drehung um `deg` Grad (ohne Standardisierung; gleiches gamma)."""
    base = W.generate("E2", min(n, 200), seed)
    rot = W.generate("E2", min(n, 200), seed, rot_deg=float(deg))
    gamma = M.gamma_from_scale(M.median_sigma(base.X), 1.0)
    return float(np.max(np.abs(M.gauss_gram(base.X, base.X, gamma) - M.gauss_gram(rot.X, rot.X, gamma))))


@st.cache_data(show_spinner=False)
def _noise_run(n, seed, k):
    """2D-Schwellenregel (E2) mit k Rauschmerkmalen (Normalzahlen, ohne Bezug zur Klasse): Fehler von Kernel Ridge und CART, Anteil der zwei relevanten Spalten am Abstandsquadrat (standardisiert)."""
    def draw(m, s):
        w2 = W.generate("E2", m, s)
        if k == 0:
            return w2.X, w2
        noise = W.normal_array(SplitMix64(s + 123456), (m, k))
        return np.column_stack([w2.X, noise]), w2

    Xtr, wtr = draw(n, seed)
    Xte, wte = draw(C.N_TEST_LIVE, 700000 + seed)
    Str, Ste = M.scale_features(Xtr, Xte, "standardisiert")
    krr = M.select_krr(Str, wtr.y, SplitMix64(seed + 31))
    e_k = float(np.mean(krr.predict(Ste) != wte.y))
    tree, leaf = M.fit_cart(Xtr, wtr.y, SplitMix64(seed + 77))
    e_t = float(np.mean(T.predict(tree, Xte) != wte.y))
    return e_k, e_t, M.relevant_share(Str, 2), wte.bayes_error_empirical()


@st.cache_data(show_spinner=False)
def _scaling_run(n, seed):
    """Lieferungen (Welt F): Kernel Ridge auf den drei Skalen (roh, standardisiert, [0, 1]); dazu die Anteile der Spalten am mittleren Abstandsquadrat (roh) und σ_med je Skala."""
    rs = 9000 + seed
    tr = W.generate("F", n, seed, rot_seed=rs)
    te = W.generate("F", C.N_TEST_LIVE, 700000 + seed, rot_seed=rs)
    out = {}
    for mode in C.SCALINGS:
        S, Tt = M.scale_features(tr.X, te.X, mode)
        S, Tt = np.nan_to_num(S), np.nan_to_num(Tt)
        krr = M.select_krr(S, tr.y, SplitMix64(seed + 31))
        out[mode] = (float(np.mean(krr.predict(Tt) != te.y)), krr.sigma_med)
    return out, M.column_shares(tr.X).tolist(), te.bayes_error_empirical()


# ---------------------------------------------------------------- Kopf, Einleitung
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🌳 Kernel oder Baum oder Netz: welches Verfahren, auf welcher Welt, gegen welchen Gegner")
st.markdown(TX.INTRO_MD)

with st.expander("So funktioniert der Vergleich", expanded=True):
    st.markdown(TX.HOW_MD)

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Welt und Instanz**")
    welt = st.selectbox("Welt", C.LIVE_WORLDS, key="welt_select", format_func=lambda w: C.WORLD_SELECT_LABELS[w], help="Erzeugermodell mit bekannter Wahrscheinlichkeit; gilt für alle Experimente außer 3 und 4.")
    n = st.slider("Zeilen n (Live-Instanz)", *bounds("n_slider"), step=C.N_STEP, key="n_slider", help="Trainingszeilen der Live-Instanz; die Studie rechnet mit 300 bis 400 Zeilen je Welt.")
    seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), key="seed_input", step=1, help="Gilt nur für die Live-Instanz; nicht der Zufalls-Seed der Studie.")
    st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed)
    st.markdown("**Gegner und Skala**")
    gegner = st.radio("Gegner des Kerns (Studie)", C.GEGNER, key="gegner_radio", help="Einstellung von Wald und Boosting in der Studie; die Untergrenze wählt die Gitterzelle je Zufalls-Seed im Nachhinein am Test.")
    skalierung = st.radio("Skala der Merkmale (Kern, live)", C.SCALINGS, key="skalierung_radio", help="Skala, auf der das Live-Modell den Abstand rechnet; Bäume sind davon unabhängig.")
    st.markdown("**Experimente 3 und 4**")
    winkel = st.slider("Drehwinkel in Grad (Experiment 3)", *bounds("winkel_slider"), step=C.ANGLE_STEP, key="winkel_slider", help="Dreht die zwei Merkmale der Schwellenregel E2 vor dem Training und dem Test.")
    rauschen = st.slider("Rauschmerkmale (Experiment 4)", *bounds("rauschen_slider"), key="rauschen_slider", help="Zusätzliche Normalzahl-Merkmale ohne Bezug zur Klasse in der 2D-Welt E2.")

sync_query_params({"welt_select": welt, "n_slider": int(n), "seed_input": int(seed), "gegner_radio": gegner, "skalierung_radio": skalierung, "winkel_slider": int(winkel),
                   "rauschen_slider": int(rauschen)})
n, seed, winkel, rauschen = int(n), int(seed), int(winkel), int(rauschen)

# ---------------------------------------------------------------- Ergebnis: eine Welt, zwei Fassungen
st.markdown(f"## 🎯 Ergebnis: {W.WORLD_LABELS[welt]}")
studie_cols = st.columns(2)

comp = E.compare_study(welt, gegner)
with studie_cols[0]:
    st.markdown("**Studie (eingefroren): Wald und Boosting " + ("abgestimmt" if gegner == "abgestimmt" else ("in Standardeinstellung" if gegner == "Standard" else "am Test gewählt")) + "**")
    kind, text = E.verdict_study(comp)
    getattr(st, kind)(text)
with studie_cols[1]:
    st.markdown("**Live-Modell (eine CART, Blattgröße per Kreuzvalidierung)**")
    kind, text = E.verdict_live_study(welt)
    getattr(st, kind)(text)
    if E.live_deviates(welt):
        st.caption(f"Das Live-Verdikt weicht in Welt {welt} vom Studien-Verdikt ab: eine einzelne CART ist schwächer als Wald und Boosting, und Kernel Ridge ist nicht die SVM.")

with st.spinner("Live-Instanz wird gerechnet (Kreuzvalidierung von Kernel Ridge und CART, unter etwa 2 Sekunden)"):
    live = _live_instance(welt, n, seed, skalierung)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Kernel Ridge (diese Instanz)", E.pct(live.err_krr), delta=f"{E.pp(live.err_krr - live.bayes)} gegen Bayes", delta_color="off")
m2.metric(f"CART, Blattgröße {live.leaf} (diese Instanz)", E.pct(live.err_cart), delta=f"{E.pp(live.err_cart - live.bayes)} gegen Bayes", delta_color="off")
m3.metric("CART voll gewachsen (diese Instanz)", E.pct(live.err_cart_full), delta=f"{live.leaves_full} Blätter", delta_color="off")
m4.metric("Bayes-Fehler (diese Testpunkte)", E.pct(live.bayes), delta="Boden", delta_color="off")
kind, text = E.verdict_live_instance(live, welt, n, seed)
getattr(st, kind)(text)
st.caption(
    "Die Live-Instanz zieht andere Daten als die Studie (andere Quelle und Reihenfolge, höchstens 600 Zeilen) und rechnet ein anderes Verfahren (Kernel Ridge gegen eine CART statt scikit-learn-SVM gegen Wald und Boosting): "
    "Zufalls-Seed und Zahlen der Instanz sind nicht die der Studie."
)

# Studie: Balken je Verfahren
rows = [("lineare SVM", FZ.haupt_array(welt, "lin").mean(), E.mean_se(FZ.haupt_array(welt, "lin"))[1], C.COLORS["linear"]),
        ("Gauß-SVM", comp.kern, E.mean_se(FZ.haupt_array(welt, "rbf"))[1], C.COLORS["kern"])]
k_rf, k_gb = FZ.TREE_KEYS[gegner]
rows.append((f"Wald ({gegner})", FZ.haupt_array(welt, k_rf).mean(), E.mean_se(FZ.haupt_array(welt, k_rf))[1], C.COLORS["wald"]))
rows.append((f"Boosting ({gegner})", FZ.haupt_array(welt, k_gb).mean(), E.mean_se(FZ.haupt_array(welt, k_gb))[1], C.COLORS["boost"]))
rows.append(("bester Baum je Zufalls-Seed", comp.baum, E.mean_se(FZ.best_tree(welt, gegner))[1], C.COLORS["baum"]))
if welt in FZ.teil("netz"):
    net = np.array(FZ.teil("netz")[welt]["net"], float)
    rows.append(("kleines Netz (abgestimmt)", net.mean(), E.mean_se(net)[1], C.COLORS["netz"]))
st.plotly_chart(V.fig_error_bars(rows, bayes=comp.bayes, title=f"Studie: Testfehler, Mittel über {comp.n} Zufalls-Seeds (Balken: Standardfehler)"), width="stretch", key="main_chart")
st.caption("Der Baum ist je Zufalls-Seed der bessere von Wald und Boosting, im Nachhinein am Test gewählt: optimistisch für die Bäume. Die Hyperparameter der Spalte „abgestimmt“ sind auf dem Training gewählt, die Auswahl zwischen Wald und Boosting aber am Test (sie kann in C2 und D ein Verdikt um eine Stufe verschieben); die am Test gewählte Gitterzelle ist die optimistische Untergrenze.")

# ---------------------------------------------------------------- Kernfragen: fünf Experimente
st.markdown("## ❓ Fünf Kernfragen")
st.markdown(TX.KERNFRAGEN_MD)

# --- Experiment 1
st.subheader("🔬 Experiment 1: Welches Verfahren liegt in dieser Welt vorn?")
kinds1 = []
if welt == "P":
    with st.spinner("Entscheidungsflächen werden gerechnet"):
        X2, y2, ext, kp, cp, f_sel, leaf_sel, leaves_sel = _boundary(n, seed, "P", 0)
    st.plotly_chart(V.fig_boundary(X2, y2, ext, {"krr": kp, "cart": cp}, {"krr": f"Kernel Ridge (Skalenfaktor f = {E.num(f_sel, 2)})", "cart": f"CART (Blattgröße {leaf_sel}, {leaves_sel} Blätter)"}, C.BOUNDARY_GRID),
                    width="stretch", key="boundary_p")
    st.caption("Live auf der 2D-Welt (Liefergebiet): glatte Kurve gegen Treppe. Entscheidungsflächen gibt es nur für Welten mit zwei Merkmalen; Experiment 3 zeigt sie für die Schwellenregel.")
else:
    st.caption("Entscheidungsflächen zeigt die App nur für Welten mit zwei Merkmalen (Liefergebiet in Experiment 1, Schwellenregel in Experiment 3).")
vl_rows = []
for w in C.LIVE_WORLDS:
    cs = E.compare_study(w, gegner)
    lv = E.live_vs_study(w, "cart")
    vl_rows.append({"Welt": w, "Bayes-Fehler": E.pct(cs.bayes), "Gauß-SVM": E.pct(cs.kern), "bester Baum (Studie)": E.pct(cs.baum), "Studie: Verdikt": f"{cs.verdict} ({cs.wins}/{cs.n})",
                    "Live: Kernel Ridge": E.pct(lv["krr"]), "Live: CART (Blattgröße per Kreuzvalidierung)": E.pct(lv["cart"]), "Live: Verdikt": f"{lv['verdict']} ({lv['wins']}/{lv['n']})"})
st.dataframe(pd.DataFrame(vl_rows), hide_index=True, width="stretch")
st.caption(f"Studie (Gegner {gegner}) und Live-Modell je Welt, Mittel über {C.N_SEEDS} Zufalls-Seeds; „k/{C.N_SEEDS}“ zählt die Zufalls-Seeds, in denen der Kern den kleineren Fehler hat. {TX.live_abweichung()}")
with st.expander("Lernkurve des abgestimmten Waldes gegen die Gauß-SVM (eingefroren)"):
    if welt in FZ.teil("lernkurve"):
        lc = E.learning_curve(welt)
        st.plotly_chart(V.fig_learning(lc, f"Welt {welt}: Fehler gegen Trainingszeilen ({lc['seeds']} Zufalls-Seeds)"), width="stretch", key="learning_chart")
        st.caption(TX.lernkurve_text(welt))
    else:
        st.info(f"Für Welt {welt} ist keine Lernkurve gemessen (gemessen: {', '.join(FZ.teil('lernkurve').keys() - {'n'})}).")
with st.expander("Das kleine Netz (eingefroren, eine verdeckte Schicht)"):
    st.markdown(TX.netz_text())

# --- Experiment 2
st.subheader("🔬 Experiment 2: Was ändert sich, wenn der Gegner abgestimmt ist?")
rows2 = []
for w in FZ.worlds_with_study():
    a = E.compare_study(w, "Standard")
    b = E.compare_study(w, "abgestimmt")
    c = E.compare_study(w, "am Test gewählt (Untergrenze)")
    rows2.append({"Welt": w, "Standard": f"{a.verdict} ({a.wins}/{a.n})", "abgestimmt": f"{b.verdict} ({b.wins}/{b.n})", "am Test gewählt (Untergrenze)": f"{c.verdict} ({c.wins}/{c.n})",
                  "Verschiebung zu den Bäumen": "ja" if E.STUFE[b.verdict] > E.STUFE[a.verdict] else "nein"})
st.dataframe(pd.DataFrame(rows2), hide_index=True, width="stretch")
mw, sw, hw = E.tuning_gain(welt, "rf")
mb, sb, hb = E.tuning_gain(welt, "gb")
kind, text = TX.verdict_tuning(welt, comp, mw, sw, hw, mb, sb, hb)
getattr(st, kind)(text)
st.caption(TX.tuning_frisch_text())

# --- Experiment 3
st.subheader("🔬 Experiment 3: Dieselbe Aufgabe, andere Koordinaten")
angles = tuple(sorted(set(ANGLES_CURVE) | {winkel}))
with st.spinner("Live-Paar für die gedrehten Fassungen wird gerechnet (unter etwa 2 Sekunden)"):
    runs = {a: _rotation_run(min(n, 400), seed, a) for a in angles}
    X2, y2, ext, kp, cp, f_sel, leaf_sel, leaves_sel = _boundary(min(n, 400), seed, "E2", winkel)
    inv = _gram_invariance(n, seed, winkel)
st.plotly_chart(V.fig_boundary(X2, y2, ext, {"krr": kp, "cart": cp}, {"krr": f"Kernel Ridge, Merkmale um {winkel}° gedreht", "cart": f"CART, Merkmale um {winkel}° gedreht ({leaves_sel} Blätter)"}, C.BOUNDARY_GRID),
                width="stretch", key="boundary_rot")
st.plotly_chart(V.fig_rotation(angles, {"Kernel Ridge": ([runs[a][0] for a in angles], C.COLORS["kern"]), "CART (Blattgröße per Kreuzvalidierung)": ([runs[a][1] for a in angles], C.COLORS["baum"]),
                                         "CART voll gewachsen": ([runs[a][2] for a in angles], C.COLORS["boost"])}, runs[angles[0]][4]), width="stretch", key="rotation_chart")
st.caption(f"Live: 2D-Schwellenregel (Welt E2), {min(n, 400)} Zeilen, eine Instanz. Voll gewachsene CART: {runs[angles[0]][3]} Blätter bei 0°, {runs[angles[-1]][3]} Blätter bei {angles[-1]}°. "
           f"Größter Unterschied der Gauß-Gram-Matrizen vor und nach der Drehung um {winkel}° (gleiches gamma, 200 Zeilen): {inv:.1e}.")
kind, text = E.verdict_rotation(E.rotation_summary("E", "Er"))
getattr(st, kind)(text)
st.caption(TX.rotation_b_text())

# --- Experiment 4
st.subheader("🔬 Experiment 4: Was den Abstand verdirbt, irrelevante Merkmale und Einheiten")
c4a, c4b = st.columns(2)
with c4a:
    st.markdown("**Irrelevante Merkmale** (live: 2D-Schwellenregel E2 plus Rauschmerkmale)")
    with st.spinner("Live-Instanz mit Rauschmerkmalen wird gerechnet"):
        e_k, e_t, share, bayes_n = _noise_run(min(n, 400), seed, rauschen)
    p_total = 2 + rauschen
    st.metric("Kernel Ridge", E.pct(e_k), delta=f"CART {E.pct(e_t)}; Bayes {E.pct(bayes_n)}", delta_color="off")
    st.markdown(f"Anteil der 2 relevanten von {p_total} Spalten am mittleren Abstandsquadrat (standardisiert, 200 Zeilen): **{E.pct(share)}**, erwartet k/p = 2/{p_total} = {E.pct(2 / p_total)}.")
with c4b:
    st.markdown("**Einheiten** (live: Lieferungen, Welt F)")
    with st.spinner("Kernel Ridge auf drei Skalen wird gerechnet (etwa 2 Sekunden)"):
        sc, shares, bayes_f = _scaling_run(min(n, 400), seed)
    st.dataframe(pd.DataFrame([{"Skala": m, "Kernel Ridge": E.pct(v[0]), "σ_med": E.num(v[1], 2)} for m, v in sc.items()]), hide_index=True, width="stretch")
    st.caption(f"Eine Instanz (Bayes {E.pct(bayes_f)}); das Verdikt über {C.N_SEEDS} Zufalls-Seeds steht unten.")
st.plotly_chart(V.fig_share(list(E.DELIVERY_COLUMNS) + [f"Rauschen {i + 1}" for i in range(3)], shares), width="stretch", key="share_chart")
st.caption("Anteil jeder Spalte am mittleren Abstandsquadrat der Lieferungen in natürlichen Einheiten (200 Zeilen dieser Instanz). Das Ladegewicht (Kilogramm) bestimmt den Abstand allein.")
kind, text = E.verdict_scaling(E.scaling_summary())
getattr(st, kind)(text)
kind, text = E.verdict_noise(E.noise_loss())
getattr(st, kind)(text)
st.caption(TX.noise_c_text())

# --- Experiment 5
st.subheader("🔬 Experiment 5: Welches Verfahren passt zu meiner Aufgabe?")
st.caption("Entscheidungsbaum ohne neue Rechnung: jede Antwort nennt ihre Zahl aus der Studie. Er beschreibt die gemessenen Welten, nicht Ihre Daten.")
antworten = {}
cols5 = st.columns(2)
for i, k in enumerate(DEC.FRAGEN_REIHENFOLGE):
    with cols5[i % 2]:
        antworten[k] = st.radio(DEC.FRAGEN[k], DEC.ANTWORTEN, index=2, horizontal=True, key=f"frage_{k}")
for aussage, beleg in DEC.empfehlung(antworten):
    st.markdown(f"- **{aussage}** {beleg}")
with st.expander("Weitere Demos der Reihe und Randsätze aus der Literatur"):
    st.markdown("**Die Reihe „Kernmethoden“:**\n" + "\n".join(f"- `{o}`: {t}" for o, t in DEC.STUECKE))
    st.markdown("**Randsätze (Literatur, nicht gemessen):**\n" + "\n".join(f"- {s}" for s in DEC.RANDSAETZE))

st.markdown("---")

# ---------------------------------------------------------------- Grenzen
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(TX.grenzen_md())
with st.expander("⚠️ Annahmen, Grenzen und bekannte Probleme"):
    st.markdown(TX.ANNAHMEN_MD)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(TX.FORMEL_MD)

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Kernmethoden: Skalarprodukte tauschen](https://sebastianhanisch.net/konzepte-kernmethoden.html)."
)
