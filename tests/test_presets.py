"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Presets nennen die Welt, nie ein Verdikt; je Preset die erwartete Verdikt-Zeile."""

import re

import kob_constants as C
import kob_evaluation as E
import kob_presets as P

# erwartetes Studien-Verdikt je Welt (Gegner abgestimmt, erste Reihe): eine Messung dieser Demo, mit Zählung in test_claims.py belegt
ERWARTET = {"P": "Kern vorn", "E": "Baum vorn", "Er": "Kern vorn", "B": "Baum vorn", "D": "gleichauf", "F": "Baum vorn", "F8": "Baum vorn"}
VERDIKT_WOERTER = ("vorn", "gleichauf", "schlägt", "gewinnt", "verliert")


def test_jedes_preset_hat_hilfe_und_alle_schluessel():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_werte_liegen_in_den_reglergrenzen():
    for name, preset in C.PRESETS.items():
        assert preset["welt"] in C.LIVE_WORLDS and preset["gegner"] in C.GEGNER and preset["skalierung"] in C.SCALINGS, name
        assert C.N_MIN <= preset["n"] <= C.N_MAX and (preset["n"] - C.N_MIN) % C.N_STEP == 0, name
        assert 0 <= preset["seed"] <= C.SEED_MAX
        assert C.ANGLE_MIN <= preset["winkel"] <= C.ANGLE_MAX and (preset["winkel"] - C.ANGLE_MIN) % C.ANGLE_STEP == 0
        assert C.NOISE_MIN <= preset["rauschen"] <= C.NOISE_MAX


def test_standard_preset_entspricht_den_standardeinstellungen():
    name = next(iter(C.PRESETS))
    for key, state_key in P.PRESET_KEYS.items():
        assert C.PRESETS[name][key] == P.SETTING_SPECS[state_key].default, key


def test_setting_specs_decken_jeden_preset_schluessel_ab():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_presets_nennen_die_welt_nie_ein_verdikt():
    """Das Live-Modell gibt die Studien-Verdikte nur teilweise wieder: Namen und Hilfetexte enthalten kein Urteilswort."""
    for name in C.PRESETS:
        text = (name + " " + C.PRESET_HELP[name]).lower()
        assert not any(w in text for w in VERDIKT_WOERTER), name


def test_jedes_preset_hat_eine_andere_welt_oder_einstellung():
    keys = [tuple(sorted(p.items())) for p in C.PRESETS.values()]
    assert len(set(keys)) == len(keys)
    assert {p["welt"] for p in C.PRESETS.values()} == set(C.LIVE_WORLDS)


def test_je_preset_die_erwartete_verdikt_zeile():
    """Jede Welt der Presets: das Studien-Verdikt (Gegner abgestimmt) ist das erwartete, und der Text der App nennt es samt Zählung; ein Preset ohne Effekt (gleichauf) meldet keinen Effekt."""
    for name, preset in C.PRESETS.items():
        comp = E.compare_study(preset["welt"], "abgestimmt")
        assert comp.verdict == ERWARTET[preset["welt"]], (name, comp)
        kind, text = E.verdict_study(comp)
        assert f"**{comp.verdict}**" in text and f"{comp.wins} von {comp.n}" in text
        assert (kind == "info") == (comp.verdict == C.GLEICHAUF)
        if comp.verdict == C.GLEICHAUF:
            assert abs(comp.d_mean) <= C.SE_FACTOR * comp.d_se


def test_clamp_begrenzt_zahlen_und_waehlt_bei_unbekannter_auswahl_den_standard():
    spec = P.SETTING_SPECS["n_slider"]
    assert P.clamp(spec, 5) == C.N_MIN and P.clamp(spec, 10 ** 6) == C.N_MAX and P.clamp(spec, 300) == 300
    sel = P.SETTING_SPECS["welt_select"]
    assert P.clamp(sel, "E") == "E" and P.clamp(sel, "Z") == C.DEFAULT_WORLD
    assert P.clamp(P.SETTING_SPECS["gegner_radio"], "böse") == C.DEFAULT_GEGNER


def test_permalink_parameter_sind_eindeutig_und_ohne_leerzeichen():
    params = [s.url_param for s in P.SETTING_SPECS.values()]
    assert len(set(params)) == len(params) and all(re.fullmatch(r"[a-z]+", p) for p in params)
