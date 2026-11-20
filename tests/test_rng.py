"""Zufallsquelle: `kob_rng.py` ist eine byteweise Kopie der Vorlage (Kopier-Test) und liefert die bekannte SplitMix64-Folge; die Ziehungen der Welten hängen nur an dieser Quelle."""

from pathlib import Path

import numpy as np
import pytest

import kob_rng
import kob_worlds as W

ROOT = Path(__file__).resolve().parent.parent
VORLAGE = ROOT.parent / "sebastianhanisch-website" / "tools" / "templates" / "rng_splitmix64_exact.py"


def test_kob_rng_ist_eine_kopie_der_vorlage():
    if not VORLAGE.exists():
        pytest.skip("Vorlage rng_splitmix64_exact.py liegt nicht neben der Demo (nur im Arbeitsordner vorhanden)")
    assert (ROOT / "kob_rng.py").read_bytes() == VORLAGE.read_bytes()


def test_splitmix64_bekannte_folge_zu_seed_0():
    """Die ersten drei Ausgaben von SplitMix64 mit Zustand 0 (veröffentlichte Referenzwerte)."""
    r = kob_rng.SplitMix64(0)
    assert [r.next_u64() for _ in range(3)] == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F]


def test_gleichverteilte_zahlen_liegen_in_null_bis_eins_und_haben_das_richtige_mittel():
    r = kob_rng.SplitMix64(5)
    u = np.array([r.random() for _ in range(20000)])
    assert u.min() >= 0.0 and u.max() < 1.0
    se = u.std(ddof=1) / np.sqrt(len(u))
    assert abs(u.mean() - 0.5) < 4.5 * se


def test_uniform_array_ist_bitgleich_zu_einzelnen_random_aufrufen():
    a = W.uniform_array(kob_rng.SplitMix64(11), (7, 5))
    r = kob_rng.SplitMix64(11)
    b = np.array([[r.random() for _ in range(5)] for _ in range(7)])
    assert np.array_equal(a, b)


def test_normalzahlen_aus_random_und_inv_cdf_haben_mittel_null_und_streuung_eins():
    z = W.normal_array(kob_rng.SplitMix64(3), (40000,))
    se = 1.0 / np.sqrt(len(z))
    assert abs(z.mean()) < 4.5 * se
    assert abs(z.std(ddof=1) - 1.0) < 4.5 * (1.0 / np.sqrt(2 * len(z)))


def test_gleicher_zufalls_seed_gibt_gleiche_folge_anderer_gibt_andere():
    a = W.normal_array(kob_rng.SplitMix64(1), (50,))
    assert np.array_equal(a, W.normal_array(kob_rng.SplitMix64(1), (50,)))
    assert not np.array_equal(a, W.normal_array(kob_rng.SplitMix64(2), (50,)))


def test_kein_numpy_zufall_in_den_app_modulen():
    """Beschluss: SplitMix64 einheitlich, kein numpy.random in der App-Quelle (die Tests und das Werkzeug dürfen es nicht brauchen, weder default_rng noch np.random)."""
    for name in ("app.py", "kob_worlds.py", "kob_models.py", "kob_evaluation.py", "kob_decision.py", "kob_frozen.py", "kob_visualization.py"):
        src = (ROOT / name).read_text(encoding="utf-8")
        assert "default_rng" not in src and "np.random" not in src and "numpy.random" not in src, name
