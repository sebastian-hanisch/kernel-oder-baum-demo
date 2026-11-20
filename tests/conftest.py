"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


@pytest.fixture
def xor4():
    """XOR mit vier Punkten (Handbeispiel 1): Klasse 1 genau dann, wenn x1 * x2 > 0."""
    X = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    y = np.array([1, 0, 0, 1])
    return X, y


@pytest.fixture
def sechs_punkte():
    """Sechs Punkte mit achsenparalleler Regel x1 > 0 (Handbeispiel 2): Klasse 0 für x1 < 0, Klasse 1 für x1 > 0; ein Schnitt, 2 Blätter."""
    X = np.array([(-1, -2), (-1, 3), (-1, 0), (1, -3), (2, 3), (1, 1)], float)
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y
