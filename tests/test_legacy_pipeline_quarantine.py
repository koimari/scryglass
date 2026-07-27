from __future__ import annotations

import argparse

import pytest

from lol_kills.enrich_games import first_inhib_side
from lol_kills.ml.train import train_all
from lol_kills.pipeline import cmd_train


def test_final_inhibitor_counts_never_become_first_inhibitor_order() -> None:
    assert first_inhib_side(3, 0) is None
    assert first_inhib_side(0, 2) is None
    assert first_inhib_side(3, 2) is None


def test_legacy_train_entry_points_are_quarantined_before_writes() -> None:
    with pytest.raises(RuntimeError, match="quarantined"):
        train_all()
    with pytest.raises(RuntimeError, match="quarantined"):
        cmd_train(argparse.Namespace(no_archive=True))
