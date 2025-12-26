from __future__ import annotations

import sys


def test_sample_qmg_cli_repair_bonds_smoke(monkeypatch, capsys) -> None:
    import scripts.sample_qmg as sample_qmg

    argv = [
        "sample_qmg",
        "--mode",
        "sqmg",
        "--backend",
        "qiskit",
        "--n",
        "5",
        "--atom-layers",
        "1",
        "--bond-layers",
        "1",
        "--repair-bonds",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    sample_qmg.main()
    out = capsys.readouterr().out
    assert "repair_bonds=True" in out