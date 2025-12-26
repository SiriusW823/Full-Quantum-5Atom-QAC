from pathlib import Path

from scripts.run_one_train import run_one_train


def test_best_tracking_smoke(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    run_one_train(
        episodes=2,
        batch_size=16,
        out_dir=out_dir,
        warm_start_repair=1,
        adaptive_exploration=True,
        adapt_threshold=1.0,
        adapt_window=1,
        sigma_max=0.5,
        device="cpu",
        atom_layers=1,
        bond_layers=1,
        eval_every=1,
        eval_shots=16,
        track_best=True,
        repair_bonds=False,
        log_every=1,
    )

    eval_path = out_dir / "eval.csv"
    assert eval_path.exists()
    content = eval_path.read_text().strip().splitlines()
    assert content
    header = content[0].split(",")
    assert "phase" in header
    assert "reward_pdf_eval" in header
    assert "reward_raw_pdf_eval" in header
    assert "validity_pdf_eval" in header
    assert "uniqueness_pdf_eval" in header
    assert "sigma_max" in header
    assert "k_batches" in header
    assert "patience" in header

    if len(content) >= 3:
        row_repair = dict(zip(header, content[1].split(","), strict=False))
        row_strict = dict(zip(header, content[2].split(","), strict=False))
        assert row_repair.get("phase") == "repair"
        assert row_strict.get("phase") == "strict"
        assert float(row_strict["sigma_max"]) > 0.5
