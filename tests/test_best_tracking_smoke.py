from pathlib import Path

from scripts.run_one_train import run_one_train


def test_best_tracking_smoke(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    run_one_train(
        episodes=2,
        batch_size=16,
        out_dir=out_dir,
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
    assert "reward_pdf_eval" in header
    assert "reward_raw_pdf_eval" in header
    assert "validity_pdf_eval" in header
    assert "uniqueness_pdf_eval" in header
