from pathlib import Path

from scripts.run_one_train import run_one_train


def test_run_one_train_smoke(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    rows = run_one_train(
        episodes=2,
        batch_size=16,
        out_dir=out_dir,
        backend="qiskit",
        device="cpu",
        atom_layers=1,
        bond_layers=1,
        eval_every=1,
        eval_shots=16,
        log_every=1,
    )

    metrics_path = out_dir / "metrics.csv"
    plot_path = out_dir / "reward.png"
    eval_path = out_dir / "eval.csv"
    reward_eval = out_dir / "reward_eval.png"
    validity_eval = out_dir / "validity_eval.png"
    uniqueness_eval = out_dir / "uniqueness_eval.png"

    assert metrics_path.exists()
    assert plot_path.exists()
    assert eval_path.exists()
    assert reward_eval.exists()
    assert validity_eval.exists()
    assert uniqueness_eval.exists()
    assert rows
    for row in rows:
        reward_step = float(row["reward_step"])
        assert 0.0 <= reward_step <= 1.0
