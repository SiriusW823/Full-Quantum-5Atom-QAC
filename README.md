## Quantum 5-Atom Generator + Quantum RL Helper (Qiskit)

This repo now contains a fully quantum sampling stack:
- **env/**: classical RDKit environment for an exact 5-heavy-atom chain (5 atoms, 4 bonds) with validity/uniqueness tracking and `target_metric = valid_ratio * unique_ratio`.
- **qmg/**: Quantum Molecular Generator — factorized Qiskit PQC (Sampler) that samples 5 atoms + 4 bonds. Trainable weights exposed for SPSA updates.
- **qrl/**: Quantum RL helper — Qiskit PQC that scores SMILES in [0,1] (novelty prior / critic) and can be trained via SPSA.
- **scripts/**: sampling and joint training entrypoints.

### Key modules
- `env/mol_env_5atom.py`  
  - Vocab: ATOM_VOCAB = [C, N, O, S, F]; BOND_VOCAB = [NONE, SINGLE, DOUBLE, TRIPLE].  
  - `FiveAtomMolEnv.build_smiles_from_actions(atoms, bonds)` builds/validates a 5-atom chain, tracks samples/valid/unique, and exposes `valid_ratio`, `unique_ratio`, `target_metric`, `stats()`.
- `qmg/generator.py`  
  - `QiskitQMGGenerator.sample_actions(batch_size)` → atoms, bonds, smiles, valids, uniques.  
  - Trainable weights via `get_weights()`, `set_weights()`. Uses Qiskit Sampler with measured PQC heads per atom/bond.
- `qrl/helper.py`  
  - `QiskitQRLHelper.score(smiles_list)` → scores in [0,1].  
  - `train_step(smiles_list, targets, lr, spsa_eps)` → SPSA update to fit novelty targets.  
  - Features from `qrl/features.py` (atom/bond counts + ring/aromatic; clipped to [0,1]).
- `scripts/sample_qmg.py`  
  - Draws N samples, prints samples/valid/unique ratios and top SMILES.
- `scripts/train_qmg_qrl.py`  
  - Alternates SPSA updates between QMG (maximize reward with uniqueness/QRL prior) and QRL (fit novelty targets).  
  - Rewards: invalid=-0.2; valid+duplicate=-0.05; valid+unique=1.0+α·qrl_score (α default 0.5).

### Commands
```bash
# sample from QMG
python -m scripts.sample_qmg --n 2000

# joint QMG+QRL training (SPSA)
python -m scripts.train_qmg_qrl --steps 2000 --batch-size 64

# tests (after installing pytest)
python -m pytest -q
```

### Install
```bash
python -m venv .venv
.\.venv\Scripts\activate   # Windows
pip install -r requirements.txt
# optional dev extras
pip install -r requirements-dev.txt
```

### Notes
- Qiskit Sampler is used for both generator and helper; circuits are small (4 qubits) and measured.  
- Exploration against mode collapse: uniqueness-aware rewards, QRL novelty prior, SPSA noise annealing in the trainer.
