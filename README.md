# Full-Quantum 5-Atom SQMG + Quantum A2C QRL

This repository implements a **quantum molecular generation** pipeline (SQMG/QMG) and a **quantum actor–critic RL scaffold** (QRL) using **Qiskit** and **RDKit**. The primary objective is to maximize:

**Validity × Uniqueness**

Validity is defined by RDKit sanitization; uniqueness is tracked via canonical SMILES.

## Classical BO vs Quantum RL
The reference PDF discusses classical Bayesian Optimization (BO) for parameter search. This repo **does not use BO** for training. Instead, it uses a **quantum A2C (actor–critic)** setup with PQC-based actor and critic to directly optimize **Validity × Uniqueness**.

## Representation (N = 5, full graph)
We use a fixed 5-site representation:

- **Atoms**: 5 categorical decisions (one per site)
- **Bonds**: 10 categorical decisions for the fully connected graph:
  `(0–1), (0–2), (0–3), (0–4), (1–2), (1–3), (1–4), (2–3), (2–4), (3–4)`

If either endpoint atom is `NONE`, the corresponding bond is masked to `NONE`.

## Fixed vocabularies and decode maps

**Atom vocabulary**
```python
ATOM_VOCAB = ["NONE", "C", "O", "N", "S", "P", "F", "Cl"]
```

**Bond vocabulary**
```python
BOND_VOCAB = ["NONE", "SINGLE", "DOUBLE", "TRIPLE"]
```

**Atom 3-bit decode**
- `000 -> NONE`
- `001 -> C`
- `010 -> O`
- `011 -> N`
- `100 -> S`
- `101 -> P`
- `110 -> F`
- `111 -> Cl`

**Bond 2-bit decode**
- `00 -> NONE`
- `01 -> SINGLE`
- `10 -> DOUBLE`
- `11 -> TRIPLE`

## SQMG circuit summary (PDF-inspired hybrid design)
SQMG follows a dynamic circuit design with **bond-qubit reuse**:

- **Atom blocks**: 5 independent 3-qubit PQC blocks (15 qubits total), measured to 3-bit atom codes.
- **Bond register**: 2 qubits reused across the 10 full-graph edges.
- **Dynamic steps** per bond: `reset → bond module → measure → reset`.

### In-circuit quantum-controlled masking (no classical feedforward)
The conditional bond behavior is implemented with quantum control (no `if_test`/`c_if`):

- Compute `none_i` and `none_j` flags for atom codes using reversible logic.
- Invert to `active_i`, `active_j`.
- Apply the **entire bond ansatz** as a double-controlled operation using `(active_i, active_j)` as controls.
- Uncompute ancillas back to `|0⟩`.

This matches the SQMG/QCNC spirit without classical control flow.

## Environment validity rules (RDKit)
`FiveAtomMolEnv.build_smiles_from_actions(atoms, bonds)`:

- If active atoms `< 2` → invalid.
- Add only non-`NONE` atoms.
- Add a bond only when both endpoints exist and bond type is not `NONE`.
- Deterministic bond repair: if a bond exceeds valence, it is downgraded
  (TRIPLE→DOUBLE→SINGLE→NONE) until valid.
- Sanitize + canonicalize SMILES.
- Reject disconnected fragments if SMILES contains `"."`.

### repair_bonds option + raw metrics
`FiveAtomMolEnv(repair_bonds=True)` enables deterministic bond repair for valence.
When `repair_bonds=False`, no downgrading is applied.

Both raw and repaired PDF metrics are exposed:
- `validity_raw_pdf`, `uniqueness_raw_pdf`, `reward_raw_pdf`
- `validity_pdf`, `uniqueness_pdf`, `reward_pdf`

## Quantum A2C (QRL)
The A2C loop uses **PQC actor/critic** models (no classical BO), trained with SPSA schedules. The reward is step-local and PDF-aligned:

`score_pdf_step = (valid_count / samples) * (unique_valid_in_batch / max(valid_count, 1))`

with a mild repeat penalty and a small novelty bonus for exploration. A short reward window smooths variance.
`k_batches` controls reward smoothing (larger values reduce noise).

- **Actor**: PQC outputs a mean vector `mu` for a Gaussian policy (action_dim=16).
- **Critic**: PQC outputs a scalar value `V(s)` mapped to `[0, 1]`.
- **Optimization**: SPSA updates for both actor and critic.
- **Action-to-QMG**: a fixed random projection maps action → parameter delta for QMG.

### Optimized composite score
The optimized PDF-style composite is computed per batch:

`composite = (valid_count / samples) * (unique_valid_count / max(valid_count, 1))`

which simplifies to `unique_valid_count / samples`.

The reward plotted by the one-command trainer is:

`reward_step = validity_step * uniqueness_step`  (each in [0, 1]).

### State features (A2C)
The state vector includes:

- `valid_ratio`, `unique_ratio`, `target_metric`
- normalized counts (`valid_count/samples`, `unique_valid_count/samples`)
- **novelty feature**: `log_unique = log1p(unique_valid_count) / log1p(samples)`
- QMG weight statistics (`mean`, `std`, `L2`, `min`, `max`)

## Installation (CPU / GPU)
**Important**: Do not mix Qiskit major versions. Use the pinned requirements.

CPU-only (Qiskit 0.46.x + Aer 0.13.x):
```bash
pip install -r requirements-cpu.txt
pip install -r requirements-dev.txt
```

GPU (Aer GPU):
```bash
pip install -r requirements-gpu.txt
pip install -r requirements-dev.txt
```

Optional analysis:
```bash
pip install \"pandas>=2.3.3\"
```

## Quickstart
```bash
# run tests
python -m pytest -q

# sample molecules using SQMG
python -m scripts.sample_qmg --mode sqmg --n 2000

# train with quantum A2C (K-batch averaging)
python -m scripts.train_qmg_qrl --algo a2c --steps 1000 --batch-size 256 --k-batches 2 --eval-every 100 --eval-batch-size 2000
```

## DGX Quickstart (one-command training)
The one-command entrypoint trains QMG+QRL end-to-end and logs the **PDF-style reward**
`validity_step * uniqueness_step` in the range `[0, 1]`, saving `metrics.csv` and
`reward.png` under the output directory. GPU is auto-detected if available; otherwise
the script falls back to CPU.

```bash
python -m scripts.run_one_train --episodes 300 --batch-size 256 --device auto --out runs/dgx_run
```

## Training presets
Smoke test (fast):
```bash
python -m scripts.run_one_train --episodes 20 --batch-size 64 --device cpu --out /tmp/qmg_qrl_smoke
```

Best-effort long run:
```bash
python -m scripts.run_one_train --episodes 2000 --batch-size 256 --device auto --out runs/qmg_qrl_long --k-batches 2
```

## Environment cleanup (SSH)
```bash
rm -rf ~/Full-Quantum-5Atom-QAC
conda deactivate
conda env remove -n qrl_fq_5atom_gpu -y
conda create -n qrl_fq_5atom_gpu python=3.10 -y
conda activate qrl_fq_5atom_gpu
git clone git@github.com:SiriusW823/Full-Quantum-5Atom-QAC.git
cd Full-Quantum-5Atom-QAC
pip install -r requirements-gpu.txt
```

## Notes
You may see Qiskit deprecation warnings during tests; these are from upstream APIs and do not affect correctness.

## License
See LICENSE.
