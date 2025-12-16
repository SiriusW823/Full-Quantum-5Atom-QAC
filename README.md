## Quantum 5-Site Molecule Generator + Quantum RL Helper (Qiskit)

This repo contains a quantum sampling + training stack:
- **env/**: classical RDKit environment for 5 sites with optional `NONE` atoms and **full-graph** bonds (10 edges), with validity/uniqueness tracking and `target_metric = valid_ratio * unique_ratio`.
- **qmg/**: Quantum Molecular Generators
  - `generator.py`: factorized-head PQC (small, uses Qiskit Sampler) for atoms + bonds.
  - `sqmg_generator.py`: SQMG/QCNC-inspired **hybrid** QMG (PDF-aligned spirit) with **3N+2 qubits** (N=5 → 17), bond reuse (2 qubits reused), and full-graph bonds.
- **qrl/**: Quantum RL helper — Qiskit PQC that scores SMILES in [0,1] (novelty prior / critic) and can be trained via SPSA.
- **scripts/**: sampling and joint training entrypoints.

### Version 2 SQMG (PDF-aligned)

**Hybrid circuit structure (N=5):**
- **Atom registers:** 5 independent blocks × 3 qubits = 15 qubits.
- **Bond register:** 2 qubits, **reused** sequentially across all edges.
- **Total:** **17 qubits** (3N+2).

**Full-graph bonds:**
- We generate bond codes for all unordered pairs `(i<j)` using a fixed order:
  `EDGE_LIST = [(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]`
- Generator output uses:
  - `atoms`: length 5
  - `bonds`: length 10 aligned with `EDGE_LIST`

**Why no in-circuit conditionals?**
- Aer can fail when compiling conditional instructions with multiple classical registers.
- This repo implements the PDF “only create a bond if both endpoint atoms exist” behavior via **decode-time masking**:
  - If `atom_i == NONE` or `atom_j == NONE`, force the decoded bond for edge `(i,j)` to `NONE`.

### Key modules

- `env/mol_env_5atom.py`
  - Vocab:
    - `ATOM_VOCAB = ["NONE", "C", "O", "N"]`
    - `BOND_VOCAB = ["NONE", "SINGLE", "DOUBLE", "TRIPLE"]`
  - `FiveAtomMolEnv.build_smiles_from_actions(atoms, bonds)`:
    - expects `atoms` length 5 and `bonds` length 10 aligned with `EDGE_LIST`
    - builds an RDKit `RWMol` using only non-`NONE` sites and only non-`NONE` bonds between existing endpoints
    - sanitizes, canonicalizes, rejects fragments (`"."` in SMILES) when `enforce_single_fragment=True`
    - tracks `samples`, `valid_count`, `unique_valid_count` and exposes `valid_ratio`, `unique_ratio`, `target_metric`, `stats()`
- `qmg/sqmg_circuit.py`
  - Builds the 17-qubit hybrid circuit and measures:
    - atoms: 5×3 bits (15)
    - bonds: 10×2 bits (20)
    - total: 35 classical bits
  - No `if_test`/`c_if` blocks are used.
- `qmg/sqmg_generator.py`
  - Runs `AerSimulator(...).run(circuit, shots=batch_size, memory=True)` and decodes per-shot into:
    - `atom_ids` (len=5) using a fixed 3-bit mapping
    - `bond_ids` (len=10) using a fixed 2-bit mapping
  - Applies decode-time masking for `NONE` endpoints.
- `scripts/sample_qmg.py`
  - Samples N molecules and prints:
    `samples, valid_count, unique_valid_count, valid_ratio, unique_ratio, target_metric`,
    plus up to 10 unique SMILES.
- `scripts/train_qmg_qrl.py`
  - Alternates SPSA updates between QMG (maximize reward with uniqueness/QRL prior) and QRL (fit novelty targets).

### Commands
```bash
# sample from QMG (SQMG default)
python -m scripts.sample_qmg --n 2000 --mode sqmg

# baseline factorized generator
python -m scripts.sample_qmg --n 2000 --mode factorized

# joint QMG+QRL training (SPSA)
python -m scripts.train_qmg_qrl --steps 2000 --batch-size 64

# tests
python -m pytest -q
```

### Install
```bash
python -m venv .venv
.\.venv\Scripts\activate   # Windows
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Notes
- `sqmg_generator.py` uses `AerSimulator` with `memory=True` to decode hybrid measurements; the factorized QMG and QRL helper use the Aer Sampler primitive.
- Because `NONE` atoms and `NONE` bonds are allowed, early sampling can have low `valid_ratio`. The intended learning signal comes from reward shaping + uniqueness pressure during training.

