## Quantum 5-Site Molecule Generator + Quantum RL Helper (Qiskit)

This repo contains a quantum sampling + training stack:
- **env/**: classical RDKit environment for 5 sites with optional `NONE` atoms and **chain** bonds (4 edges), with validity/uniqueness tracking and `target_metric = valid_ratio * unique_ratio`.
- **qmg/**: Quantum Molecular Generators
  - `generator.py`: factorized-head PQC (small, uses Qiskit Sampler) for atoms + bonds.
  - `sqmg_generator.py`: SQMG/QCNC-inspired **hybrid** QMG (PDF-aligned spirit) with **3N+2 data qubits** (N=5 → 17), bond reuse (2 qubits reused), and an in-circuit quantum-controlled mask (uses 2 ancillas).
- **qrl/**: Quantum RL helper — Qiskit PQC that scores SMILES in [0,1] (novelty prior / critic) and can be trained via SPSA.
- **scripts/**: sampling and joint training entrypoints.

### Version 2 SQMG (PDF-aligned)

**Hybrid circuit structure (N=5):**
- **Atom registers:** 5 independent blocks × 3 qubits = 15 qubits.
- **Bond register:** 2 qubits, **reused** sequentially across chain edges.
- **Ancillas:** 2 qubits, computed/uncomputed per bond for quantum-controlled masking.
- **Total:** 17 data qubits (3N+2) + 2 ancillas = 19 qubits in the circuit.

**Chain bonds:**
- We generate bond codes for the 5-site chain:
  `EDGE_LIST = [(0,1),(1,2),(2,3),(3,4)]`
- Generator output uses:
  - `atoms`: length 5
  - `bonds`: length 4 aligned with `EDGE_LIST`

**No classical feedforward**
- No `if_test`/`c_if` are used. The PDF conditional bond behavior is implemented **inside the circuit**
  via quantum-controlled masking: the bond ansatz is applied only when both endpoints decode to a
  non-`NONE` atom.

### Key modules

- `env/mol_env_5atom.py`
  - Vocab:
    - `ATOM_VOCAB = ["NONE", "C", "N", "O"]`
    - `BOND_VOCAB = ["NONE", "SINGLE", "DOUBLE", "TRIPLE"]`
  - `FiveAtomMolEnv.build_smiles_from_actions(atoms, bonds)`:
    - expects `atoms` length 5 and `bonds` length 4 aligned with `EDGE_LIST`
    - builds an RDKit `RWMol` using only non-`NONE` sites and only non-`NONE` chain bonds between existing endpoints
    - sanitizes, canonicalizes, rejects fragments (`"."` in SMILES) when `enforce_single_fragment=True`
    - tracks `samples`, `valid_count`, `unique_valid_count` and exposes `valid_ratio`, `unique_ratio`, `target_metric`, `stats()`
- `qmg/sqmg_circuit.py`
  - Builds the hybrid circuit and measures:
    - atoms: 5×3 bits (15)
    - bonds: 4×2 bits (8)
    - total: 23 classical bits (ancillas are not measured)
  - Bond module is quantum-controlled by ancillas (no classical conditionals).
- `qmg/sqmg_generator.py`
  - Runs `AerSimulator(...).run(circuit, shots=batch_size, memory=True)` and decodes per-shot into:
    - `atom_ids` (len=5) using a fixed 3-bit mapping
    - `bond_ids` (len=4) using a fixed 2-bit mapping
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
