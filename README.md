# Full-Quantum 5-Atom SQMG + QRL Scaffold

This repository implements a **quantum molecular generation** pipeline (SQMG/QMG) and a **quantum-assisted RL scaffold** (QRL) using **Qiskit** and **RDKit**. The primary objective is to maximize:

**Validity × Uniqueness**

where validity is defined by RDKit sanitization and uniqueness is tracked via canonical SMILES.

## Representation (N = 5 chain)
We use a fixed 5-site chain:

- **Atoms**: 5 categorical decisions (one per site)
- **Bonds**: 4 categorical decisions for a chain topology  
  `(0–1), (1–2), (2–3), (3–4)`

## Fixed vocabularies and decode maps

**Atom vocabulary**
```python
ATOM_VOCAB = ["NONE", "C", "N", "O"]
```

**Bond vocabulary**
```python
BOND_VOCAB = ["NONE", "SINGLE", "DOUBLE", "TRIPLE"]
```

**Atom 3-bit decode**
- `000 -> NONE`
- `001 -> C`
- `010 -> N`
- `011 -> O`
- `100..111 -> NONE`

**Bond 2-bit decode**
- `00 -> NONE`
- `01 -> SINGLE`
- `10 -> DOUBLE`
- `11 -> TRIPLE`

## SQMG circuit summary (PDF-inspired hybrid design)
SQMG follows a dynamic circuit design with **bond-qubit reuse**:

- **Atom blocks**: 5 independent 3-qubit PQC blocks (15 qubits total), measured to 3-bit atom codes.
- **Bond register**: 2 qubits reused across the 4 chain bonds.
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
- Sanitize + canonicalize SMILES.
- Reject disconnected fragments if SMILES contains `"."`.

## Quickstart
```bash
# run tests
python -m pytest -q

# sample molecules using SQMG
python -m scripts.sample_qmg --mode sqmg --n 2000

# train with quantum A2C
python -m scripts.train_qmg_qrl --algo a2c --steps 200 --batch-size 256
```

## Notes
You may see Qiskit deprecation warnings during tests; these are from upstream APIs and do not affect correctness.

## License
See LICENSE.
