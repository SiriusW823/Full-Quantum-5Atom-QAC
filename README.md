# Full-Quantum 5-Atom QAC (SQMG + QRL Scaffold)

This repository implements a **quantum molecular generation** pipeline (QMG) and a **quantum-assisted learning scaffold** (QRL scaffold) built on **Qiskit** and **RDKit**.

The current emphasis is the **SQMG mode**: a **hybrid, PDF-inspired** design using **dynamic circuits** (mid-circuit measurement + reset) and **bond-qubit reuse** to sample molecules under a fixed **5-site** topology. The primary optimization objective throughout the project is:

> **Validity × Uniqueness**

where validity and uniqueness are defined via RDKit sanitization and canonical SMILES tracking.

---

## Key Concepts

### 1) Fixed 5-site molecular representation
We represent a molecule as:
- **Atoms**: 5 categorical decisions (one per site)
- **Bonds**: 4 categorical decisions for a **chain** topology:
  - `(0–1), (1–2), (2–3), (3–4)`

This matches the intended SQMG/QCNC “chain” spirit (as opposed to full-graph bonds).

### 2) Vocabularies and decode maps (fixed)
Atoms and bonds are decoded from bitstrings using the following fixed mappings:

**Atom vocabulary**
```python
ATOM_VOCAB = ["NONE", "C", "N", "O"]
Bond vocabulary

python
複製程式碼
BOND_VOCAB = ["NONE", "SINGLE", "DOUBLE", "TRIPLE"]
Atom 3-bit decode mapping

000 -> NONE

001 -> C

010 -> N

011 -> O

100..111 -> NONE (reserved → NONE)

Bond 2-bit decode mapping

00 -> NONE

01 -> SINGLE

10 -> DOUBLE

11 -> TRIPLE

3) SQMG mode: dynamic circuit + bond reuse + in-circuit masking
In SQMG mode, the quantum circuit follows a hybrid structure:

Atom blocks: 5 independent PQC blocks

3 qubits per atom site → 15 qubits total

each block outputs a 3-bit atom code via measurement

Bond register: 2 qubits reused across all 4 chain bonds

for each bond, we reset → apply bond module → measure → reset

Ancillas (optional, small): a small number (e.g., 2) may be used to implement masking inside the circuit, and must be uncomputed back to |0⟩ before moving to the next bond to avoid cross-bond contamination.

In-circuit masking (core requirement)
The SQMG/QCNC “conditional bond module” behavior is implemented without classical feedforward (no if_test, no c_if). Instead, a quantum-controlled mask is constructed:

Compute none_i = (atom_i == 000) and none_j = (atom_j == 000) with reversible logic (using ancillas).

Convert to active flags:

active_i = NOT(none_i)

active_j = NOT(none_j)

Apply the entire bond ansatz (rotation + entangling gates on bond qubits) as a double-controlled operation, controlled on (active_i, active_j).

Uncompute ancillas back to |0⟩.

As a result, the bond module is only effective when both endpoints are non-NONE (as in the PDF spirit), but accomplished with quantum control, not classical control flow.

Repository Layout
env/
RDKit environment for building molecules from (atoms, bonds), computing metrics, and enforcing validity rules.

qmg/
Quantum Molecular Generators

SQMG circuits and generator (dynamic circuits, bond reuse, in-circuit masking)

optional factorized/baseline generator utilities (if present)

qrl/
QRL scaffold / helper components (kept for later integration; not the primary focus of SQMG wiring).

scripts/
Entry points:

sampling (sample_qmg.py)

optional training scripts (if present in repo)

tests/
Pytest smoke tests verifying shape contracts and non-crashing execution.

Environment and Metrics (RDKit)
FiveAtomMolEnv.build_smiles_from_actions(atoms, bonds)
Inputs:

atoms: length 5, values index into ATOM_VOCAB

bonds: length 4, values index into BOND_VOCAB, aligned with chain edges

Validity rules:

active_atoms < 2 → invalid

Add only non-NONE atoms to RDKit

Add a bond only if endpoints exist and bond type is not NONE

RDKit sanitize + canonicalize

Reject disconnected fragments if canonical SMILES contains "."

Metrics
valid_ratio

unique_ratio

target_metric = valid_ratio × unique_ratio

Installation
Recommended environment: WSL + conda

bash
複製程式碼
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qrl_fq_5atom
pip install -r requirements.txt
pip install -r requirements-dev.txt
Quickstart
bash
複製程式碼
# run tests
python -m pytest -q

# sample molecules using SQMG
python -m scripts.sample_qmg --mode sqmg --n 2000
License
See LICENSE.
