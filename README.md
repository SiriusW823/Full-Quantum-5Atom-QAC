# Full-Quantum-5Atom-QAC

## Overview
Full-Quantum-5Atom-QAC is a reference implementation of Scalable Quantum Molecular Generation (SQMG) combined with a Quantum Actor-Critic (QRL) training loop. The goal is to generate chemically valid 5-atom molecules using fully quantum circuits and to optimize the composite objective:

Validity x Uniqueness

Validity is determined by RDKit sanitization. Uniqueness is computed over valid molecules only.

## Representation and circuit design

### Atom vocabulary (3-bit)
```python
ATOM_VOCAB = ["NONE", "C", "O", "N", "S", "P", "F", "Cl"]
```

Atom decode mapping:
- 000 -> NONE
- 001 -> C
- 010 -> O
- 011 -> N
- 100 -> S
- 101 -> P
- 110 -> F
- 111 -> Cl

### Bond vocabulary (2-bit)
```python
BOND_VOCAB = ["NONE", "SINGLE", "DOUBLE", "TRIPLE"]
```

Bond decode mapping:
- 00 -> NONE
- 01 -> SINGLE
- 10 -> DOUBLE
- 11 -> TRIPLE

### Full-graph bonds (N=5)
We use a complete graph with 10 edges:
```python
EDGE_LIST = [
  (0, 1), (0, 2), (0, 3), (0, 4),
  (1, 2), (1, 3), (1, 4),
  (2, 3), (2, 4),
  (3, 4),
]
```

### SQMG hybrid circuit (PDF-aligned)
For N=5 we use:
- 5 atom blocks x 3 qubits = 15 qubits
- 1 bond register of 2 qubits, dynamically reused per edge
- 2 ancilla qubits for in-circuit masking

For each edge, the circuit:
1) Computes NONE flags for the two endpoint atoms using reversible logic
2) Applies the bond ansatz only if both endpoints are non-NONE
3) Measures bond qubits and resets them for reuse

After all edges, the atom qubits are measured. This matches the SQMG/QCNC design in the PDF without classical feedforward (no if_test / c_if).

## Environment and metrics
The environment decodes atoms and bonds into an RDKit molecule:
- active_atoms < 2 -> invalid
- add only non-NONE atoms
- add a bond only if both endpoints exist and bond != NONE
- sanitize + canonicalize SMILES
- reject fragmented molecules if SMILES contains '.'

### Strict vs repair mode
- strict (repair_bonds=False, default): no bond repair
- repair (repair_bonds=True): deterministic bond repair by downgrading bond orders that exceed valence

Both strict and repair metrics are available:
- validity_raw_pdf, uniqueness_raw_pdf, reward_raw_pdf
- validity_pdf, uniqueness_pdf, reward_pdf

Composite reward is always defined as Validity x Uniqueness and bounded in [0, 1].

## Quantum Actor-Critic (QRL)
We use a quantum actor-critic (A2C) loop with SPSA updates. The actor outputs a Gaussian policy in a fixed action space, and the critic predicts a value baseline. The QMG parameters are updated by projecting actions into the generator parameter space.

### State features
The A2C state includes:
- valid_ratio, unique_ratio, target_metric
- normalized counts (valid_count/samples, unique_valid_count/samples)
- log-unique novelty feature
- generator weight statistics (mean, std, L2, min, max)

### Reward definition
Step-local reward is PDF-aligned:
```
reward_step = validity_step * uniqueness_step
```
where:
- validity_step = valid_in_batch / batch_size
- uniqueness_step = unique_valid_in_batch / max(valid_in_batch, 1)

### Warm-start and adaptive exploration
- --warm-start-repair N runs the first N episodes in repair mode, then switches to strict
- --adaptive-exploration adapts sigma_max, k_batches, and patience during strict training if eval reward stalls

## CUDA-Q support
This repo supports both Qiskit and CUDA-Q backends:
- --backend qiskit uses Qiskit Aer
- --backend cudaq uses CUDA-Q kernels

Device routing:
- --device cpu or --device gpu (Qiskit)
- --device cuda-cpu or --device cuda-gpu (CUDA-Q)
- --device auto attempts GPU first, then falls back to CPU

## Installation
Create a Python 3.10 environment (recommended with conda):
```bash
conda create -n qrl_fq_5atom python=3.10 -y
conda activate qrl_fq_5atom
```

CPU / Qiskit:
```bash
pip install -r requirements-cpu.txt
pip install -r requirements-dev.txt
```

GPU / Qiskit (requires a supported GPU):
```bash
pip install -r requirements-gpu.txt
pip install -r requirements-dev.txt
```

CUDA-Q:
```bash
pip install -r requirements-cudaq.txt
pip install -r requirements-dev.txt
```

Run tests:
```bash
python -m pytest -q
```

## Usage
### Sampling (SQMG)
```bash
python -m scripts.sample_qmg --mode sqmg --backend qiskit --n 500
python -m scripts.sample_qmg --mode sqmg --backend cudaq --n 500
```

### One-command training (metrics.csv + plots)
```bash
python -m scripts.run_one_train \
  --episodes 2000 \
  --batch-size 256 \
  --backend qiskit \
  --device auto \
  --out runs/exp1 \
  --warm-start-repair 500 \
  --adaptive-exploration \
  --eval-every 50 \
  --eval-shots 2000
```

### Advanced training
```bash
python -m scripts.train_qmg_qrl \
  --algo a2c \
  --steps 5000 \
  --batch-size 256 \
  --backend qiskit \
  --out-dir runs/a2c \
  --warm-start-repair 500 \
  --adaptive-exploration \
  --eval-every 50 \
  --eval-shots 2000
```

## Outputs
Training produces:
- runs/<exp>/metrics.csv (step-level)
- runs/<exp>/eval.csv (evaluation-level)
- reward_eval.png, validity_eval.png, uniqueness_eval.png
- best_weights.npy and best_eval.json (when --track-best is enabled)

## Cleanup
```bash
conda deactivate
conda env remove -n qrl_fq_5atom -y
rm -rf /path/to/Full-Quantum-5Atom-QAC
```

## License
See LICENSE.