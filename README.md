## QRDQN-based Molecule Generation (1-step bandit) with Quantum Generator stub

The project now exposes two paths:
- **QRDQN bandit (default)**: fast, classical stability check with aggressive uniqueness reward.
- **Quantum generator (Qiskit)**: real quantum circuit mapping 5-qubit bitstrings to SMILES scaffolds.

### Environment (classical bandit path)
- **Action space:** Discrete(4) → predefined SMILES templates `["C","N","O","CO"]`.
- **Observation:** Dummy zero vector (shape=(1,)).
- **Terminal reward:** `10.0 * (valid * unique)`; uniqueness tracked across episodes.
- **Validity:** RDKit sanitization + heavy atoms ≤ 5.
- **Episode length:** 1 step.

### Quantum generator (Qiskit path)
- File: `src/environment.py` includes `_run_qiskit_circuit` with AerSimulator and 5-qubit circuit.
- Gates: per-qubit Ry/Rz encoding (10 params) + topology-controlled CX (Linear/Full/Ring).
- Measurement: single-shot bitstring mapped via `BITSTRING_TO_SMILES` (32 scaffolds).
- Reward: validity × uniqueness (scaled by 10).

### Agent (experimental quantum policy)
- File: `src/quantum_agent.py` uses `qiskit_machine_learning` EstimatorQNN + RealAmplitudes (4 qubits) wrapped in a tiny Torch head to emit 11 continuous outputs (10 angles + structure selector).
- File: `train_qrl.py` runs a lightweight REINFORCE-style loop over 2,000 episodes; saves `training_convergence.png`.

### Algorithm & training (QRDQN default)
- **Algorithm:** `sb3_contrib.qrdqn.QRDQN` (MlpPolicy).
- **Hyperparameters:** `learning_rate=3e-4`, `gamma=0.99`, `exploration_fraction=0.5`, `exploration_initial_eps=1.0`, `exploration_final_eps=0.1`.
- **Timesteps:** 100,000.
- **Logging:** Rewards via callback; convergence plot `training_convergence.png`.
- **Evaluation:** 2,000 rollouts computing Validity, Uniqueness, and Golden Metric = Validity × Uniqueness.

### Final QRDQN run (latest)
- Episodes evaluated: 2000  
- Average reward: 0.0050  
- Valid molecules: 2000  
- Unique valid molecules: 1  
- Validity fraction: 1.0000  
- Uniqueness fraction (unique/valid): 0.0005  
- Golden Metric: 0.0005  
- Convergence plot: `training_convergence.png` (early spike then collapse).

### Project Layout
```
.
├── README.md
├── requirements.txt        # rdkit, sb3/sb3-contrib, gymnasium, qiskit, qiskit-aer, qiskit-ml
├── setup_git.sh
├── src/
│   ├── __init__.py
│   ├── environment.py      # one-step bandit env + Qiskit generator mapping bitstrings→SMILES
│   ├── quantum_agent.py    # experimental quantum policy (EstimatorQNN + Torch head)
│   ├── circuits.py         # legacy
│   ├── embedding.py        # legacy
│   └── agent.py            # legacy
├── train.py                # QRDQN training/eval/plot (bandit path)
└── train_qrl.py            # experimental REINFORCE with quantum_agent + environment
```

### Installation
```bash
python -m venv .venv
.\.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```
If `rdkit-pypi` fails, install RDKit via conda-forge and pip the rest.

### Run QRDQN training (default)
```bash
python train.py
```
Outputs:
- Console logs during training; final summary after 2,000 eval episodes.
- `training_convergence.png` in project root.

### Run quantum policy experiment
```bash
python train_qrl.py
```
Produces a simple convergence plot and prints unique count from the Qiskit-driven environment.
