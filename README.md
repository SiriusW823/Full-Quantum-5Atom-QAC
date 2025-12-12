## Full Quantum RL for 5-Atom Molecule Generation

End-to-end **quantum reinforcement learning** system for de novo molecules with up to **5 heavy atoms**. Both policy and value functions are PennyLane **variational quantum circuits (VQCs)**; no classical neural networks are used. RDKit is used only for assembly/validation, and Matplotlib provides convergence visualization.

### Chemistry & Sequence
- Sequence length 9: Atom1 → Bond1 → Atom2 → Bond2 → Atom3 → Bond3 → Atom4 → Bond4 → Atom5.
- Allowed atoms: `['NONE', 'C', 'N', 'O']`; allowed bonds: `['NONE', 'SINGLE', 'DOUBLE', 'TRIPLE']`.
- Choosing `NONE` on an atom step halts generation early (padding the rest with NONE). Maximum heavy atoms = 5.

### Quantum Actor–Critic
- **Actor (Agent A):** 9-qubit VQC (AngleEmbedding → StronglyEntanglingLayers, 4 layers) returning 4 expectation values → softmax → discrete action.
- **Critic (Agent B):** Separate 9-qubit VQC with the same topology, outputs scalar `V(s)` for the advantage.
- **State encoding:** 9-step discrete history (IDs 0–3) mapped to rotation angles in `[0, π]`.
- **Reward shaping:** `Reward = (Validity * Uniqueness) + 0.1 * Validity` to stabilize gradients while preserving the optimal policy (Valid × Unique → 1). Golden Metric is still logged for convergence tracking.

### Training Loop (train.py)
- Hyperparameters: `episodes=2000`, `batch_size=32`, `lr=0.0005`, `entropy_beta=0.001`, `epsilon=0.15`, gradient clip `0.5`.
- Actor loss includes explicit policy entropy `H(π)` with β = 0.001 to reduce mode collapse.
- Logs every ~50 episodes: batch score, valid/unique counts, actor/critic losses, and top-3 SMILES from the batch.
- Saves convergence plot `training_convergence.png` with raw batch scores and a moving-average trendline.
- Prints final Golden Metric: `(valid/episodes) * (unique/episodes)`.

### Project Layout
```
.
├── README.md
├── requirements.txt
├── setup_git.sh
├── src/
│   ├── __init__.py
│   ├── circuits.py       # 9-qubit actor/critic qnodes with StronglyEntanglingLayers
│   ├── embedding.py      # map 9-step history -> [0, π] rotation angles
│   ├── environment.py    # RDKit-backed linear builder + Golden Metric helper
│   └── agent.py          # QuantumActorCritic (sampling + batch updates)
└── train.py              # Training loop, logging, convergence plotting
```

### Installation
```bash
python -m venv .venv
.\.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```
If `rdkit-pypi` fails on your platform, install RDKit via conda-forge and pip the rest. Packages are pinned to NumPy 1.26.x to avoid `_ARRAY_API` incompatibilities with RDKit and PennyLane.

### Run Training
```bash
python train.py
```
Outputs:
- Console logs with batch Golden Metric and sample SMILES.
- `training_convergence.png` in the project root tracking the Golden Metric over training.
