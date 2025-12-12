## QRDQN-based Molecule Generation (1-step bandit, aggressive uniqueness reward)

This project now uses **SB3-Contrib QRDQN** as a one-step contextual bandit to pick a discrete SMILES template. The quantum VQC actor-critic stack was removed for stability; exploration and distributional Q-learning handle mode collapse. RDKit validates molecules; Matplotlib plots training convergence.

### Current environment & reward
- **Action space:** Discrete(4) → predefined SMILES templates `["C","N","O","CO"]`.
- **Observation:** Dummy zero vector (shape=(1,)).
- **Terminal reward:** `10.0 * (valid * unique)`; uniqueness tracked across episodes.
- **Validity:** RDKit sanitization + heavy atoms ≤ 5.
- **Episode length:** 1 step (bandit).

### Algorithm & training
- **Algorithm:** `sb3_contrib.qrdqn.QRDQN` (MlpPolicy).
- **Hyperparameters:** `learning_rate=3e-4`, `gamma=0.99`, `exploration_fraction=0.5`, `exploration_initial_eps=1.0`, `exploration_final_eps=0.1`.
- **Timesteps:** 100,000.
- **Logging:** Rewards logged via callback; convergence plot saved to `training_convergence.png`.
- **Evaluation:** 2,000 rollouts computing Validity, Uniqueness, and Golden Metric = Validity × Uniqueness.

### Final run (latest)
- Episodes evaluated: 2000
- Average reward: 0.0050
- Valid molecules: 2000
- Unique valid molecules: 1
- Validity fraction: 1.0000
- Uniqueness fraction (unique/valid): 0.0005
- Golden Metric: 0.0005
- Convergence plot: `training_convergence.png` (shows early spike then collapse to 0 reward).

### Files
```
.
├── README.md
├── requirements.txt        # pinned deps (rdkit, sb3, sb3-contrib, gymnasium, etc.)
├── setup_git.sh
├── src/
│   ├── __init__.py
│   ├── environment.py      # one-step bandit env, aggressive reward = 10 * valid * unique
│   ├── circuits.py         # legacy (unused in current pipeline)
│   ├── embedding.py        # legacy (unused)
│   └── agent.py            # legacy (unused)
└── train.py                # QRDQN training, logging, plotting, evaluation
```

### Installation
```bash
python -m venv .venv
.\.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```
If `rdkit-pypi` fails, install RDKit via conda-forge and pip the rest.

### Run training
```bash
python train.py
```
Outputs:
- Console logs during training; final summary after 2,000 eval episodes.
- `training_convergence.png` convergence curve in project root.
