Full-Quantum‑5Atom‑QAC
Overview

Full‑Quantum‑5Atom‑QAC is a reference implementation of Scalable Quantum Molecular Generation (SQMG) combined with a Quantum Actor–Critic (QRL) training loop. The goal is to explore the generation of chemically valid 5‑atom molecules using fully quantum circuits and to train the generator with reinforcement learning in strict adherence to the composite objective described in our accompanying paper.

The SQMG circuit uses a 3‑qubit encoding for each heavy atom and a 2‑qubit encoding for each bond. Five heavy atoms are connected in a complete graph (10 edges). Only two bond‑qubits are used; they are dynamically reused for each edge. Ancilla qubits detect whether the two atoms of an edge are both non‑NONE before applying the bond ansatz. The generator produces raw bitstrings for atom types and bond types, which are decoded into molecular graphs.

Training is performed with a quantum actor–critic (A2C) algorithm using SPSA/parameter‑shift for gradient estimation. The state is a high‑level summary of generator behaviour (validity ratio, uniqueness ratio, composite score, and statistics of the generator’s parameters). The action is a direction in parameter space; the actor outputs update directions and the critic predicts state values. The reward is defined as the product of validity and uniqueness and is always bounded between 0 and 1. Our implementation offers both strict mode (raw validity/uniqueness from the quantum circuit) and repair mode (chemically repair invalid structures by downgrading bonds). Strict mode is the default and should be used for official evaluations.

Representation and circuit design

Atom vocabulary:

NONE, C, O, N, S, P, F, Cl


Each heavy atom is represented by 3 qubits (8 possible states). The mapping 000→NONE, 001→C, 010→O, etc., matches the PDF definition
github.com
.

Bond vocabulary:

NONE, SINGLE, DOUBLE, TRIPLE


Each bond is represented by 2 qubits. Bonds are only defined between the 5 heavy atoms, forming a complete graph of 10 edges
github.com
.

Circuit layout: For 5 atoms we use 5×3 + 2 + 2 qubits: 15 qubits for atoms, 2 bond qubits that are dynamically reused per edge, and 2 ancilla qubits to detect NONE states. For each edge (i,j) the circuit:

Computes NONE flags for atoms i and j and stores them on ancilla qubits.

Applies the bond ansatz only if both flags indicate non‑NONE atoms.

Measures the bond qubits and resets them for the next edge.

After processing all edges, measures the 15 atom qubits.

This dynamic reuse and conditional bond operation mirrors the description in the PDF
github.com
.

Environment and metrics

The environment decodes the bitstrings into atoms and bonds and validates the resulting molecule:

Validity: fraction of sampled molecules that are chemically valid (correct valence, single connected component, no hydrogen deficiency, etc.).

Uniqueness: proportion of valid molecules that are distinct SMILES strings.

Composite score: validity × uniqueness. In strict mode this is reward_raw_pdf; in repair mode it is reward_pdf. The maximum value is 1
github.com
.

The environment can optionally repair bonds by downgrading bond orders that violate valence rules. Strict mode (repair_bonds=False) disables this post‑processing.

Reinforcement learning (QRL)

The actor–critic agent learns to update the generator parameters to maximise the composite score. Key features:

Strict default: training and evaluation use strict mode unless warm start indicates otherwise. The agent receives a reward of validity_raw_pdf × uniqueness_raw_pdf.

Warm start: you can start training in repair mode for the first N episodes to avoid a “cold start” where no valid molecules are produced. Use --warm-start-repair N to enable; after N episodes, training switches to strict mode automatically.

Adaptive exploration: optionally adjust exploration hyper‑parameters (sigma_max, patience, k_batches) when the strict reward remains below a threshold over a sliding window. Use --adaptive-exploration together with --adapt-threshold and --adapt-window.

CUDA‑Q support: the generator, actor and critic have both Qiskit and CUDA‑Q implementations. Select the backend via --backend {qiskit,cudaq}. For CUDA‑Q, the environment automatically selects an available GPU if you specify --device cuda-gpu or uses the CPU (--device cuda-cpu) as a fallback.

State features

At each step, the agent observes:

valid_ratio, unique_ratio, target_metric: running averages over the generated batch;

dup_ratio: fraction of duplicated samples;

log(samples), log(unique);

statistics of the generator’s parameters: mean, standard deviation, L2 norm, minimum and maximum.

Reward and maximum value

During training, the reward for each batch is calculated as:

reward = validity_step × uniqueness_step


The maximum possible reward is 1, achieved when all sampled molecules are valid and unique.

Installation

Clone this repository and create a Python 3.10 environment. We recommend using Conda:

conda create -n qrl_fq_5atom python=3.10 -y
conda activate qrl_fq_5atom


Install the required packages for your target backend:

CPU / Qiskit:

pip install -r requirements-cpu.txt


GPU / Qiskit: you need CUDA 11.7+ and a supported GPU. Install:

pip install -r requirements-gpu.txt


CUDA‑Q: you need NVIDIA GPU support and the CUDA‑Q SDK. Install:

pip install -r requirements-cudaq.txt


After installation, run the tests:

python -m pytest -q


Ensure that all tests pass before running long training runs.

Usage
Sampling molecules

To sample molecules using SQMG:

python -m scripts.sample_qmg --mode sqmg --backend qiskit --n 500


Key arguments:

--mode {sqmg,factorized}: choose between the new SQMG generator or the older factorised model.

--backend {qiskit,cudaq}: select the quantum backend.

--n: number of samples.

--repair-bonds: enable repair mode during sampling (defaults to strict mode).

Training with A2C

Use scripts/run_one_train.py for single‑GPU or CPU training with evaluation logging:

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


This command will:

Warm‑start in repair mode for 500 episodes, then switch to strict mode.

Every 50 episodes, evaluate the generator on 2000 samples and write the results to eval.csv.

Produce plot files: reward_eval.png, validity_eval.png, uniqueness_eval.png, and step‑level reward curves.

To use CUDA‑Q with GPU support:

python -m scripts.run_one_train \
    --episodes 2000 \
    --batch-size 256 \
    --backend cudaq \
    --device cuda-gpu \
    --warm-start-repair 500 \
    --adaptive-exploration

Advanced training with multi‑hyperparameters

For more customisable RL training (multiple actors, critics, or experiment sweeps), use scripts/train_qmg_qrl.py. This script supports additional hyper‑parameters and can perform grid search if desired.

Evaluating and plotting

After training, you can inspect:

runs/<exp>/eval.csv: a CSV containing episode, phase, reward_raw_pdf_eval, reward_pdf_eval, validity_eval, uniqueness_eval, sigma_max, k_batches, patience and other metrics.

reward_eval.png, validity_eval.png, uniqueness_eval.png: automatically generated plots showing the composite score and its components over evaluation episodes, including the warm‑start transition marker and best‑so‑far curve.

Use any plotting tool to visualise the training curves or further analyse the CSV.

GPU considerations

Qiskit GPU backend: requires qiskit-aer-gpu (installed via requirements-gpu.txt). It accelerates state‑vector simulation but may still be slower than CUDA‑Q on large sample sizes.

CUDA‑Q backend: offers GPU‑accelerated tensor‑network and state‑vector simulation. Use --device cuda-gpu to select a GPU. If no GPU is available or the backend fails, it automatically falls back to CPU.

Check GPU availability with nvidia-smi. Ensure the environment variables (e.g. CUDA_VISIBLE_DEVICES) are set correctly.

Cleaning up the environment

To remove the project and its Conda environment from your system:

conda deactivate
conda env remove -n qrl_fq_5atom -y
rm -rf /path/to/Full-Quantum-5Atom-QAC

SSH cleanup and re-clone (GPU env example):

rm -rf ~/Full-Quantum-5Atom-QAC
conda deactivate
conda env remove -n qrl_fq_5atom_gpu -y
conda create -n qrl_fq_5atom_gpu python=3.10 -y
conda activate qrl_fq_5atom_gpu
git clone git@github.com:SiriusW823/Full-Quantum-5Atom-QAC.git
cd Full-Quantum-5Atom-QAC
pip install -r requirements-gpu.txt

License

See LICENSE
 for the full license text. The project is released under the MIT License.

Contact

For questions or contributions, please open an issue or pull request on the repository.