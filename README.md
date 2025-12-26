

# Full-Quantum-5Atom-QAC

Full-Quantum-5Atom-QAC is a **reference implementation of Scalable Quantum Molecular Generation (SQMG/QMG)** combined with **Quantum Reinforcement Learning (QRL)** for molecule generation with **exactly five heavy atoms**.

This project is **strictly aligned** with the design and objective defined in  
**QMG_QCNC_final_submission.pdf**, and intentionally follows a **pure QMG + QRL approach**  
(no Bayesian Optimization, no COBYLA).

The repository supports **Qiskit** and **CUDA-Q** backends, implements **strict (no-repair) evaluation by default**, and provides **warm-start and adaptive exploration** to make strict QRL training feasible.

---

## 1. Key Features

- **PDF-aligned SQMG / QMG representation**
  - 5 heavy atoms
  - Full graph with 10 edges
  - Dynamic bond-qubit reuse
  - Ancilla-controlled bond activation
- **Strict chemical evaluation by default**
  - No post-hoc bond repair unless explicitly enabled
- **Composite reward**
  - `reward = validity × uniqueness`
  - Maximum reward = **1.0**
- **Quantum Reinforcement Learning (QRL)**
  - Actor–Critic (A2C-style)
  - Qiskit and CUDA-Q backends
- **Warm-start curriculum**
  - Optional repair → strict transition
- **Adaptive exploration**
  - Automatically increases exploration when reward is too sparse
- **Reproducible evaluation**
  - Fixed eval protocol
  - CSV logs and convergence plots
- **CPU / GPU / CUDA-Q support**

---

## 2. SQMG / QMG Design (PDF-Aligned)

### 2.1 Atom and Bond Encoding

| Component | Encoding | Qubits |
|----------|---------|--------|
| Atoms | `["NONE","C","O","N","S","P","F","Cl"]` | 3 |
| Bonds | `["NONE","SINGLE","DOUBLE","TRIPLE"]` | 2 |

---

### 2.2 Graph Structure

- Exactly **5 heavy atoms**
- **Full molecular graph** with **10 edges**:
```

(1,2) (1,3) (2,3)
(1,4) (2,4) (3,4)
(1,5) (2,5) (3,5) (4,5)

````
- **Bond qubits are dynamically reused**:
- For each edge:
  1. Apply bond ansatz
  2. Measure bond qubits
  3. Reset bond qubits
- **Ancilla-controlled wiring**:
- Bond ansatz is applied **only if both endpoint atoms are not `NONE`**

This exactly follows the **strict SQMG wiring** described in the PDF.

---

## 3. Reward Definition (Strict by Default)

### 3.1 Strict Mode (Default)

```text
reward_raw_pdf = validity_raw_pdf × uniqueness_raw_pdf
````

* `repair_bonds = False`
* No post-processing or bond correction
* Used for **official evaluation and best checkpoint selection**

---

### 3.2 Repair Mode (Optional)

```text
reward_pdf = validity_pdf × uniqueness_pdf
```

* `repair_bonds = True`
* Bond repair is applied before validation
* Intended **only for warm-start or analysis**
* **Not used for final PDF-aligned evaluation**

---

## 4. Quantum Reinforcement Learning (QRL)

### 4.1 Algorithm

* Actor–Critic (A2C-style)
* Policy outputs **parameter updates (Δθ)** for the SQMG circuit
* Critic provides a baseline value estimate
* Gradient estimation via **SPSA / parameter-shift–style updates**

---

### 4.2 Warm-Start Curriculum (Optional)

```bash
--warm-start-repair N
```

Behavior:

* First `N` episodes:

  * `repair_bonds = True`
  * Reward uses `reward_pdf`
* After episode `N`:

  * Automatically switches to **strict mode**
  * Reward uses `reward_raw_pdf`

The evaluation log (`eval.csv`) records the phase:

```text
phase = "repair" | "strict"
```

---

### 4.3 Adaptive Exploration (Optional)

```bash
--adaptive-exploration
```

During **strict phase only**, if the reward remains sparse:

Default condition:

```text
reward_raw_pdf_eval < 0.01 for last 5 evals
```

Automatic adjustments:

* `sigma_max = min(sigma_max × 1.2, 2.0)`
* `patience += 20` (up to 200)
* `k_batches += 1` (up to 8)

All changes are logged in `eval.csv`.

CLI controls:

```bash
--adaptive-exploration
--no-adaptive-exploration
--adapt-threshold
--adapt-window
```

---

## 5. Backends and Devices

### 5.1 Supported Backends

```text
--backend {qiskit, cudaq}
```

### 5.2 Device Selection

```text
--device {auto, cpu, gpu, cuda-cpu, cuda-gpu}
```

Behavior:

* `auto`:

  * CUDA-Q GPU if available
  * otherwise CPU
* Clear startup logs print:

  * backend
  * device
  * CUDA-Q target
  * GPU availability

---

## 6. Installation

### 6.1 CPU (Qiskit)

```bash
pip install -r requirements-cpu.txt
```

### 6.2 GPU (Qiskit Aer)

```bash
pip install -r requirements-gpu.txt
```

### 6.3 CUDA-Q

```bash
pip install -r requirements-cudaq.txt
```

> CUDA-Q GPU execution requires a supported NVIDIA GPU and CUDA runtime.

---

## 7. Usage

### 7.1 Sampling (SQMG only)

```bash
python -m scripts.sample_qmg \
  --mode sqmg \
  --backend qiskit \
  --n 200
```

With bond repair:

```bash
python -m scripts.sample_qmg \
  --mode sqmg \
  --backend qiskit \
  --n 200 \
  --repair-bonds
```

---

### 7.2 Training (Warm-Start + Adaptive Exploration)

```bash
python -m scripts.run_one_train \
  --episodes 2000 \
  --batch-size 256 \
  --backend qiskit \
  --device cpu \
  --out runs/exp1 \
  --warm-start-repair 500 \
  --adaptive-exploration \
  --eval-every 50 \
  --eval-shots 2000
```

CUDA-Q (auto GPU):

```bash
python -m scripts.run_one_train \
  --episodes 2000 \
  --batch-size 256 \
  --backend cudaq \
  --device auto \
  --out runs/exp_cudaq \
  --warm-start-repair 500 \
  --adaptive-exploration
```

---

## 8. Outputs

Each run directory contains:

* `metrics.csv` — step-level training metrics
* `eval.csv` — evaluation metrics
* `best_weights.npy` — best checkpoint (eval-based)

Generated plots:

* `reward_eval.png`
* `validity_eval.png`
* `uniqueness_eval.png`

Plots include:

* Best-so-far curves
* Warm-start → strict transition marker

---

## 9. Testing

```bash
python -m pytest -q
```

Includes smoke tests for:

* CLI parameters
* Repair / strict behavior
* Warm-start phase transition
* Adaptive exploration updates

---

## 10. Environment Cleanup (Recommended)

```bash
conda deactivate
conda env remove -n qrl_fq_5atom_gpu -y
rm -rf Full-Quantum-5Atom-QAC
```

---

## 11. License

This project is released under the **MIT License**.

