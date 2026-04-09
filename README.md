Controlled Regularization and Optimization Study on CIFAR-10

## Overview

This repository presents a **controlled deep learning study** on **regularization**, **robustness**, and **optimization** using an intentionally overfitting multilayer perceptron (MLP) on **CIFAR-10**.

The project was designed as a **methodology-first experimental pipeline** rather than a generic image classification project. The main idea is to keep the backbone, split, and core training protocol as fixed as possible, then introduce methods step by step so that their effect can be isolated and interpreted clearly.

Instead of jumping directly to a “best model,” the project follows a layered structure:

1. build an overfitting baseline,
2. compare standard regularization methods under identical settings,
3. extend the analysis with additional regularization techniques,
4. combine multiple techniques into a stronger integrated setup,
5. and finally optimize that stronger setup through optimizer / learning-rate / scheduler comparisons.

This means the repository functions as both:

- a **regularization ablation study**, and
- an **optimization study** built on top of a stronger regularized pipeline.

> **Note:** This `README.md` is meant to document the project technically and methodologically.  
> A separate `report.md` is intended for deeper discussion, interpretation, and final academic analysis.

---

## Project Goals

The project does not aim only to maximize final test accuracy. Its main objective is to understand:

- how different regularization methods affect **overfitting**,
- which methods improve **validation and test behavior**,
- how methods with different mechanisms behave under the same setup,
- how robustness-oriented training differs from classical regularization,
- what happens when several methods are combined,
- and how optimizer choice changes **convergence**, **stability**, and **generalization**.

So the guiding question is not only **“what works?”** but also:

- **why does it help?**
- **what kind of problem does it address?**
- **how does it fit into the full training pipeline?**

---

## Methodological Design

The strongest design choice in this repository is the use of a **controlled comparison framework**.

For the main regularization branch, the following are held fixed as much as possible:

- same dataset: **CIFAR-10**
- same split: **15,000 train / 5,000 validation / 10,000 test**
- same MLP backbone
- same hidden dimensions
- same batch size
- same epoch budget
- same random seed
- same train / validation / test protocol

This makes the **regularization method itself** the primary changing factor.

Later notebooks intentionally move beyond the initial minimal setup, but still do so in a structured order:

- first by studying isolated regularizers,
- then by adding more advanced methods,
- then by building a **full-combo regularized configuration**,
- and finally by tuning optimization on top of that configuration.

This staged design makes the project much stronger than a simple collection of loosely related notebooks.

---

## Dataset and Split Strategy

All experiments are based on **CIFAR-10**.

The main split used throughout the project is:

- **Train:** 15,000 images
- **Validation:** 5,000 images
- **Test:** 10,000 images

The split is fixed using:

- **seed = 42**

This fixed split is reused across experiments so that performance changes come from the method itself rather than sampling variation.

In the modular regularization branch, the split is saved and reused via:

- `splits/fixed_cifar10_split.npz`

This design supports:

- fair method comparison,
- reproducibility,
- and direct interpretation of generalization gaps.

---

## Base Model

The shared backbone is a fully connected **MLP** with:

- **Input dimension:** 3072 (`32 × 32 × 3`)
- **Hidden layers:** `[1024, 512, 256]`
- **Activation:** `ReLU`
- **Output:** 10 logits for CIFAR-10 classes

Optional components, depending on the experiment:

- **Dropout**
- **Batch Normalization**

### Parameter scale

The model is intentionally large relative to the reduced training subset:

- Plain MLP (without BatchNorm): approximately **3,805,450** trainable parameters
- BatchNorm-enabled variant: approximately **3,809,034** trainable parameters

This is large enough to make overfitting clearly visible on the fixed 15k training subset.

---

## Core Controlled Setup

The first regularization notebooks share the same main configuration:

- **Optimizer:** `SGD`
- **Learning rate:** `0.01`
- **Batch size:** `128`
- **Max epochs:** `50`
- **Seed:** `42`
- **Hidden sizes:** `[1024, 512, 256]`

This setup is reused across:

- baseline,
- dropout,
- early stopping,
- batch normalization,
- dropout + batch norm,
- and L2 + dropout.

As a result, the only meaningful changing factor in these notebooks is the regularization strategy itself.

---

## Evaluation Philosophy

Across the repository, experiments track combinations of:

- training loss
- validation loss
- test loss
- training accuracy
- validation accuracy
- test accuracy
- generalization gap
- robustness-oriented evaluation under adversarial perturbation
- summary plots
- comparison tables

For this project, accuracy alone is not enough. A method is also evaluated by:

- how much it reduces overfitting,
- how stable its validation behavior is,
- how it changes train–validation and train–test gaps,
- and whether it improves robustness when applicable.

---

## Implemented Methods and Notebook Progression

The project progresses notebook by notebook, with each notebook having a distinct methodological role.

---

### `reg_opt_00_baseline.ipynb` — Standalone Overfitting Baseline

This notebook constructs the plain overfitting baseline and demonstrates why regularization is necessary.

Configuration:

- MLP `[1024, 512, 256]`
- no dropout
- no batch normalization
- no early stopping
- no explicit L1 / L2 regularization
- optimizer: `SGD`
- learning rate: `0.01`
- batch size: `128`
- max epochs: `50`

Purpose:

- establish a clean reference point,
- show severe overfitting clearly,
- quantify the generalization gap before applying any correction.

This notebook is important because it proves that the problem setup is non-trivial: the model fits the training subset very well but generalizes poorly without regularization.

---

### `reg_00_modular_baseline.ipynb` — Modular Branch Baseline

This is the branch-style version of the same baseline idea, built on top of the reusable `src/` pipeline.

Its purpose is to create the same no-regularization reference point under the modular experimental framework used by the regularization ablation notebooks.

This notebook establishes the **fixed conditions** for the main ablation branch:

- same fixed split
- same MLP architecture
- same optimizer
- same learning rate
- same epoch limit
- same seed

---

### `reg_01_dropout.ipynb` — Dropout

This notebook introduces **dropout** as the first stochastic regularization method.

Configuration:

- dropout enabled
- dropout probability: `0.5`
- no batch norm
- no early stopping
- no explicit weight decay
- optimizer remains `SGD`
- learning rate remains `0.01`

Purpose:

- reduce co-adaptation between hidden units,
- inject stochastic noise into the representation,
- test whether the generalization gap narrows under the same controlled setup.

This notebook isolates a purely stochastic regularization mechanism.

---

### `reg_02_early_stopping.ipynb` — Early Stopping

This notebook studies **early stopping** as an implicit regularizer.

Configuration:

- no dropout
- no batch normalization
- no explicit L1 / L2 penalty
- early stopping enabled with:
  - `patience = 5`
  - `min_delta = 1e-4`

Purpose:

- stop training when validation loss stops improving,
- prevent harmful late-stage memorization,
- reduce overfitting through training dynamics rather than weight constraints or noise injection.

This notebook is important because it shows that regularization does not always mean “change the model” or “add a penalty”; it can also mean **controlling training duration**.

---

### `reg_03_batchnorm.ipynb` — Batch Normalization

This notebook studies **batch normalization** as an architectural / optimization-related regularizer.

Configuration:

- batch norm enabled
- no dropout
- no early stopping
- no explicit weight decay

Purpose:

- stabilize hidden activation distributions,
- smooth optimization,
- observe whether representation stabilization indirectly improves generalization.

This method occupies a different conceptual category from L1/L2, dropout, or early stopping, which makes it especially useful in a methodology-oriented comparison.

---

### `reg_04_dropout_batchnorm.ipynb` — Dropout + BatchNorm

This notebook studies the interaction between:

- stochastic regularization (`Dropout`)
- architectural / normalization-based regularization (`BatchNorm`)

Configuration:

- dropout enabled
- dropout probability: `0.5`
- batch norm enabled
- no early stopping
- no explicit L2

Purpose:

- test whether two methods with different mechanisms complement each other,
- compare the combination directly against each method in isolation.

This is one of the first notebooks where the project moves from isolated methods to controlled combination studies.

---

### `reg_05_l2_map_gaussian_prior_dropout.ipynb` — L2 + Dropout with MAP Interpretation

This notebook introduces explicit **L2 regularization** together with dropout and connects the implementation to the **MAP estimation** view.

Training objective:

`L_MAP = L_CE + λ Σ_i w_i^2`

Interpretation:

- `CrossEntropyLoss` corresponds to the negative log-likelihood,
- the L2 penalty corresponds to the negative log-prior under a zero-mean Gaussian prior.

Configuration:

- dropout enabled
- dropout probability: `0.5`
- `weight_decay = 1e-4`
- no batch norm
- no early stopping

Purpose:

- connect theory to implementation,
- treat L2 not just as a coding trick but as an explicit probabilistic regularizer,
- compare explicit parameter constraint + stochastic masking under the same setup.

This notebook is particularly important for technical depth because it links the code to a statistical interpretation.

---

### `reg_06_comparison.ipynb` — Aggregated Core Comparison

This notebook consolidates the results of the core regularization branch.

Compared experiments:

- Baseline
- Dropout
- Early Stopping
- Batch Normalization
- Dropout + BatchNorm
- L2 + Dropout

Tracked metrics include:

- training loss
- validation loss
- test loss
- training accuracy
- validation accuracy
- test accuracy
- generalization gap

Purpose:

- move from individual notebook outputs to a true ablation-style comparison,
- evaluate how different regularization families behave under identical conditions.

---

### `reg_07_run_all.ipynb` — Main Branch Execution Notebook

This notebook runs the core branch experiments programmatically in a more systematic way.

It defines experiment configurations such as:

- baseline
- dropout (`dropout_p = 0.5`)
- early stopping (`patience = 5`)
- batch norm
- dropout + batch norm
- L2 + dropout (`weight_decay = 1e-4`)

Purpose:

- avoid manual rerunning of each notebook one by one,
- make the branch more reproducible,
- support batch-style execution and result collection.

---

### `reg_08_l1_l2_v2.ipynb` — Explicit Regularization Study: Baseline vs L1 vs L2

This notebook deepens the project’s explicit regularization analysis by directly comparing:

- baseline
- L1 regularization
- L2 regularization

Shared setup:

- optimizer: `SGD`
- learning rate: `0.01`
- max epochs: `50`
- seed: `42`

Main regularization settings:

- `L1_LAMBDA = 1e-4`
- `L2_WEIGHT_DECAY = 1e-3`

Additional follow-up:

- stronger L2 experiment with:
  - `L2_WEIGHT_DECAY = 5e-3`

This notebook goes beyond simple accuracy comparison and also analyzes:

- L1 norm tracking
- L2 norm behavior
- weight distributions
- sparsity behavior
- fraction of near-zero weights

Purpose:

- compare two explicit norm penalties directly,
- study the trade-off between sparsity and performance,
- observe how L1 and L2 differ structurally, not just numerically.

This is one of the most methodologically rich notebooks in the project because it links regularization to **weight geometry** and **parameter behavior**.

---

### `reg_09_data_augmentation.ipynb` — Data Augmentation as Regularization

This notebook treats **data augmentation** as a regularization mechanism by modifying the training set only while keeping validation and test clean.

Training-time augmentation pipeline:

- `RandomHorizontalFlip(p=0.5)`
- `RandomCrop(32, padding=4)`
- `ColorJitter(brightness=0.2, contrast=0.2)`

Controlled setup:

- model: MLP `[1024, 512, 256]`
- optimizer: `SGD`
- learning rate: `0.01`
- max epochs: `50`
- batch size: `128`
- seed: `42`

Purpose:

- increase effective training diversity,
- discourage memorization,
- reduce the generalization gap without changing the parameter-space penalty.

This notebook expands the project beyond classical weight-based regularization and into **input-space regularization**.

---

### `reg_10_adversarial_training.ipynb` — FGSM Adversarial Training

This notebook studies **robustness-oriented regularization** using **FGSM**.

Method summary:

- adversarial examples are generated during training,
- the training objective combines:
  - **50% clean loss**
  - **50% adversarial loss**
- perturbation magnitude:
  - `epsilon = 0.03`

Controlled setup:

- model: MLP `[1024, 512, 256]`
- optimizer: `SGD`
- learning rate: `0.01`
- max epochs: `50`
- seed: `42`

Evaluation includes:

- clean test accuracy
- adversarial test accuracy

Purpose:

- study how adversarial training affects generalization,
- test whether robustness-oriented training behaves like another form of regularization,
- compare it with more classical anti-overfitting techniques.

This notebook is valuable because it extends the project from standard generalization to **robust generalization**.

---

### `reg_11_full_combo_before_optimization.ipynb` — Full Combined Setup

This notebook is the bridge between the regularization study and the optimization study.

It combines multiple methods into one integrated training pipeline.

#### Data pipeline

Training augmentation in this notebook includes:

- `RandomHorizontalFlip()`
- `RandomCrop(32, padding=4)`
- `ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)`

#### Model / training configuration

- dropout enabled
  - `dropout_p = 0.2`
- batch normalization enabled
- L1 regularization
  - `l1_lambda = 5e-7`
- L2 regularization
  - `weight_decay = 1e-4`
- adversarial training enabled
  - `adv_ratio = 0.15`
  - `epsilon = 0.03`
- early stopping enabled
  - `patience = 10`
  - `min_delta = 0.001`
- optimizer upgraded to:
  - `AdamW`
- learning rate:
  - `1e-3`
- scheduler:
  - `CosineAnnealingLR`

Important detail:

Unlike the standalone FGSM notebook, this full-combo setup does **not** use a strict 50/50 clean + adversarial loss on every batch. Instead, adversarial batches are injected with probability:

- `adv_ratio = 0.15`

So this notebook implements a **lighter adversarial regime** inside a richer multi-regularizer pipeline.

Purpose:

- combine complementary regularization mechanisms,
- test whether isolated gains can be accumulated,
- prepare a strong base configuration for optimizer tuning.

This notebook is one of the most important ones in the repository because it represents the transition from **analysis of parts** to **construction of a strong full system**.

---

### `opt_01_optimization_final.ipynb` — Final Optimization Study

This notebook keeps the strong regularized pipeline fixed and studies optimization on top of it.

The fixed full-combo techniques include:

- dropout: `0.2`
- batch norm: on
- L1 regularization: `5e-7`
- L2 weight decay: `1e-4`
- data augmentation: on
- adversarial training: on
  - `adv_ratio = 0.15`
  - `epsilon = 0.03`
- early stopping: on
  - `patience = 10`

### Optimization search process

#### Step A — Fast optimizer comparison

Candidates:

- `SGD` with `lr = 0.01`
- `SGD + Momentum` with `lr = 0.01`
- `AdamW` with `lr = 0.001`

#### Step B — Fast learning-rate comparison

After choosing the best optimizer, the notebook compares learning rates.

For `AdamW`, the candidates are:

- `1e-2`
- `1e-3`
- `1e-4`

#### Step C — Fast scheduler comparison

Schedulers tested:

- `None`
- `StepLR`
- `CosineAnnealingLR`
- `ReduceLROnPlateau`

#### Step D — Final best run

After selecting the best optimizer / learning rate / scheduler combination, the notebook launches the final longer training run.

Purpose:

- move beyond regularization and study training dynamics directly,
- identify which optimizer family works best for the strong combined pipeline,
- understand the effect of scheduler choice on convergence and final accuracy.

This notebook completes the project by turning the strongest regularized setup into a fully tuned training pipeline.

---

## Key Reported Results from Saved Notebook Outputs

Since the repository contains multiple stages and some later notebooks rerun related ideas under different conditions, results belong to their own notebook contexts. The tables below summarize **reported outputs saved inside the notebooks**.

---

### 1. Standalone baseline result (`reg_opt_00_baseline.ipynb`)

| Setup | Train Acc | Val Acc | Test Acc | Train-Val Gap | Train-Test Gap |
|---|---:|---:|---:|---:|---:|
| Baseline (No Regularization) | 85.81% | 44.50% | 45.82% | 41.31% | 39.99% |

Interpretation:

- the MLP fits the training subset very strongly,
- but validation and test performance remain much lower,
- making the baseline a valid starting point for a regularization study.

---

### 2. Explicit regularization comparison (`reg_08_l1_l2_v2.ipynb`)

| Model | Train Acc | Val Acc | Test Acc | Train-Val Gap |
|---|---:|---:|---:|---:|
| Baseline (No Regularization) | 84.87% | 43.84% | 45.37% | 41.03% |
| L1 Regularization | 65.58% | 40.46% | 41.40% | 25.12% |
| L2 Regularization (SGD) | 80.43% | 43.36% | 44.67% | 37.07% |

Additional sparsity result reported in the same notebook:

- fraction of weights with `|w| < 0.01`
  - Baseline: **50.5%**
  - L1: **78.2%**
  - L2: **53.5%**

Interpretation:

- **L1** substantially reduces the generalization gap,
- but it also reduces train / validation / test accuracy,
- and it clearly promotes sparse weights,
- while **L2** is more conservative and keeps performance closer to baseline.

---

### 3. Data augmentation result (`reg_09_data_augmentation.ipynb`)

| Model | Train Acc | Val Acc | Test Acc | Train-Val Gap |
|---|---:|---:|---:|---:|
| Data Augmentation | 47.75% | 44.68% | 46.02% | 3.07% |

Interpretation:

- augmentation sharply reduces the generalization gap,
- training accuracy becomes much lower than the baseline,
- but validation and test performance remain competitive.

This is one of the clearest examples in the repository of a method that sacrifices raw fitting ability in exchange for improved generalization stability.

---

### 4. Standalone adversarial training result (`reg_10_adversarial_training.ipynb`)

| Model | Train Acc | Val Acc | Clean Test Acc | Adversarial Test Acc | Train-Val Gap | Robustness Drop |
|---|---:|---:|---:|---:|---:|---:|
| FGSM Adversarial Training | 70.95% | 46.00% | 48.05% | 31.19% | 24.95% | 16.86% |

Interpretation:

- adversarial training improves robustness awareness,
- but by itself it does not produce the strongest overall generalization profile in the repository,
- and the clean-to-adversarial degradation remains substantial.

So in this project, adversarial training becomes more effective when used as part of a broader combined pipeline.

---

### 5. Full combined setup result (`reg_11_full_combo_before_optimization.ipynb`)

| Setup | Train Acc | Val Acc | Test Acc | Adversarial Test Acc | Train-Val Gap | Train-Test Gap |
|---|---:|---:|---:|---:|---:|---:|
| Full Combo + AdamW + CosineAnnealingLR | 58.73% | 54.92% | 55.78% | 43.83% | 3.81% | 2.95% |

Interpretation:

- this is one of the strongest overall configurations in the repository,
- it combines strong validation / test performance with very small generalization gaps,
- and it improves adversarial accuracy substantially compared with standalone FGSM training.

This supports the main methodological lesson of the project: **complementary methods work better together than in isolation**.

---

### 6. Final optimization search results (`opt_01_optimization_final.ipynb`)

#### Fast optimizer comparison

| Optimizer | LR | Train Acc | Val Acc | Test Acc |
|---|---:|---:|---:|---:|
| SGD | 0.01 | 45.79% | 42.94% | 44.35% |
| SGD + Momentum | 0.01 | 46.41% | 43.02% | 44.86% |
| AdamW | 0.001 | 48.42% | 45.78% | 47.16% |

Selected optimizer: **AdamW**

#### Fast learning-rate comparison for AdamW

| Optimizer | LR | Train Acc | Val Acc | Test Acc | Adv Acc |
|---|---:|---:|---:|---:|---:|
| AdamW | 0.01 | 47.39% | 44.12% | 45.27% | 37.84% |
| AdamW | 0.001 | 48.15% | 45.88% | 46.25% | 37.76% |
| AdamW | 0.0001 | 48.05% | 45.44% | 46.71% | 36.04% |

Selected learning rate: **0.001**

#### Fast scheduler comparison

| Scheduler | Train Acc | Val Acc | Test Acc | Adv Acc |
|---|---:|---:|---:|---:|
| None | 47.82% | 44.76% | 46.24% | 37.51% |
| StepLR | 50.48% | 47.08% | 47.76% | 39.06% |
| CosineAnnealingLR | 50.71% | 46.90% | 49.12% | 39.78% |
| ReduceLROnPlateau | 49.74% | 46.30% | 47.97% | 39.68% |

Selected scheduler by validation performance: **StepLR**

#### Final optimization result

| Final Setup | Train Acc | Val Acc | Test Acc | Adversarial Acc | Train-Val Gap | Train-Test Gap |
|---|---:|---:|---:|---:|---:|---:|
| AdamW + `lr=0.001` + StepLR on full combo | 60.89% | 54.00% | 54.87% | 43.35% | 6.89% | 6.02% |

Interpretation:

- `AdamW` is the strongest optimizer among the tested candidates for the full-combo regime,
- the final tuned setup remains much stronger than the plain baseline,
- and adversarial accuracy stays far above the standalone FGSM baseline.

---

## Main Findings

### 1. The plain baseline overfits heavily

Across baseline runs, the model reaches roughly:

- **~85% training accuracy**
- but only **~44–46% validation/test accuracy**

This creates a very large generalization gap of around **41%**, confirming that the project starts from a genuinely overfitting setup.

---

### 2. Different regularizers help through different mechanisms

The project shows that regularization is not one single phenomenon.

Different methods act in different spaces:

- **L1 / L2** act in parameter space,
- **Dropout** injects noise into hidden representations,
- **Early stopping** acts on training dynamics,
- **BatchNorm** stabilizes internal activations and optimization,
- **Data augmentation** acts in input space,
- **Adversarial training** regularizes through hard perturbed examples.

This is why the repository is organized as a method comparison rather than a single benchmark race.

---

### 3. L1 is especially effective for sparsity

The explicit regularization notebook reports:

- Baseline near-zero weights: **50.5%**
- L1 near-zero weights: **78.2%**
- L2 near-zero weights: **53.5%**

This clearly shows that L1 changes the weight distribution much more aggressively than L2.

---

### 4. Data augmentation strongly narrows the gap

The augmentation notebook achieves:

- train accuracy: **47.75%**
- validation accuracy: **44.68%**
- test accuracy: **46.02%**
- train-val gap: **3.07%**

So although it lowers training accuracy, it acts as a very strong anti-overfitting mechanism.

---

### 5. Adversarial training alone helps robustness, but combination is stronger

The standalone FGSM notebook improves robustness awareness, but its final adversarial accuracy is still lower than the adversarial accuracy achieved by the later full-combo setup.

This suggests that adversarial training works best in this project as part of a broader regularized pipeline rather than as a standalone fix.

---

### 6. The strongest overall behavior comes from combination

The full-combo pipeline achieves:

- **54.92% validation accuracy**
- **55.78% test accuracy**
- **43.83% adversarial accuracy**
- and very small train-val / train-test gaps

This is one of the central conclusions of the project:

> strong generalization in this setup does not come from a single trick,  
> but from combining complementary regularization mechanisms in a principled way.

---

### 7. AdamW is the best optimizer among the tested candidates

In the final optimization notebook, `AdamW` outperforms both `SGD` and `SGD + Momentum` under the fixed full-combo setup.

That result is consistent with the later-stage project design, where optimization is performed on an already strong regularized pipeline rather than on the weak baseline.

---

## Repository Structure

```text
DL-Project-One/
├── README.md
├── report.md / report.pdf
├── notebooks/
│   ├── reg_opt_00_baseline.ipynb
│   ├── reg_00_modular_baseline.ipynb
│   ├── reg_01_dropout.ipynb
│   ├── reg_02_early_stopping.ipynb
│   ├── reg_03_batchnorm.ipynb
│   ├── reg_04_dropout_batchnorm.ipynb
│   ├── reg_05_l2_map_gaussian_prior_dropout.ipynb
│   ├── reg_06_comparison.ipynb
│   ├── reg_07_run_all.ipynb
│   ├── reg_08_l1_l2_v2.ipynb
│   ├── reg_09_data_augmentation.ipynb
│   ├── reg_10_adversarial_training.ipynb
│   ├── reg_11_full_combo_before_optimization.ipynb
│   └── opt_01_optimization_final.ipynb
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── model.py
│   ├── early_stopping.py
│   ├── train_eval.py
│   └── plots.py
├── splits/
│   └── fixed_cifar10_split.npz
├── results/
│   ├── figures/
│   ├── tables/
│   └── histories/
├── responsibilities/
│   └── *.md
└── legacy/
    └── *.ipynb
```
> Some notebooks are fully modular and use the shared `src/` pipeline, while others are more self-contained for focused experimentation.

## Suggested Reading Order

A good order for reading the repository is:

1. `reg_opt_00_baseline.ipynb`
2. `reg_00_modular_baseline.ipynb`
3. `reg_01_dropout.ipynb`
4. `reg_02_early_stopping.ipynb`
5. `reg_03_batchnorm.ipynb`
6. `reg_04_dropout_batchnorm.ipynb`
7. `reg_05_l2_map_gaussian_prior_dropout.ipynb`
8. `reg_06_comparison.ipynb`
9. `reg_08_l1_l2_v2.ipynb`
10. `reg_09_data_augmentation.ipynb`
11. `reg_10_adversarial_training.ipynb`
12. `reg_11_full_combo_before_optimization.ipynb`
13. `opt_01_optimization_final.ipynb`

This order reflects the actual logic of the project:

**baseline → isolated regularizers → extended regularizers → full combo → optimization tuning**

## Running the Project

The repository is primarily **notebook-driven**.

A typical workflow is:

1. Install the required dependencies:
   - `torch`
   - `torchvision`
   - `numpy`
   - `pandas`
   - `matplotlib`

2. Make sure CIFAR-10 can be downloaded locally.

3. Run the notebooks in the recommended order.

For the modular branch notebooks, reusable code lives under:

- `src/data.py`
- `src/model.py`
- `src/early_stopping.py`
- `src/train_eval.py`
- `src/plots.py`

This makes the regularization branch reproducible and easier to maintain.

## Why This Repository Is Methodologically Strong

This project is stronger than a simple “try a few techniques” repository because it is organized as a **progressive controlled study**.

It does not stop at one stage. Instead, it:

- starts from a deliberately overfitting baseline,
- compares methods one by one,
- expands the method pool,
- studies explicit parameter effects,
- studies input-space regularization,
- studies adversarial robustness,
- builds a stronger integrated pipeline,
- and then tunes optimization systematically.

So the repository is not just a set of experiments. It is a structured argument:

**baseline → isolated regularizers → extended regularizers → full combo → optimization tuning**

That progression is the main methodological contribution of the project.

## Final Takeaway

This repository shows that different techniques improve learning through different mechanisms:

- **L1 / L2** constrain parameter growth
- **Dropout** reduces co-adaptation
- **Early stopping** limits harmful late-stage fitting
- **BatchNorm** stabilizes internal feature distributions
- **Data augmentation** increases effective data diversity
- **Adversarial training** improves robustness and harder-example learning
- **Optimizer / scheduler choices** shape convergence and final performance

The most important lesson from the project is that strong performance does not come from a single trick. It comes from:

- understanding what each method does,
- testing it under controlled conditions,
- comparing it fairly,
- and combining complementary methods in a principled way.
