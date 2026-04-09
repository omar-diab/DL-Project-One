# CIFAR-10 Regularization and Optimization Study

This project investigates how different techniques affect the performance and generalization of an MLP trained on CIFAR-10.

The repository is organized around two main experiment groups:

- **Regularization:** controlled ablation study on overfitting reduction
- **Optimization / Robustness:** augmentation, adversarial training, and final combined experiments

The goal is to keep the project structure unified while preserving the distinct contribution of each experiment.

## Project Scope

The project studies how different methods influence training behavior, overfitting, and test performance under a mostly shared setup.

It includes:

- an intentionally overfitting MLP baseline,
- regularization-based experiments,
- optimization / robustness-oriented experiments,
- final combined experiments that build on earlier results.

## Controlled Experimental Setup

To keep comparisons meaningful, the experiments are designed to reuse the same core pipeline as much as possible:

- fixed train / validation / test split,
- same MLP architecture,
- same optimizer family,
- same general training pipeline,
- shared preprocessing and evaluation logic where applicable.

Some optimization notebooks introduce additional training strategies, but the common project structure is preserved through the shared `src/` codebase.

## Dataset Split

A fixed subset of CIFAR-10 is used across the project:

- **Train:** 15,000 samples
- **Validation:** 5,000 samples
- **Test:** 10,000 samples

The split is stored in:

- `splits/fixed_cifar10_split.npz`

A single shared copy of the CIFAR-10 raw batch files is kept in:

- `data/cifar-10-batches-py/`

## Model

The main model used in the project is a fully connected MLP with:

- **Input dimension:** 3072
- **Hidden layers:** [1024, 512, 256]
- **Activation:** ReLU

Optional components depending on the experiment:

- Dropout
- Batch Normalization

The output layer does not include softmax, since it is handled by `CrossEntropyLoss`.

## Experiment Groups

### 1. Regularization Experiments

These notebooks focus on reducing overfitting while keeping the baseline setting controlled.

Covered methods include:

- Baseline
- Dropout
- Early Stopping
- Batch Normalization
- Dropout + BatchNorm
- L2 + Dropout (MAP Gaussian Prior)
- L1 + L2
- Comparison / run-all evaluation

### 2. Optimization / Robustness Experiments

These notebooks extend the project with training strategies beyond the regularization ablation setting.

Covered methods include:

- Data Augmentation
- Adversarial Training
- Final combined experiment before optimization
- Final optimization experiment

## Evaluation

Experiments track:

- train / validation / test loss,
- train / validation / test accuracy,
- generalization gap:
  - train − validation,
  - train − test.

These metrics are used to compare how well each method improves generalization and reduces overfitting.

## Repository Structure

```text
DL-Project-One/
│
├─ notebooks/
│  ├─ initial_00_baseline_full.ipynb
│  ├─ reg_00_modular_baseline.ipynb
│  ├─ reg_01_dropout.ipynb
│  ├─ reg_02_early_stopping.ipynb
│  ├─ reg_03_batchnorm.ipynb
│  ├─ reg_04_dropout_batchnorm.ipynb
│  ├─ reg_05_l2_map_gaussian_prior_dropout.ipynb
│  ├─ reg_06_comparison.ipynb
│  ├─ reg_07_run_all.ipynb
│  ├─ reg_08_l1_l2_v2.ipynb
│  ├─ reg_09_data_augmentation.ipynb
│  ├─ reg_10_adversarial_training.ipynb
│  ├─ reg_11_full_combo_before_optimization.ipynb
│  └─ reg_12_optimization_final.ipynb
│
├─ src/
│  ├─ data.py
│  ├─ model.py
│  ├─ early_stopping.py
│  ├─ train_eval.py
│  └─ plots.py
│
├─ data/
│  └─ cifar-10-batches-py/
│
├─ splits/
│  └─ fixed_cifar10_split.npz
│
├─ results/
│  ├─ figures/
│  └─ tables/
│
├─ legacy/
│  └─ fatima_l1_l2_old.ipynb
│
└─ README.md
