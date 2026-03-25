# Regularization Ablation Study (CIFAR-10)

This part of the project focuses on analyzing the effect of different regularization methods on an overfitting MLP baseline trained on CIFAR-10.

Rather than presenting a standalone project, this branch represents the portion of the overall work dedicated to **regularization analysis under controlled experimental settings**.

## Scope of This Branch

The purpose of this branch is to isolate and compare several regularization techniques while keeping all other training conditions fixed.  
The goal is not to redesign the full pipeline, but to study how different regularization families affect overfitting and generalization within the shared project framework.

In this context, this branch is responsible for:
- constructing an intentionally overfitting baseline,
- applying selected regularization methods,
- comparing their impact under identical settings,
- reporting their effect on training, validation, and test performance.

## Controlled Experimental Setup

To ensure a fair comparison, all experiments use the same configuration:
- fixed train / validation / test split,
- same MLP architecture,
- same optimizer (SGD),
- same learning rate,
- same batch size,
- same epoch limit,
- same random seed.

Therefore, the only changing factor across experiments is the **regularization method**.

## Dataset Split

A fixed subset of CIFAR-10 is used throughout this branch:
- **Train:** 15,000 samples
- **Validation:** 5,000 samples
- **Test:** 10,000 samples

The split is saved and reused across all experiments to preserve consistency and reproducibility.

## Model

The branch uses a fully connected MLP with:
- **Input dimension:** 3072
- **Hidden layers:** [1024, 512, 256]
- **Activation:** ReLU

Optional components depending on the experiment:
- Dropout
- Batch Normalization

The output layer does not include softmax, since it is handled by `CrossEntropyLoss`.

## Regularization Methods Covered

This branch examines regularization methods from multiple categories:

### 1. Explicit Regularization
- **L2 Regularization**
- Implemented directly in the loss function
- Interpreted as MAP estimation with a Gaussian prior

$$
\mathcal{L}_{MAP} = \mathcal{L}_{CE} + \lambda \sum_i w_i^2
$$

### 2. Stochastic Regularization
- **Dropout**
- Randomly deactivates neurons during training
- Reduces co-adaptation between features

### 3. Implicit Regularization
- **Early Stopping**
- Stops training based on validation behavior
- Limits overfitting through training dynamics

### 4. Architectural / Normalization-Based Regularization
- **Batch Normalization**
- Stabilizes optimization and may improve generalization indirectly

## Experiments Included

The following experiment notebooks are included in this branch:

1. **Baseline**
   - no regularization
   - intended to show clear overfitting

2. **Dropout**
   - stochastic regularization

3. **Early Stopping**
   - implicit regularization

4. **Batch Normalization**
   - architectural regularization

5. **Dropout + BatchNorm**
   - combined stochastic and architectural regularization

6. **L2 + Dropout (MAP Gaussian Prior)**
   - combined explicit and stochastic regularization

7. **Comparison**
   - consolidated evaluation of all experiment results

## Evaluation

Each experiment tracks:
- train / validation / test loss,
- train / validation / test accuracy,
- generalization gap:
  - train − validation,
  - train − test.

These metrics are used to compare how effectively each method reduces overfitting.

## Branch Structure

```
beyza/
│
├─ notebooks/
│  ├─ 00_baseline.ipynb
│  ├─ 01_dropout.ipynb
│  ├─ 02_early_stopping.ipynb
│  ├─ 03_batchnorm.ipynb
│  ├─ 04_dropout_batchnorm.ipynb
│  ├─ 05_l2_map_gaussian_prior_dropout.ipynb
│  └─ 06_comparison.ipynb
│
├─ src/
│  ├─ data.py
│  ├─ model.py
│  ├─ early_stopping.py
│  ├─ train_eval.py
│  └─ plots.py
│
├─ splits/
│  └─ fixed_cifar10_split.npz
│
├─ results/
│  ├─ figures/
│  └─ tables/
│
└─ README.md
```
---

## Key Insight

Different regularization methods act in fundamentally different ways:

- Explicit → constrains weights  
- Stochastic → introduces noise  
- Implicit → limits training dynamics  
- Architectural → changes representation behavior  

This branch compares all of them under identical conditions.
