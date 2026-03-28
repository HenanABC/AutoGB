# AutoGB

**AutoGB** is a research prototype for **parameter-free adaptive granular-ball learning** on tabular data.

It is designed as an **interpretable, risk-aware, and selective prediction framework**, rather than a purely accuracy-driven classifier.

Unlike conventional models that directly learn a global decision boundary, AutoGB first constructs a set of **granular balls** to represent local data structure, then performs **feature transformation + meta-level prediction + calibration**.

---

## Overview

flowchart LR
    A[Input Data] --> B[Granular Ball Partition]
    B --> C[Ball-level Statistics]
    C --> D[Feature Mapping]
    D --> E[Meta Predictor]
    E --> F[Probability Calibration]
    F --> G[Risk-aware Output]
    G --> H[Risk-Coverage Evaluation]

---

## Motivation

Most machine learning models focus on maximizing accuracy. However, in many real-world scenarios:

* incorrect predictions are costly
* uncertainty must be considered
* interpretability is required
* abstaining can be safer than forcing a decision

AutoGB addresses this by shifting from **hard classification** to **risk-aware modeling**.

---

## Core Idea

AutoGB separates the learning process into three stages:

1. **Local structure construction (granular balls)**
2. **Feature transformation based on local regions**
3. **Standard supervised prediction + calibration**

This design enables:

* interpretable region-based modeling
* flexible integration with existing models
* support for selective prediction

---

## Mathematical Formulation

Given a dataset

$$
\mathcal{D} = {(x_i, y_i)}_{i=1}^N, \qquad x_i \in \mathbb{R}^d
$$

AutoGB constructs a set of granular balls

$$
\mathcal{B} = {B_1, B_2, \dots, B_M}
$$

via recursive partitioning of the input space.

### Granular Ball Statistics

For each ball $B_m$:

**Center**

$$
c_m = \frac{1}{|B_m|}\sum_{x_i \in B_m} x_i
$$

**Radius**

$$
r_m = \max_{x_i \in B_m} |x_i - c_m|_2
$$

**Local label ratio (binary case)**

$$
\rho_m = \frac{1}{|B_m|}\sum_{(x_i, y_i)\in B_m} \mathbf{1}(y_i = 1)
$$

These quantities summarize local geometric and statistical structure.

### Feature Transformation

For a new sample $x$, AutoGB builds a structure-aware feature vector:

$$
\phi(x) = \mathrm{Feature}(x; \mathcal{B})
$$

This transformation may include:

* distance to nearest ball center
* ball radius
* local purity $\rho_m$
* ball size / density
* other region-level statistics

This step converts raw input into a **granular-structure-aware representation**.

### Meta-Level Prediction

A standard model is applied:

$$
z(x) = f_\theta(\phi(x))
$$

where $f_\theta$ can be logistic regression, MLP, or other classifiers.

### Probability Calibration

To improve reliability:

$$
\hat{p}(x) = g(z(x))
$$

where $g(\cdot)$ is a calibration function such as sigmoid calibration.

### Selective Prediction

AutoGB supports **selective prediction analysis**:

* samples are ranked by confidence or risk
* only high-confidence samples are retained
* performance is evaluated as a function of coverage

This produces a **risk-coverage trade-off curve**, rather than a fixed decision rule.

---

## Highlights

* Parameter-free adaptive partitioning
* Interpretable region-based representation
* Risk-aware prediction pipeline
* Built-in support for selective prediction
* Fully reproducible notebook prototype

---

## Repository Contents

This repository contains a **single notebook implementation**:

```text
AutoGB_V4.ipynb
```

The notebook includes:

* end-to-end pipeline
* multi-seed evaluation
* ablation study
* risk-coverage analysis

---

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Open and run:

```bash
jupyter notebook AutoGB_V4.ipynb
```

---

## Experimental Insights

Current results indicate that:

* AutoGB performs well on **low-dimensional tabular data**
* provides **interpretable local structure**
* naturally supports **risk-based rejection**

However:

* performance degrades on **high-dimensional datasets**
* local Euclidean structure becomes less informative

---

## Scope

AutoGB is a **general framework for tabular learning**, not tied to a specific domain.

Potential applications include:

* healthcare risk modeling
* predictive maintenance
* AIOps monitoring
* anomaly detection
* financial risk analysis

This repository focuses on **framework demonstration**, not domain-specific optimization.

---

## Limitations

* Not suitable for high-dimensional feature spaces
* Heuristic ball partition strategy
* Not designed as a SOTA classifier

---

## Future Work

* learned partition strategies
* integration with deep representations
* conformal prediction / risk control
* stronger uncertainty estimation

---

## Research Positioning

AutoGB is positioned as a **framework-level approach** that connects:

* local structure modeling via granular balls
* feature-based learning
* risk-aware prediction
* selective decision making

It is not a standalone classifier, but a **modular pipeline for interpretable and uncertainty-aware learning**.

---

## Author

Henan Zhao

---

## License

MIT License

