---
name: WM-GP Research Paper Plan
overview: "A plan for writing a Science/Nature-caliber research paper presenting WM-GP: a unified algorithmic framework for visual working memory that uses sparse variational Gaussian processes to model encoding, goal-driven maintenance, and retrieval as dynamic Bayesian inference over a continuous location-color function space."
todos:
  - id: schematic-fig
    content: "Create Figure 1: conceptual schematic of the WM-GP pipeline (encoding -> maintenance -> retrieval) with mathematical notation"
    status: completed
  - id: polish-figures
    content: Polish all existing figures to publication quality (vector format, consistent styling, proper labels, Nature/Science figure size constraints)
    status: completed
  - id: human-data-overlay
    content: Obtain and overlay actual human behavioral data from cited papers (Van den Berg 2012, Chunharas 2022) for quantitative validation
    status: completed
  - id: param-sensitivity
    content: Run systematic parameter sensitivity analysis (inducing grid size, beta, lengthscale) and create supplementary figures
    status: in_progress
  - id: model-selection
    content: Add formal BIC/AIC model comparison to the IL/EP/SA/VP analysis in set_size_model_comparison.py
    status: completed
  - id: write-intro
    content: "Write Introduction section (~800 words): WM debate, process-level gap, WM-GP contribution"
    status: in_progress
  - id: write-results
    content: "Write Results section (~2000 words): organized around Figures 2-4 with quantitative claims"
    status: pending
  - id: write-discussion
    content: "Write Discussion section (~1200 words): unification argument, neural plausibility, predictions, limitations"
    status: pending
  - id: write-methods
    content: "Write Supplementary Methods (~3000 words): full mathematical formalism, parameter tables, statistical details"
    status: pending
  - id: write-abstract
    content: Write Abstract (~150 words) and finalize title
    status: pending
  - id: bibliography
    content: Audit and complete references.bib -- ensure all cited works are included and properly formatted
    status: pending
isProject: false
---

# WM-GP Research Paper Plan

## Positioning and Core Thesis

**Central claim:** Visual working memory can be understood as a single continuous Bayesian inference process over a 2D location-feature function, where a sparse variational Gaussian process naturally gives rise to capacity limits, precision trade-offs, attentional modulation, and inter-item interference -- without needing separate mechanisms for slots, resources, or guessing.

**Why this is Science/Nature-worthy:**

- **Unification**: One principled model with a shared set of parameters explains 7+ distinct empirical phenomena that previously required separate ad-hoc models (IL, EP, SA, VP, mixture models)
- **Process-level account**: Unlike descriptive models (Zhang & Luck 2008; Van den Berg et al. 2012), WM-GP specifies *how* encoding, maintenance, and retrieval *actually work* as stages of Bayesian inference
- **Dynamic resource allocation**: Attention is not a separate module but a natural reweighting of the variational objective, explaining retrocue benefits, flexible re-allocation, and cue timing effects
- **Formal elegance**: Built on well-understood GP theory (kernel composition, variational inference, inducing points) -- the entire framework has clear mathematical grounding

---

## Paper Structure (Science format: ~4500 words main text + Supplementary)

### Title (working)

*"Working memory as dynamic Bayesian inference over continuous feature space"*

or

*"A unified Gaussian process framework for encoding, maintenance, and retrieval in visual working memory"*

### Abstract (~150 words)

Key points to cover:

- Visual WM has been modeled as discrete slots or divisible resources, but neither explains the full breadth of behavioral phenomena
- Introduce WM-GP: memory as a continuous function over location-feature space, maintained via sparse variational Gaussian process inference
- Encoding = fitting a GP posterior to noisy sensory observations
- Maintenance = self-rehearsal against a forgetting pressure (KL toward prior), with goal-driven attention reweighting the variational objective
- Retrieval = evaluating the posterior at a queried location
- Single framework reproduces: set-size effects, precision gradients, mixture-model signatures, retrocue benefits, retrocue x set-size interaction, cue timing effects, flexible re-cueing, and encoding-time x distance-dependent repulsion bias
- Quantitatively matches or outperforms IL/EP/SA/VP resource models on mixture statistics

---

### 1. Introduction (~800 words)

**Opening** (2 paragraphs):

- The paradox of WM: limited capacity but flexible, goal-driven, and precise. Decades of debate between slot-based (Luck & Vogel 1997; Zhang & Luck 2008) and resource-based (Bays & Husain 2008; Ma et al. 2014) accounts.
- Neither class of models provides a *process-level* account of how representations are encoded, actively maintained over time, and retrieved. They describe *what* the error distributions look like, not *how* they arise from computational operations on neural representations.

**The gap** (1 paragraph):

- Missing: a single algorithmic framework that explains (a) how encoding noise maps to representations, (b) how maintenance can degrade or sharpen memories, (c) how attention dynamically reallocates resources, (d) how inter-item interference creates systematic biases -- all from the same set of principles. Recent calls for such integration (Oberauer et al. 2018 benchmark; Awh et al. 2025; Bays 2024).

**Our contribution** (2 paragraphs):

- Propose WM-GP: working memory as Bayesian inference over a *continuous function* on a circular location x color space. A sparse variational GP provides the computational substrate.
- Key insight: the inducing-point representation naturally has limited capacity (finite inducing points), precision varies with load (kernel bandwidth shared across items), attention acts by reweighting the ELBO during maintenance, and inter-item bias arises from kernel smoothness. No separate "slot" or "resource" mechanisms needed.

**Key citations from** [`docs/references.bib`](docs/references.bib):

- Luck & Vogel 1997, Zhang & Luck 2008, Van den Berg et al. 2012, Bays & Husain 2008, Ma et al. 2014, Oberauer et al. 2018, Chunharas et al. 2022, Souza & Oberauer 2016, Radmard et al. 2025

---

### 2. Results (~2000 words, organized around 4-5 main figures)

#### Figure 1: Model Schematic and Mechanism

A conceptual overview figure (no code equivalent yet -- needs to be created):

- **Panel A**: The WM-GP computational pipeline: Stimulus array --> Noisy encoding (weighted samples on torus) --> GP posterior (2D surface) --> Maintenance (self-rehearsal + KL + attention) --> Retrieval (posterior slice at probed location)
- **Panel B**: The 2D GP surface for N=4 items after encoding (from existing `gp_surface_N=4.png`)
- **Panel C**: Inducing point grid and how it constrains capacity
- **Panel D**: The attention gain function and how it reshapes the maintenance ELBO

#### Figure 2: Set-Size Effects and Model Comparison

Uses data from [`set_size_model_comparison.py`](src/set_size_model_comparison.py) and existing outputs:

- **Panel A**: Error distributions by set size (N=1,2,4,6) -- the hallmark broadening and flattening. Source: `error_distributions_combined.png`
- **Panel B**: MAE and SD vs set size. Source: [`visualizations/set_size_results.csv`](visualizations/set_size_results.csv)
- **Panel C**: Mixture statistics (w, CSD) vs the 4 benchmark models (IL, EP, SA, VP). Source: [`visualizations/model_comparison_figure4a_style.png`](visualizations/model_comparison_figure4a_style.png), [`visualizations/mixture_summary_empirical.csv`](visualizations/mixture_summary_empirical.csv)

**Narrative**: WM-GP naturally produces (1) increasing errors with set size, (2) decreasing mixture weight w (apparent "guess rate"), and (3) increasing CSD -- matching the VP model's predictions most closely, but arising from a *process* model rather than a descriptive distribution.

#### Figure 3: Goal-Driven Maintenance and Retrocue Effects

Uses data from [`validation.py`](src/validation.py), [`validation_retrocue_setsize.py`](src/validation_retrocue_setsize.py), [`validation_cue_timing.py`](src/validation_cue_timing.py), [`validation_cue_flexibility.py`](src/validation_cue_flexibility.py):

- **Panel A**: GP surface before vs after retrocue (existing `retrocue_comparison_N=4.png`) -- visual demonstration of resource reallocation
- **Panel B**: Retrocue benefit (neutral - cued MAE) with significance. Source: `retrocue_benefit_*.png`
- **Panel C**: Retrocue benefit x set size interaction (larger at N=4 vs N=2). Source: `retrocue_setsize_benefit.png`
- **Panel D**: Cue timing effect (earlier cue = larger benefit). Source: `cue_timing_benefit.png`
- **Panel E**: Cue flexibility (sequential cues shift benefit dynamically). Source: `cue_flexibility.png`

**Narrative**: The attention mechanism reweights the maintenance ELBO, causing the GP posterior to sharpen around the cued item at the expense of uncued items. This naturally produces: retrocue benefits, set-size x retrocue interaction (more items = more to lose from not cueing), timing effects (more maintenance epochs with cue = more sharpening), and flexible reallocation (cue switch = benefit transfer).

#### Figure 4: Inter-Item Repulsion Bias

Uses data from [`validation_3d_bias.py`](src/validation_3d_bias.py) and [`validation.py`](src/validation.py):

- **Panel A**: Repulsion bias vs color distance (the inverted-U pattern). Source: `bias_effect.png`
- **Panel B**: 3D interaction of encoding time x color distance on bias (replicating Chunharas et al. 2022 Exp 4). Source: `bias_3d_plot_deg.png`, `bias_line_plot_deg.png`
- **Panel C**: The divisive normalization mechanism and how it produces repulsion from the GP surface

**Narrative**: With spatial divisive normalization (Carandini & Heeger 2012) at readout, the model produces repulsion bias that (1) increases with color proximity (up to a point), and (2) is stronger at shorter encoding times -- matching Chunharas et al. 2022 quantitatively. The bias emerges from kernel smoothing in the joint location-color space: nearby items create overlapping activation that, after normalization, shifts the peak away from the distractor.

#### Figure 5 (optional): Process-Level Dynamics

- **Panel A**: Training loss trajectories (encoding + maintenance). Source: `training_trajectories_N=*.png`
- **Panel B**: GP surface evolution as animated snapshots (from GIFs, select key frames)
- **Panel C**: Lengthscale adaptation during learning

---

### 3. Discussion (~1200 words)

**Key points:**

- **Unification through process**: WM-GP bridges the slots-vs-resources debate by showing both phenomena emerge from the same GP inference. "Slots" = inducing points; "resources" = kernel bandwidth shared across items; "guessing" = flat posterior regions.
- **Relation to neural implementation**: The GP posterior surface could map to population-coded neural activity (Bays 2014). Inducing points are analogous to neural tuning centers. Maintenance ELBO optimization is analogous to attractor dynamics.
- **Predictions**: The framework makes novel, testable predictions:
  1. Retrocue benefit should depend on maintenance duration in a specific (not just qualitative) way
  2. Inter-item bias should interact with retrocue in predictable ways (cueing reduces bias for cued item)
  3. Set-size effects should show smooth degradation, not abrupt capacity limits
- **Limitations**: No explicit temporal dynamics (epochs are abstract time steps), no multi-feature binding beyond 2D, fixed kernel family
- **Future directions**: Extension to multi-feature binding (higher-dimensional GPs), neural GP correspondence, temporal kernels for sequence memory, deep kernel learning for richer representations

---

### 4. Methods (Supplementary Materials, ~3000 words)

Can be extensive. Organized as:

#### 4.1 Model Specification

Formal mathematical presentation of:

- The GP prior: $f \sim \mathcal{GP}(m(\mathbf{x}), k(\mathbf{x}, \mathbf{x}'))$ where $\mathbf{x} = (\ell, c) \in [-180, 180)^2$
- Kernel: $k(\mathbf{x}, \mathbf{x}') = \sigma^2 \cdot k_{\text{per}}(\ell, \ell'; \lambda_\ell) \cdot k_{\text{per}}(c, c'; \lambda_c)$ with period 360
- Sparse variational approximation: inducing points $\mathbf{Z}$, variational distribution $q(\mathbf{u}) = \mathcal{N}(\mathbf{m}, \mathbf{S})$
- Reference to GPyTorch implementation: [`src/gp_model.py`](src/gp_model.py)

#### 4.2 Encoding Phase

- Noisy observation model from [`src/generator.py`](src/generator.py): weighted samples with Gaussian importance weights
- ELBO optimization: $\mathcal{L}_{\text{enc}} = \mathbb{E}_{q(f)}[\sum_i w_i \log p(y_i | f(\mathbf{x}_i))] - \text{KL}(q(\mathbf{u}) \| p(\mathbf{u}))$
- Hyperparameters: inducing grid size, lengthscales, encoding LR, noise variance

#### 4.3 Maintenance Phase

- Self-rehearsal targets: $\tilde{y}_g = \mathbb{E}_{q(f)}[f(\mathbf{x}_g)]$ (frozen posterior mean on dense grid)
- Attention-weighted ELBO: $\mathcal{L}_{\text{maint}} = -\frac{1}{|\mathcal{G}|}\sum_{g} a_g \mathbb{E}_q[\log p(\tilde{y}_g | f(\mathbf{x}_g))] + \beta \text{KL}(q(\mathbf{u}) \| p(\mathbf{u}))$
- Attention gain from [`src/attention_mechanisms.py`](src/attention_mechanisms.py): $a(\ell) = 1 + (G - 1)\exp(-d(\ell, \ell_{\text{cued}})^2 / 2\sigma_s^2)$
- The $\beta$ parameter controls forgetting rate: larger $\beta$ = faster decay toward prior

#### 4.4 Retrieval Phase

- MAP retrieval: $\hat{c} = \arg\max_c \mathbb{E}_{q(f)}[f(\ell_{\text{probe}}, c)]$
- Optional divisive normalization at readout (for bias experiments)
- Probabilistic retrieval variant (posterior sampling)

#### 4.5 Simulation Parameters

Table of all config parameters across the three configs in [`src/config/`](src/config/):

- `config_set_size.yaml`: inducing_grid=10, encoding_epochs=100, maintenance_epochs=100, beta=1
- `config_retrocue.yaml`: attended_gain=20, maintenance_epochs=200, beta=5
- `config_bias.yaml`: loc_encoding_noise_std=20, normalization parameters

#### 4.6 Comparison Models

Fisher-information mapping, IL/EP/SA/VP log-densities, MLE fitting procedure from [`src/set_size_model_comparison.py`](src/set_size_model_comparison.py)

#### 4.7 Statistical Tests

Paired t-tests for retrocue benefits, bootstrap SEM for mixture fits, cross-condition comparisons

---

## Figures to Create (new) vs Reuse (existing)

| Figure | Status | Source |
|--------|--------|--------|
| Fig 1: Schematic | **New** -- needs a clean conceptual diagram |
| Fig 2A: Error distributions | Exists: `error_distributions_combined.png` (may need polish) |
| Fig 2B: MAE vs set size | Exists: `set_size_effect.png` (may need polish) |
| Fig 2C: Model comparison | Exists: `model_comparison_figure4a_style.png` |
| Fig 3A: GP surface before/after cue | Exists: `retrocue_comparison_N=4.png` |
| Fig 3B: Retrocue benefit | Exists: `retrocue_benefit_*.png` |
| Fig 3C: Retrocue x set size | Exists: `retrocue_setsize_benefit.png` |
| Fig 3D: Cue timing | Exists: `cue_timing_benefit.png` |
| Fig 3E: Cue flexibility | Exists: `cue_flexibility.png` |
| Fig 4A: Bias vs distance | Exists: `bias_effect.png` |
| Fig 4B: 3D bias | Exists: `bias_3d_plot_deg.png` / `bias_line_plot_deg.png` |
| Fig 4C: Normalization diagram | **New** -- needs a clean diagram |

---

## What Needs to Be Done Before Writing

### Code / Simulation Gaps

1. **Formal parameter sensitivity analysis**: Systematically vary key parameters (inducing grid size, beta, lengthscale) and show robustness
2. **Quantitative comparison to human data**: Currently validation is simulation-only. For a top journal, overlaying actual human data (from cited papers where available) would strengthen the case enormously
3. **BIC/AIC model comparison**: Add formal model selection criteria beyond visual RMSE to the IL/EP/SA/VP comparison
4. **Confidence intervals / bootstrap**: Ensure all reported statistics have proper error bars and p-values

### Writing Deliverables

1. **Main text** (~4500 words, Science format or ~5000 Nature format)
2. **Supplementary Materials** (~3000 words: full math, parameter tables, additional figures)
3. **Publication-quality figures** (vector format, consistent style, proper axis labels)
4. **Updated references.bib** -- ensure all cited papers are in the bibliography

---

## Key Differentiators for Reviewer Appeal

- **Parsimony**: One model, ~10 parameters, 7+ phenomena. Compare to needing separate IL/EP/SA/VP/mixture models.
- **Process transparency**: Every step (encoding, maintenance, retrieval) is inspectable -- you can literally watch the GP surface evolve.
- **Falsifiability**: Makes specific quantitative predictions about retrocue timing, bias-cue interaction, and capacity that differ from existing models.
- **Formal Bayesian foundation**: Grounded in well-understood variational inference theory, not ad-hoc neural network architectures.
