# Multi-Level Bidirectional Decoder Interaction for Uncertainty-Aware Breast Ultrasound Analysis

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.13](https://img.shields.io/badge/TensorFlow-2.13-orange.svg)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Official Implementation**

</div>

---

## Abstract

Breast ultrasound interpretation requires simultaneous lesion segmentation and tissue classification, yet conventional multi-task learning approaches suffer from task interference and rigid coordination strategies. We propose a framework that addresses these limitations through multi-level decoder interaction and uncertainty-aware adaptive coordination. Task Interaction Modules operate at four decoder resolutions, establishing bidirectional communication during spatial reconstruction. Uncertainty Proxy Attention uses feature activation variance for efficient per-instance adaptive weighting. Evaluation on BUSI and BUSI-WHU datasets demonstrates 74.50% IoU and 90.60% accuracy, outperforming encoder-sharing methods by 1.6–5.6% IoU.

---

## Architecture Overview

<div align="center">
  <img src="https://github.com/C-loud-Nine/Uncertainty-Aware-Multi-Level-Decoder-Interaction/blob/main/arch.png" alt="Architecture Overview" width="95%">
  <br>
  <em><strong>Figure 1:</strong> Overall architecture with multi-level decoder interaction. TIM enables bidirectional feature exchange at four decoder levels (D₁–D₄). UPA adaptively modulates information flow at each level based on feature activation variance. Multi-scale fusion modules augment encoder features to handle lesion size variation (5–40mm).</em>
</div>

---

## Key Contributions

### 1️⃣ Multi-Level Decoder Interaction

**Problem:** Conventional encoder-sharing MTL restricts task communication to abstract feature extraction. Once tasks enter separate decoders, representations diverge, preventing exploitation of complementary information during spatial reconstruction.

**Solution:** Task Interaction Modules (TIM) at **four decoder levels** enable progressive bidirectional refinement:

<div align="center">

| Decoder Level | Resolution | Channel Dim | Interaction Focus |
|:-------------:|:----------:|:-----------:|:------------------|
| **D₁** | 14×14 | 384 | Coarse semantic context |
| **D₂** | 28×28 | 192 | Mid-level feature fusion |
| **D₃** | 56×56 | 96 | Fine-grained spatial detail |
| **D₄** | 112×112 | 48 | Boundary-level precision |

</div>

**Bidirectional pathways:**
- **Segmentation → Classification:** Attention-weighted pooling extracts boundary-aware spatial context, gated addition prevents error propagation
- **Classification → Segmentation:** Semantic priors broadcast spatially, multiplicative modulation (μ = 1 + τ·gate·context) enables selective enhancement

**Algorithm 1: Task Interaction Module**
```
Input: Decoder features D_ℓ ∈ ℝ^(B×H×W×C), Classification features f_clf^ℓ ∈ ℝ^(B×256)
Output: Enhanced features (D_ℓ^enh, f_clf^ℓ,enh)

// Segmentation → Classification
1: f_seg^ctx ← Dense₂₅₆(GAP(Conv₁ₓ₁(D_ℓ)))
2: g_clf ← σ(MLP(f_seg^ctx))
3: f_clf^ℓ,enh ← f_clf^ℓ + g_clf ⊙ f_seg^ctx

// Classification → Segmentation
4: f_clf^spat ← Reshape(Dense_C(f_clf^ℓ), [1,1,C])
5: g_seg ← σ(MLP(GAP(D_ℓ)))
6: μ_ℓ ← 1 + 0.7 · g_seg ⊙ f_clf^spat
7: D_ℓ^enh ← D_ℓ ⊙ μ_ℓ

Return (D_ℓ^enh, f_clf^ℓ,enh)
```

---

### 2️⃣ Uncertainty Proxy Attention

**Problem:** Instance heterogeneity—cases with clear boundaries benefit from strong task interaction, while ambiguous cases (posterior shadowing, heterogeneous textures) require conservative weighting.

**Solution:** Adaptive coordination via feature activation variance as an efficient uncertainty proxy:

**Algorithm 2: Uncertainty Proxy Attention**
```
Input: Base features (D_ℓ, f_clf^ℓ), Enhanced features (D_ℓ^enh, f_clf^ℓ,enh)
Output: Final features (D_ℓ^final, f_clf^ℓ,final)

// Estimate uncertainties
1: u_seg ← 𝔼_c[Var_{h,w}(D_ℓ,c^enh)]           // Spatial variance per channel
2: u_clf ← Var(f_clf^ℓ,enh)                    // Feature vector variance
3: ũ_seg ← u_seg / (mean(u_seg) + ε)           // Normalize to comparable scales
4: ũ_clf ← u_clf / (mean(u_clf) + ε)

// MLP-based adaptive weighting
5: [ω_seg, ω_clf] ← Softmax(MLP₂(ReLU(MLP₃₂([ũ_seg, ũ_clf]))))

// Residual interpolation
6: D_ℓ^final ← D_ℓ + ω_seg · (D_ℓ^enh - D_ℓ)
7: f_clf^ℓ,final ← f_clf^ℓ + ω_clf · (f_clf^ℓ,enh - f_clf^ℓ)

Return (D_ℓ^final, f_clf^ℓ,final)
```

**Key properties:**
- ω = 0: Preserve base features (reject uncertain enhancement)
- ω = 1: Fully adopt enhanced features (trust task interaction)
- Softmax creates task competition: higher uncertainty in one task increases reliance on the other

---

### 3️⃣ Multi-Scale Context Fusion

Breast ultrasound lesions exhibit substantial size variation (5–40mm diameter). Multi-scale context fusion employs parallel dilated separable convolutions with instance-adaptive scale weighting:

**Algorithm 3: Multi-Scale Context Fusion**
```
Input: Feature map X ∈ ℝ^(B×H×W×C)
Output: Fused features Y ∈ ℝ^(B×H×W×C)

// Multi-scale feature extraction
1: F₁ ← SepConv₃ₓ₃^(r=1)(X)    // 3×3 effective receptive field
2: F₂ ← SepConv₃ₓ₃^(r=2)(X)    // 5×5 effective receptive field
3: F₄ ← SepConv₃ₓ₃^(r=4)(X)    // 9×9 effective receptive field

// SE-inspired scale attention
4: α ← Softmax(MLP₃(ReLU(MLP_{C/8}(GAP(X)))))

// Weighted fusion with residual
5: Y ← Conv₁ₓ₁(∑ᵢ αᵢ · Fᵢ) + X

Return Y
```

Unlike standard ASPP, separable convolutions reduce parameters while softmax normalization enforces scale competition, enabling emphasis on appropriate receptive fields per lesion.

---

## Learned Behavior Analysis

<div align="center">
  <img src="https://github.com/C-loud-Nine/Uncertainty-Aware-Multi-Level-Decoder-Interaction/blob/main/tim_upa.png" alt="TIM and UPA Analysis" width="90%">
  <br>
  <em><strong>Figure 2:</strong> (Left) Task interaction magnitudes across decoder levels reveal consistent segmentation-to-classification dominance, peaking at D₃. Dense spatial features inject richer context into compact classification vectors. (Right) UPA weight distributions show adaptive task balancing: early stages (D₁, D₂) trust TIM-enhanced features where global semantic context is reliable; deeper levels (D₃, D₄) favor base features, preserving fine-grained spatial detail during boundary reconstruction.</em>
</div>

**Key observations:**
- **Segmentation dominance:** Spatially dense tensors (H×W×C) provide richer priors to compact vectors (256-D) than the reverse
- **D₃ peak:** Intermediate-scale features (28×28) achieve optimal semantic-spatial balance
- **UPA adaptation:** Classification features receive higher weight at D₁/D₂ (semantic context), segmentation dominates at D₃/D₄ (boundary detail)

---

## Datasets

### BUSI (Breast Ultrasound Images Dataset)

<div align="center">

| Category | Images | Patients | Characteristics |
|:---------|:------:|:--------:|:----------------|
| **Normal** | 133 | - | No lesions, null segmentation masks |
| **Benign** | 437 | - | Fibroadenomas, cysts, avg. size 5–20mm |
| **Malignant** | 210 | - | Invasive carcinomas, avg. size 10–40mm |
| **Total** | **780** | **600** | Variable resolution, annotated by experts |

</div>

**Source:** [Mendeley Data](https://data.mendeley.com/datasets/wmy84gzngw/1)  
**Reference:** Al-Dhabyani et al., "Dataset of breast ultrasound images," *Data in Brief*, 2020  
**Annotations:** Multiple masks per image merged via pixel-wise maximum  
**Challenges:** Posterior acoustic shadowing, speckle noise, boundary ambiguity

### BUSI-WHU (Extended Dataset)

<div align="center">

| Category | Images | Characteristics |
|:---------|:------:|:----------------|
| **Benign** | 561 | Larger, more diverse phenotypes |
| **Malignant** | 387 | Heterogeneous textures, irregular margins |
| **Total** | **927** | Different acquisition protocols |

</div>

**Source:** [Mendeley Data](https://data.mendeley.com/datasets/...)  
**Reference:** Huang et al., "BUSI-WHU: Breast cancer ultrasound image dataset," *Mendeley Data V3*, 2025  
**Purpose:** Cross-dataset validation to assess architectural robustness

**Data preprocessing:**
- Patient-level stratified splitting (60% train / 15% validation / 25% test)
- Resize to 224×224, normalize to [0,1]
- Augmentation: horizontal/vertical flip, rotation ±15° (3× expansion per sample)

---

## Experimental Results

### Quantitative Performance

<div align="center">

**Table 1: Performance on BUSI Dataset (780 images)**

| Method | Type | Segmentation | | Classification | |
|:-------|:----:|:------------:|:---:|:-------------:|:---:|
| | | **IoU (%)** | **Dice (%)** | **Acc (%)** | **F1 (%)** |
| U-Net† | CNN | 66.80 | 80.05 | 86.50 | 84.70 |
| Attention U-Net† | CNN | 68.20 | 81.05 | 87.80 | 86.00 |
| UNet++ | CNN | 69.50 | 81.95 | – | – |
| TransUNet | Transformer | 70.30 | 82.55 | – | – |
| Swin-UNet | Transformer | 71.50 | 83.40 | – | – |
| MISSFormer | Transformer | 72.80 | 84.25 | – | – |
| MTAN | MTL | 68.90 | 81.60 | 87.20 | 85.40 |
| MTANet | MTL | 72.10 | 83.80 | 89.30 | 87.60 |
| MTL-OCA | MTL | 72.90 | 84.30 | 89.90 | 88.20 |
| **Proposed** | **MTL** | **74.50** | **82.25** | **90.60** | **89.84** |

<em>† Segmentation models extended with classification heads for multi-task evaluation</em>

</div>

<div align="center">

**Table 2: Performance on BUSI-WHU (927 images)**

| Method | IoU (%) | Dice (%) | Acc (%) | F1 (%) |
|:-------|:-------:|:--------:|:-------:|:------:|
| MISSFormer | 84.60 | 91.60 | – | – |
| MTL-OCA | 84.50 | 91.60 | 94.50 | 92.90 |
| **Proposed** | **86.40** | **92.70** | **95.00** | **94.74** |

</div>

**Key achievements:**
- **BUSI:** +1.6% IoU over best MTL baseline (MTL-OCA), +1.7% over best transformer (MISSFormer)
- **BUSI-WHU:** +1.9% IoU, +0.5% accuracy improvement with consistent performance ordering
- **Generalization:** Relative advantages preserved across different acquisition protocols

---

### Ablation Study

<div align="center">

**Table 3: Component Contributions on BUSI Dataset**

| Components | | | Segmentation | | | Classification | | |
|:----------:|:---:|:---:|:------------:|:-------:|:------:|:--------------:|:-------:|:-------:|
| **HMSF** | **TIM** | **UPA** | **IoU (%)** | **Dice (%)** | **Sens (%)** | **Acc (%)** | **F1 (%)** | **AUC (%)** |
| – | – | – | 67.43 | 75.48 | 75.06 | 84.62 | 82.33 | 94.90 |
| ✓ | – | – | 69.95 | 77.63 | 79.12 | 88.89 | 88.34 | 96.30 |
| – | ✓ | – | 69.20 | 77.30 | 79.83 | 86.32 | 84.00 | 94.41 |
| – | ✓ | ✓ | 69.74 | 78.55 | **83.07** | 88.89 | 88.34 | 97.31 |
| **✓** | **✓** | **✓** | **74.50** | **82.25** | 80.87 | **90.60** | **89.83** | **97.66** |

</div>

**Analysis:**
- **HMSF** (Multi-scale fusion): +2.52% IoU, addresses lesion size variation and appearance heterogeneity
- **TIM** (Task interaction): +1.77% IoU, bidirectional communication during spatial reconstruction
- **UPA** (Uncertainty): 94.41% → 97.31% AUC, adaptive per-instance coordination reduces over-commitment to unreliable signals
- **Full system**: +7.07% IoU over baseline, demonstrating genuine functional synergy

---

### Qualitative Results

<div align="center">
  <img src="https://github.com/C-loud-Nine/Uncertainty-Aware-Multi-Level-Decoder-Interaction/blob/main/mask.png" alt="Qualitative Segmentation Results" width="95%">
  <br>
  <em><strong>Figure 3:</strong> Qualitative segmentation comparison on BUSI dataset. The proposed method demonstrates improved boundary localization in posterior shadowing regions (row 1), better handling of heterogeneous textures (row 2), and accurate delineation of irregular margins (row 3). Encoder-sharing baselines (U-Net, Attention U-Net) struggle with spatial detail recovery, while transformer methods (Swin-UNet) produce over-smoothed boundaries. Our decoder-level interaction preserves fine-grained spatial structure while incorporating semantic constraints.</em>
</div>

---

## Repository Structure
```
decoder-task-interaction/
│
├── config.py                    # Hyperparameters and training configuration
│   ├── Model architecture (decoder channels, dropout)
│   ├── Training setup (learning rate, batch size, epochs)
│   ├── Data split ratios (60/15/25)
│   └── Loss weights and callback settings
│
├── data_loader.py              # Dataset loading and preprocessing
│   ├── load_image_and_merge_masks()  # Merge multiple annotations
│   ├── augment_dataset()              # 3× augmentation per sample
│   └── prepare_datasets()             # Patient-level stratification
│
├── loss.py                     # Multi-task loss formulation
│   ├── enhanced_lesion_focus_loss()   # Focal Tversky + boundary + texture
│   ├── efficient_boundary_detection() # Curvature-based boundary loss
│   ├── efficient_texture_consistency() # Sobel gradient variance
│   └── enhanced_multi_modal_focal_loss() # Classification focal CE
│
├── modules.py                  # Custom neural network layers
│   ├── TaskInteractionModule          # Bidirectional Seg↔Clf
│   ├── UncertaintyGuidedAttention     # Variance-based adaptive weighting
│   ├── HCTMultiScaleFusion           # Dilated separable convolutions
│   ├── HCTAttentionGate              # Skip connection attention
│   ├── HCTResidualBlock              # Residual + multi-scale + attention
│   └── HCTDualPathAttention          # Channel + spatial attention
│
├── model.py                    # Model architecture
│   ├── enhanced_hct_model()           # Main model builder
│   ├── build_decoder_block()          # Decoder construction
│   └── build_tim_uga()                # TIM + UPA integration
│
├── train.py                    # Training pipeline
│   ├── CompositeMetric                # 0.7×IoU + 0.3×Acc monitoring
│   ├── AdaptiveLossWeights            # Dynamic weight adjustment
│   ├── CosineDecayScheduler           # Learning rate annealing
│   └── Training loop (warmup + main)
│
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## Implementation Details

### Loss Formulation

**Multi-task objective:**
```
ℒ_total = 0.80 · ℒ_seg + 0.20 · ℒ_clf
```

**Segmentation loss:**
```
ℒ_seg = ℒ_FocalTversky + 0.25 · ℒ_boundary + 0.15 · ℒ_texture

where:
  ℒ_FocalTversky: α=0.3 (FP penalty), β=0.7 (FN penalty), γ=0.75 (focal)
  ℒ_boundary: 𝔼[|K_curv ⊗ M - K_curv ⊗ M̂|], K_curv = (x²-y²)e^(-r²/2σ²)
  ℒ_texture: |Var(S_x ⊗ M) - Var(S_x ⊗ M̂)|, S_x = Sobel horizontal
```

**Classification loss:**
```
ℒ_clf = FocalCrossEntropy(γ=2.0)
```

### Training Configuration

<div align="center">

| Hyperparameter | Value | Purpose |
|:---------------|:-----:|:--------|
| **Batch size** | 8 | Memory-efficient training |
| **Initial LR** | 3×10⁻⁴ | Stable convergence |
| **LR schedule** | Cosine annealing | Smooth decay to 1.5×10⁻⁶ |
| **Optimizer** | Adam (β₁=0.91, β₂=0.999) | Adaptive moments |
| **Gradient clipping** | 1.0 (global norm) | Prevent instability |
| **Warmup epochs** | 3 | Feature extractor stabilization |
| **Total epochs** | 100 | With early stopping (patience 22) |
| **Data augmentation** | 3× per sample | HFlip, VFlip, Rotate±15° |

</div>

### Model Complexity

<div align="center">

| Component | Parameters | FLOPs | Description |
|:----------|:----------:|:-----:|:------------|
| EfficientNet-B4 Encoder | 17.7M | 4.2G | ImageNet pretrained, first 50 layers frozen |
| Multi-Scale Fusion (×5) | 1.4M | 0.8G | Dilated separable convolutions |
| Decoder (4 levels) | 3.8M | 2.1G | U-Net with attention gates |
| TIM (×4 levels) | 2.1M | 0.5G | Bidirectional task interaction |
| UPA (×4 levels) | 0.2M | 0.1G | Variance-based weighting |
| **Total** | **~25.2M** | **~7.7G** | Inference: 42ms per image (RTX 3090) |

</div>

---

## Dependencies

**Core framework:**
```
tensorflow==2.13.0
numpy==1.24.3
```

**Computer vision:**
```
opencv-python==4.8.0.74
albumentations==1.3.1
scikit-image==0.21.0
```

**Machine learning utilities:**
```
scikit-learn==1.3.0
```

**Visualization:**
```
matplotlib==3.7.2
seaborn==0.12.2
```

Full dependencies available in `requirements.txt`

---

## Installation
```bash
git clone https://github.com/C-loud-Nine/Uncertainty-Aware-Multi-Level-Decoder-Interaction.git
cd Uncertainty-Aware-Multi-Level-Decoder-Interaction
pip install -r requirements.txt
```



---

## License

This project is licensed under the MIT License. See `LICENSE` file for details.

---

## Acknowledgments

**Datasets:**
- BUSI: Al-Dhabyani et al., "Dataset of breast ultrasound images," *Data in Brief* 28:104863, 2020
- BUSI-WHU: Huang et al., "BUSI-WHU: Breast cancer ultrasound image dataset," *Mendeley Data V3*, 2025

**Technical foundations:**
- EfficientNet: Tan & Le, "EfficientNet: Rethinking model scaling for CNNs," *ICML*, 2019
- Focal Loss: Lin et al., "Focal loss for dense object detection," *ICCV*, 2017
- Attention U-Net: Oktay et al., "Attention U-Net: Learning where to look for the pancreas," *MIDL*, 2018

---

<div align="center">

**Code released for research reproducibility**

</div>
