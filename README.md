Official implementation of "Multi-Level Bidirectional Decoder Interaction for Uncertainty-Aware Breast Ultrasound Analysis"

## Installation
```bash
pip install -r requirements.txt
```

## Dataset Setup

**BUSI:**
```
data/BUSI/
├── benign/
├── malignant/
└── normal/
```

**BUSI-WHU:**
```
data/BUSI-WHU/
├── img/
├── gt/
└── Patient_infos_BUSI-WHU.xlsx
```

Update paths in `config.py`.

## Training
```bash
python train.py --dataset BUSI --epochs 100
python train.py --dataset BUSI-WHU --epochs 100
```

## Results

| Method | IoU | Dice | Acc |
|--------|-----|------|-----|
| U-Net | 66.80 | 80.05 | 86.50 |
| MISSFormer | 72.80 | 84.25 | -- |
| MTL-OCA | 72.90 | 84.30 | 89.90 |
| **Ours** | **74.50** | **85.25** | **90.60** |

## Architecture

- EfficientNet-B4 encoder
- 4-level decoder with TIM at each level (D₁-D₄)
- UPA for adaptive task weighting
- Multi-stream classification ensemble

