# Sine-Interference Neural Network for Periodic Regression

A custom PyTorch neural network that combines linear projections with sine-based interference patterns, evaluated on synthetic periodic signal regression.

## Key finding

**Learnable per-neuron frequency (omega) does NOT improve over a fixed-frequency baseline on this task.**

| Model | Config | Final Val Loss |
|---|---|---|
| V1 | Fixed frequency (sin(x)) | **0.3321** |
| V2 | Learnable frequency (sin(omega·x)) | 0.3630 |

V2 is ~9.3% worse than V1, suggesting the added parameters introduce training instability without measurable benefit.

## Files

| File | Purpose |
|---|---|
| `novel_ai.py` | Training script (v1 baseline) |
| `novel_ai_v1.py` | Backup of the v1 baseline |
| `novel_ai_v2.py` | V2 with learnable omega |
| `compare.py` | Side-by-side comparison of v1 and v2 |
| `NOTES.md` | Experiment log |
| `loss_curve.png` | Training curve |
| `comparison.png` | V1 vs V2 comparison figure |

## How to run
pip install torch numpy matplotlib
python novel_ai.py
python novel_ai_v2.py
python compare.py

text

## Dataset

Synthetic — 3000 sine waves with random frequency (0.5–4.5 Hz) and phase (0–2π), sampled at 20 points with 5% Gaussian noise. The model must recover both frequency and phase.

## Author

Adam Fulugence — Independent Researcher
