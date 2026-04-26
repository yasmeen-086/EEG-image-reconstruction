# EEG-Based Image Retrieval — THINGS-EEG2

Brain-to-image retrieval using a dual-branch EEG encoder 
aligned with CLIP image embeddings.

## Results

| Metric | DS1 (in-subject) | DS2 (cross-subject) | Random |
|--------|-----------------|--------------------:|--------|
| CLIP-Sim | 0.1290 | 0.0781 | ~0.03 |
| Top-1 | 7.75% | 0.75% | 0.5% |
| Top-5 | 21.00% | 7.50% | 2.5% |
| Top-10 | 32.00% | 13.75% | 5.0% |

![Retrieval Results](figures/retrieval_results.png)

## Architecture

Dual-branch encoder:
- Temporal branch: Transformer (63ch → 256-dim CLS token)
- Frequency branch: Multi-scale CNN (4 kernel sizes)
- Subject conditioning: Embedding layer (n_subjects=2)
- Fusion: FC layers → 768-dim CLIP-aligned embedding
- Loss: Hard-negative InfoNCE (τ=0.07, k=48)

## Dataset

[THINGS-EEG2](https://www.nature.com/articles/s41597-022-01651-5)
- 4 subjects (sub-01 to sub-04)
- 16,740 training images, 200 test images
- 63 EEG channels, 100 timepoints @ 100Hz

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/eeg-image-retrieval
cd eeg-image-retrieval
pip install -r requirements.txt
```

## Usage

```bash
# Train
python train.py

# Evaluate
python evaluate.py --checkpoint dual_branch_best.pt

# Visualize
python visualize.py
```

## Training

- 200 epochs, batch size 128
- AdamW optimizer, lr=3e-4, weight_decay=1e-4
- Cosine annealing LR schedule (eta_min=1e-6)
- Best loss: 0.4010 at epoch 193
