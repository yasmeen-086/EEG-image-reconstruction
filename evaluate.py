import os
import glob
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import open_clip

from models.dual_branch_encoder import DualBranchEncoder
from utils.preprocessing import (
    preprocess_subject,
    preprocess_subject_with_stats
)

# ── Paths — update for your environment ───────────────────────
SUB01_TEST = '/kaggle/input/datasets/yaismeenkhan/eeg-sub01/preprocessed_eeg_test.npy'
SUB02_TEST = '/kaggle/input/datasets/yaismeenkhan/eeg-sub02/preprocessed_eeg_test.npy'
SUB03_TEST = '/kaggle/input/datasets/yaismeenkhan/eeg-sub03/preprocessed_eeg_test.npy'
SUB04_TEST = '/kaggle/input/datasets/yaismeenkhan/eeg-sub04/preprocessed_eeg_test.npy'
TEST_IMGS  = '/kaggle/input/**/*test_images*/**/*.jpg'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def build_pool_embeddings(test_imgs, device):
    """Compute CLIP embeddings for all test images."""
    clip_m, _, _ = open_clip.create_model_and_transforms(
        'ViT-L-14', pretrained='openai')
    clip_m = clip_m.to(device).eval()
    for p in clip_m.parameters():
        p.requires_grad_(False)

    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                    T.Normalize([.485,.456,.406],[.229,.224,.225])])

    pool = []
    for i in range(0, len(test_imgs), 64):
        batch = torch.stack([
            tf(Image.open(p).convert('RGB'))
            for p in test_imgs[i:i+64]
        ]).to(device)
        with torch.no_grad():
            emb = F.normalize(clip_m.encode_image(batch), dim=-1)
        pool.append(emb.cpu())

    return torch.cat(pool)   # (N_test, 768)


@torch.no_grad()
def evaluate(model, eeg_data, sids, pool, label, device):
    model.eval()
    pool      = pool.to(device)
    eeg_t     = torch.from_numpy(eeg_data).to(device)
    sid_t     = torch.from_numpy(sids).long().to(device)
    pool_size = len(pool)

    eeg_embs = []
    for i in range(0, len(eeg_t), 64):
        emb = model(eeg_t[i:i+64], sid_t[i:i+64])
        eeg_embs.append(emb.cpu())
    eeg_embs = torch.cat(eeg_embs).to(device)

    N          = len(eeg_embs)
    gt_indices = [i % pool_size for i in range(N)]
    gt_embs    = pool[gt_indices]

    clip_sim = F.cosine_similarity(eeg_embs, gt_embs).mean().item()

    sim_mat = eeg_embs @ pool.T
    t1 = t5 = t10 = 0
    for i in range(N):
        ranked = sim_mat[i].topk(10).indices.tolist()
        if gt_indices[i] in ranked[:1]:  t1  += 1
        if gt_indices[i] in ranked[:5]:  t5  += 1
        if gt_indices[i] in ranked[:10]: t10 += 1

    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  Samples         : {N} ({N//pool_size} subjects x {pool_size})")
    print(f"  CLIP Similarity : {clip_sim:.4f}")
    print(f"  Top-1  Accuracy : {t1/N:.4f}  ({t1}/{N})")
    print(f"  Top-5  Accuracy : {t5/N:.4f}  ({t5}/{N})")
    print(f"  Top-10 Accuracy : {t10/N:.4f}  ({t10}/{N})")
    print(f"{'='*50}")
    return clip_sim, t1/N, t5/N, t10/N


def main(ckpt_path):
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {ckpt_path}")

    # Load model
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = DualBranchEncoder(n_subjects=2).to(DEVICE)
    model.load_state_dict(ckpt['model'])
    print(f"Loaded epoch {ckpt['epoch']}, loss={ckpt['loss']:.4f}")

    # Load subject stats
    stats = np.load('subject_stats.npy', allow_pickle=True).item()

    # Load test EEG
    sub01_test, _, _ = preprocess_subject_with_stats(
        SUB01_TEST, stats[0]['mean'], stats[0]['std'])
    sub02_test, _, _ = preprocess_subject_with_stats(
        SUB02_TEST, stats[1]['mean'], stats[1]['std'])
    sub03_test, _, _ = preprocess_subject(SUB03_TEST)
    sub04_test, _, _ = preprocess_subject(SUB04_TEST)

    ds1_eeg  = np.concatenate([sub01_test, sub02_test])
    ds1_sids = np.array([0]*len(sub01_test) + [1]*len(sub02_test),
                         dtype=np.int64)
    ds2_eeg  = np.concatenate([sub03_test, sub04_test])
    ds2_sids = np.array([0]*len(sub03_test) + [1]*len(sub04_test),
                         dtype=np.int64)

    # Build test image pool
    test_imgs = sorted(set(
        glob.glob(TEST_IMGS, recursive=True)
    ))
    print(f"Test images: {len(test_imgs)}")
    pool = build_pool_embeddings(test_imgs, DEVICE)
    print(f"Pool: {pool.shape}")

    # Evaluate
    r1 = evaluate(model, ds1_eeg, ds1_sids, pool,
                  "DS1 — sub-01 + sub-02 (in-subject)", DEVICE)
    r2 = evaluate(model, ds2_eeg, ds2_sids, pool,
                  "DS2 — sub-03 + sub-04 (cross-subject)", DEVICE)

    # Summary table
    print(f"\n{'='*52}")
    print(f"  {'Metric':<20} {'DS1':>8} {'DS2':>8}")
    print(f"  {'-'*40}")
    for name, v1, v2 in zip(
        ['CLIP-Sim', 'Top-1', 'Top-5', 'Top-10'],
        r1, r2
    ):
        print(f"  {name:<20} {v1:>8.4f} {v2:>8.4f}")
    print(f"{'='*52}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='dual_branch_best.pt')
    args = parser.parse_args()
    main(args.checkpoint)
