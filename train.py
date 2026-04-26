import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import open_clip

from models.dual_branch_encoder import DualBranchEncoder
from utils.preprocessing import preprocess_subject, augment_eeg
from utils.losses import hard_negative_infonce

# ── Config ────────────────────────────────────────────────────
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT       = 'dual_branch_best.pt'
CACHE_PATH = 'clip_cache.pt'
EPOCHS     = 200
BATCH_SIZE = 128
LR         = 3e-4
N_SUBJECTS = 2

# ── Paths — update for your environment ───────────────────────
SUB01_TRAIN = '/kaggle/input/datasets/yaismeenkhan/eeg-sub01/preprocessed_eeg_training.npy'
SUB02_TRAIN = '/kaggle/input/datasets/yaismeenkhan/eeg-sub02/preprocessed_eeg_training.npy'
IMAGES_GLOB = '/kaggle/input/datasets/yaismeenkhan/things-images/training_images/**/*.jpg'


# ── Dataset ───────────────────────────────────────────────────
class EEGDataset(Dataset):
    def __init__(self, eeg, subject_ids, cache):
        self.eeg  = eeg
        self.sids = subject_ids
        self.cache = cache

    def __len__(self): return len(self.eeg)

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.eeg[i]),
            self.cache.get(i, torch.zeros(768)),
            torch.tensor(self.sids[i], dtype=torch.long)
        )


# ── CLIP cache ────────────────────────────────────────────────
def build_clip_cache(eeg_len, img_paths, device):
    if os.path.exists(CACHE_PATH):
        print(f"Loading cache from {CACHE_PATH}")
        return torch.load(CACHE_PATH)

    print("Building CLIP cache...")
    clip_m, _, _ = open_clip.create_model_and_transforms(
        'ViT-L-14', pretrained='openai')
    clip_m = clip_m.to(device).eval()
    for p in clip_m.parameters():
        p.requires_grad_(False)

    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                    T.Normalize([.485,.456,.406],[.229,.224,.225])])

    unique_ids = list(set(i % len(img_paths) for i in range(eeg_len)))

    class ImgDS(Dataset):
        def __init__(self, ids):
            self.ids = ids
        def __len__(self): return len(self.ids)
        def __getitem__(self, idx):
            img_id = self.ids[idx]
            try:
                return tf(Image.open(img_paths[img_id]).convert('RGB')), img_id
            except:
                return torch.zeros(3, 224, 224), img_id

    img_cache = {}
    for img_b, img_ids in DataLoader(ImgDS(unique_ids), batch_size=256,
                                      num_workers=2):
        with torch.no_grad():
            emb = F.normalize(clip_m.encode_image(img_b.to(device)), dim=-1)
        for j, img_id in enumerate(img_ids):
            img_cache[int(img_id)] = emb[j].cpu()

    cache = {i: img_cache[i % len(img_paths)] for i in range(eeg_len)}
    torch.save(cache, CACHE_PATH)
    print(f"Saved {len(cache)} embeddings to {CACHE_PATH}")
    return cache


# ── Evaluation ────────────────────────────────────────────────
@torch.no_grad()
def evaluate_clip_sim(model, loader, device):
    model.eval()
    sims = []
    for x, y, sid in loader:
        emb = model(x.to(device), sid.to(device))
        cos = F.cosine_similarity(emb, y.to(device)).mean().item()
        sims.append(cos)
    model.train()
    return float(np.mean(sims))


# ── Main ──────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")

    # Load EEG
    sub01, mean01, std01 = preprocess_subject(SUB01_TRAIN)
    sub02, mean02, std02 = preprocess_subject(SUB02_TRAIN)
    N1, N2 = len(sub01), len(sub02)

    eeg         = np.concatenate([sub01, sub02])
    subject_ids = np.array([0]*N1 + [1]*N2, dtype=np.int64)
    print(f"EEG: {eeg.shape}  |  sub-01: {N1}  sub-02: {N2}")

    # Save stats for evaluation
    np.save('subject_stats.npy', {
        0: {'mean': mean01, 'std': std01},
        1: {'mean': mean02, 'std': std02}
    })

    # Build CLIP cache
    img_paths = sorted(glob.glob(IMAGES_GLOB, recursive=True))
    print(f"Training images: {len(img_paths)}")
    cache = build_clip_cache(len(eeg), img_paths, DEVICE)

    # DataLoader
    dataset = EEGDataset(eeg, subject_ids, cache)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE,
                         shuffle=True, num_workers=2, pin_memory=True)
    print(f"Loader: {len(loader)} batches/epoch")

    # Model
    model = DualBranchEncoder(n_subjects=N_SUBJECTS).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=EPOCHS, eta_min=1e-6)

    best_loss = float('inf')
    best_ep   = 0
    patience  = 0
    MAX_PAT   = 30

    print("\n" + "="*50)
    print(f"TRAINING — {EPOCHS} epochs")
    print("="*50)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0

        for x, y, sid in loader:
            x   = augment_eeg(x.to(DEVICE))
            y   = y.to(DEVICE)
            sid = sid.to(DEVICE)

            pred = model(x, sid)
            loss = hard_negative_infonce(pred, y)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()

        sched.step()
        avg = total / len(loader)

        if avg < best_loss - 1e-4:
            best_loss = avg
            best_ep   = epoch
            patience  = 0
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'loss':  avg
            }, CKPT)
            print(f"Epoch {epoch:3d}/{EPOCHS}  loss={avg:.4f}  <- best")
        else:
            patience += 1
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}/{EPOCHS}  loss={avg:.4f}"
                      f"  (patience {patience}/{MAX_PAT})")
            if epoch > 150 and patience >= MAX_PAT:
                print(f"\nEarly stop at epoch {epoch}"
                      f" (best={best_loss:.4f} at epoch {best_ep})")
                break

    print(f"\nDone — best loss: {best_loss:.4f} at epoch {best_ep}")


if __name__ == '__main__':
    main()
