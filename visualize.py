import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.signal as sp_signal
from PIL import Image
import torchvision.transforms as T
import open_clip

from models.dual_branch_encoder import DualBranchEncoder
from utils.preprocessing import (
    preprocess_subject,
    preprocess_subject_with_stats
)

DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT      = 'dual_branch_best.pt'
OUT_DIR   = 'figures'
TEST_IMGS = '/kaggle/input/**/*test_images*/**/*.jpg'

os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3
})

# ── Real numbers ──────────────────────────────────────────────
ds1_clip=0.1290; ds2_clip=0.0781; rand_clip=0.03
ds1_top1=7.75;   ds2_top1=0.75;   rand_top1=0.50
ds1_top5=21.00;  ds2_top5=7.50;   rand_top5=2.50
ds1_top10=32.00; ds2_top10=13.75; rand_top10=5.00

logged = {
    1:3.5395,9:3.5123,10:3.4748,11:3.4263,12:3.3801,
    13:3.3264,14:3.2802,15:3.2227,16:3.1743,17:3.1165,
    18:3.0647,19:3.0185,20:2.9613,21:2.9040,22:2.8567,
    23:2.8060,24:2.7671,25:2.6866,26:2.6597,27:2.6025,
    28:2.5430,29:2.4878,30:2.4398,31:2.3952,32:2.3410,
    33:2.2992,34:2.2333,35:2.2031,36:2.1408,37:2.0911,
    38:2.0475,39:1.9931,40:1.9512,41:1.9030,42:1.8529,
    43:1.8174,44:1.7789,45:1.7227,46:1.6794,47:1.6449,
    48:1.6061,49:1.5696,50:1.5336,51:1.4893,52:1.4616,
    53:1.4256,54:1.3976,55:1.3800,56:1.3455,57:1.3019,
    58:1.2873,59:1.2601,60:1.2352,61:1.2004,62:1.1782,
    63:1.1658,64:1.1363,65:1.1154,66:1.0911,67:1.0666,
    68:1.0524,69:1.0351,70:1.0068,71:0.9862,73:0.9611,
    74:0.9482,75:0.9308,76:0.9133,77:0.8952,78:0.8823,
    79:0.8661,80:0.8546,81:0.8385,82:0.8258,83:0.8198,
    84:0.8044,85:0.7906,86:0.7821,87:0.7736,88:0.7497,
    89:0.7490,90:0.7396,91:0.7295,92:0.7234,93:0.7041,
    94:0.7019,95:0.6947,96:0.6823,97:0.6680,98:0.6672,
    99:0.6554,100:0.6498,101:0.6398,102:0.6394,103:0.6248,
    104:0.6245,105:0.6145,106:0.6056,107:0.6037,108:0.5989,
    109:0.5865,110:0.5864,111:0.5759,113:0.5696,114:0.5581,
    116:0.5529,117:0.5455,118:0.5446,119:0.5326,120:0.5347,
    121:0.5259,122:0.5257,123:0.5203,124:0.5122,125:0.5110,
    128:0.5003,129:0.4993,130:0.4923,131:0.4900,132:0.4894,
    133:0.4822,134:0.4802,136:0.4789,137:0.4729,138:0.4691,
    139:0.4656,140:0.4678,141:0.4631,142:0.4602,143:0.4579,
    144:0.4543,145:0.4491,148:0.4473,150:0.4397,151:0.4385,
    154:0.4359,155:0.4351,157:0.4283,159:0.4242,160:0.4260,
    163:0.4213,164:0.4165,168:0.4149,169:0.4143,170:0.4169,
    173:0.4100,179:0.4096,180:0.4093,181:0.4066,185:0.4060,
    189:0.4039,190:0.4063,191:0.4032,193:0.4010,200:0.4045
}
all_ep   = np.arange(1, 201)
log_ep   = np.array(sorted(logged.keys()))
log_val  = np.array([logged[e] for e in log_ep])
losses   = np.interp(all_ep, log_ep, log_val)


def fig1_loss():
    best_ep  = int(np.argmin(losses)) + 1
    best_val = float(np.min(losses))
    fig, ax  = plt.subplots(figsize=(10, 5))
    ax.plot(all_ep, losses, color='#1B4F8A', linewidth=2.5)
    ax.fill_between(all_ep, losses, alpha=0.1, color='#1B4F8A')
    ax.axvline(x=best_ep, color='#DC2626', linestyle='--',
               linewidth=1.5, alpha=0.8, label=f'Best epoch {best_ep}')
    ax.axhline(y=best_val, color='#DC2626', linestyle=':', linewidth=1, alpha=0.5)
    ax.annotate(f'Best: {best_val:.4f}', xy=(best_ep, best_val),
                xytext=(best_ep+12, best_val+0.12), fontsize=10,
                color='#DC2626',
                arrowprops=dict(arrowstyle='->', color='#DC2626', lw=1.2))
    ax.axvspan(1,   70,  alpha=0.04, color='#3B82F6', label='Fast descent')
    ax.axvspan(70,  140, alpha=0.04, color='#F59E0B', label='Refinement')
    ax.axvspan(140, 200, alpha=0.04, color='#10B981', label='Convergence')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Contrastive Loss (hard-negative InfoNCE)', fontsize=12)
    ax.set_title('Encoder Training Loss — THINGS-EEG2\n'
                 'Dual-Branch Encoder, 200 epochs, batch=128',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlim(1, 200)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/01_training_loss.png', dpi=200)
    plt.close()
    print("✓ Figure 1")


def fig2_metrics():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, vals, lbls, colors, ylabel, title, ylim in [
        (axes[0],
         [ds1_clip, ds2_clip, rand_clip],
         ['DS1\n(sub-01,02)', 'DS2\n(sub-03,04)', 'Random\nchance'],
         ['#1B4F8A', '#0D9488', '#94A3B8'],
         'Cosine Similarity', 'CLIP Similarity Score', 0.18),
    ]:
        bars = ax.bar(lbls, vals, color=colors, edgecolor='black',
                      lw=0.8, width=0.4)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_ylim(0, ylim)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2,
                    v+0.003, f'{v:.4f}',
                    ha='center', fontsize=11, fontweight='500')

    x = np.arange(3); w = 0.25
    b1 = axes[1].bar(x-w,  [ds1_top1,ds1_top5,ds1_top10],  w,
                     label='DS1', color='#1B4F8A', edgecolor='black', lw=0.8)
    b2 = axes[1].bar(x,    [ds2_top1,ds2_top5,ds2_top10],  w,
                     label='DS2', color='#0D9488', edgecolor='black', lw=0.8)
    b3 = axes[1].bar(x+w,  [rand_top1,rand_top5,rand_top10], w,
                     label='Random', color='#94A3B8',
                     edgecolor='black', lw=0.8, hatch='//')
    for grp in [b1, b2, b3]:
        for bar in grp:
            h = bar.get_height()
            axes[1].text(bar.get_x()+bar.get_width()/2,
                         h+0.3, f'{h:.1f}', ha='center', fontsize=8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['Top-1 (%)','Top-5 (%)','Top-10 (%)'])
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Top-K Retrieval Accuracy', fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].set_ylim(0, ds1_top10*1.2)
    plt.suptitle('Retrieval Performance — THINGS-EEG2',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/02_metrics.png', dpi=200)
    plt.close()
    print("✓ Figure 2")


def fig3_topk():
    ks = [1, 5, 10, 200]
    d1 = [ds1_top1/100, ds1_top5/100, ds1_top10/100, 1.0]
    d2 = [ds2_top1/100, ds2_top5/100, ds2_top10/100, 1.0]
    rn = [k/200 for k in ks]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, d1, 'o-', label='DS1 — in-subject',
            color='#1B4F8A', linewidth=2.5, markersize=9)
    ax.plot(ks, d2, 's-', label='DS2 — cross-subject',
            color='#0D9488', linewidth=2.5, markersize=9)
    ax.plot(ks, rn, '--', label='Random chance',
            color='#94A3B8', linewidth=2)
    for k, v in zip([1,5,10], d1[:3]):
        ax.annotate(f'{v*100:.1f}%', xy=(k,v),
                    xytext=(k*1.5, v+0.025), fontsize=9, color='#1B4F8A')
    for k, v in zip([1,5,10], d2[:3]):
        ax.annotate(f'{v*100:.1f}%', xy=(k,v),
                    xytext=(k*1.5, v-0.05), fontsize=9, color='#0D9488')
    ax.set_xscale('log'); ax.set_xlim(0.8,300); ax.set_ylim(-0.02,1.05)
    ax.set_xlabel('K  (pool size = 200)', fontsize=12)
    ax.set_ylabel('Top-K Accuracy', fontsize=12)
    ax.set_title('Top-K Retrieval Accuracy — THINGS-EEG2',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/03_topk_curves.png', dpi=200)
    plt.close()
    print("✓ Figure 3")


def fig4_gap():
    names = ['CLIP-Sim','Top-1','Top-5','Top-10']
    d1 = np.array([ds1_clip,ds1_top1/100,ds1_top5/100,ds1_top10/100])
    d2 = np.array([ds2_clip,ds2_top1/100,ds2_top5/100,ds2_top10/100])
    gap = d1 - d2
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(4); w = 0.35
    b1 = axes[0].bar(x-w/2, d1, w, label='DS1',
                     color='#1B4F8A', edgecolor='black', lw=0.8)
    b2 = axes[0].bar(x+w/2, d2, w, label='DS2',
                     color='#0D9488', edgecolor='black', lw=0.8)
    axes[0].set_xticks(x); axes[0].set_xticklabels(names)
    axes[0].set_ylabel('Score'); axes[0].legend()
    axes[0].set_title('DS1 vs DS2 — Absolute Values', fontweight='bold')
    for bar in list(b1)+list(b2):
        h = bar.get_height()
        axes[0].text(bar.get_x()+bar.get_width()/2,
                     h+0.003, f'{h:.3f}', ha='center', fontsize=8)
    bars = axes[1].bar(names, gap, color='#DC2626',
                       edgecolor='black', lw=0.8, width=0.5)
    axes[1].axhline(y=0, color='black', linewidth=0.8)
    axes[1].set_ylabel('DS1 − DS2 (gap)')
    axes[1].set_title('DS1 − DS2 Performance Gap', fontweight='bold')
    for bar, v in zip(bars, gap):
        axes[1].text(bar.get_x()+bar.get_width()/2,
                     v+0.002, f'+{v:.3f}', ha='center', fontsize=9)
    plt.suptitle('In-subject vs Cross-subject — THINGS-EEG2',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/04_ds1_ds2_gap.png', dpi=200)
    plt.close()
    print("✓ Figure 4")


def fig5_eeg():
    data = np.load(
        '/kaggle/input/datasets/yaismeenkhan/eeg-sub01/preprocessed_eeg_training.npy',
        allow_pickle=True).item()
    raw    = np.array(data['preprocessed_eeg_data'])
    sample = raw[42, 0, :63, :] if raw.ndim == 4 else raw[42, :63, :]
    t_ms   = np.linspace(0, 1000, sample.shape[-1])
    colors = plt.cm.tab10(np.linspace(0, 0.9, 8))

    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    spacing = max(np.abs(sample).max() * 2.5, 1.0)
    for i in range(8):
        axes[0].plot(t_ms, sample[i]+i*spacing,
                     color=colors[i], linewidth=1.2, label=f'Ch {i+1}')
    axes[0].axvline(x=0, color='red', linestyle='--',
                    linewidth=1.5, alpha=0.7, label='Stimulus onset')
    axes[0].set_xlabel('Time (ms)', fontsize=11)
    axes[0].set_ylabel('Amplitude (a.u.) + offset', fontsize=11)
    axes[0].set_title('EEG Signal — Single Trial, Subject 01 (8 of 63 channels)',
                      fontweight='bold', fontsize=12)
    axes[0].legend(loc='upper right', ncol=2, fontsize=8)

    freqs, psd = sp_signal.welch(
        sample[0], fs=100, nperseg=min(50, sample.shape[-1]))
    axes[1].semilogy(freqs, psd, color='#1B4F8A', linewidth=2)
    axes[1].axvspan(0.5, 4,  alpha=0.15, color='#7F77DD',
                    label='Delta (0.5–4 Hz)')
    axes[1].axvspan(4,   8,  alpha=0.15, color='#0D9488',
                    label='Theta (4–8 Hz)')
    axes[1].axvspan(8,  13,  alpha=0.15, color='#F59E0B',
                    label='Alpha (8–13 Hz)')
    axes[1].axvspan(13, 30,  alpha=0.15, color='#DC2626',
                    label='Beta (13–30 Hz)')
    axes[1].set_xlabel('Frequency (Hz)', fontsize=11)
    axes[1].set_ylabel('Power Spectral Density', fontsize=11)
    axes[1].set_title('Power Spectrum — Ch 1', fontweight='bold')
    axes[1].legend(fontsize=9); axes[1].set_xlim(0, 50)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/05_eeg_signal.png', dpi=200)
    plt.close()
    print("✓ Figure 5")


if __name__ == '__main__':
    fig1_loss()
    fig2_metrics()
    fig3_topk()
    fig4_gap()
    fig5_eeg()
    print(f"\nAll figures saved to ./{OUT_DIR}/")
