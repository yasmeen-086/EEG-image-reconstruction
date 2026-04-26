
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBranch(nn.Module):
    def __init__(self, n_ch=63, n_t=100, d=256, heads=8, layers=4):
        super().__init__()
        self.proj = nn.Linear(n_ch, d)
        self.cls  = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pos  = nn.Parameter(torch.randn(1, n_t + 1, d) * 0.02)
        enc = nn.TransformerEncoderLayer(
            d, heads, d * 4, dropout=0.1,
            activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc, layers)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):           # (B, C, T)
        x   = x.permute(0, 2, 1)   # (B, T, C)
        x   = self.proj(x)          # (B, T, d)
        B   = x.size(0)
        cls = self.cls.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1) + self.pos[:, :x.size(1) + 1]
        return self.norm(self.transformer(x)[:, 0])


class FrequencyBranch(nn.Module):
    def __init__(self, n_ch=63, n_t=100, d=256):
        super().__init__()
        self.band_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(n_ch, 64, k, padding=k // 2),
                nn.BatchNorm1d(64), nn.GELU()
            ) for k in [3, 7, 15, 31]
        ])
        self.merge = nn.Sequential(
            nn.Conv1d(256, 256, 1),
            nn.BatchNorm1d(256), nn.GELU(),
            nn.AdaptiveAvgPool1d(8)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8, d * 2), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(d * 2, d), nn.LayerNorm(d)
        )

    def forward(self, x):           # (B, C, T)
        bands = [conv(x) for conv in self.band_convs]
        x = torch.cat(bands, dim=1) # (B, 256, T)
        return self.head(self.merge(x))


class DualBranchEncoder(nn.Module):
    def __init__(self, n_ch=63, n_t=100, branch_dim=256,
                 clip_dim=768, n_subjects=2):
        super().__init__()
        self.temporal = TemporalBranch(n_ch, n_t, branch_dim)
        self.freq     = FrequencyBranch(n_ch, n_t, branch_dim)

        self.subj_embed = nn.Embedding(n_subjects, 64)
        self.subj_proj  = nn.Linear(64, branch_dim * 2)

        self.fusion = nn.Sequential(
            nn.Linear(branch_dim * 2, 1024), nn.LayerNorm(1024),
            nn.GELU(), nn.Dropout(0.2),
            nn.Linear(1024, 1024), nn.LayerNorm(1024),
            nn.GELU(), nn.Dropout(0.1),
            nn.Linear(1024, clip_dim)
        )

    def forward(self, x, subject_id=None):
        t = self.temporal(x)
        f = self.freq(x)
        h = torch.cat([t, f], dim=-1)

        if subject_id is not None:
            h = h + self.subj_proj(self.subj_embed(subject_id))

        return F.normalize(self.fusion(h), dim=-1)


if __name__ == '__main__':
    model = DualBranchEncoder()
    x   = torch.randn(4, 63, 100)
    sid = torch.tensor([0, 1, 0, 1])
    out = model(x, sid)
    print(f"Output shape: {out.shape}")
    print(f"Parameters:   {sum(p.numel() for p in model.parameters()):,}")
