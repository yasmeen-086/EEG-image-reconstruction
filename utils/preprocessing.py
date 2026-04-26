
import numpy as np
import scipy.signal as signal


def preprocess_subject(path):
    """
    Load and preprocess EEG for a single subject.
    Returns normalized EEG, mean, std (for test-time reuse).
    """
    raw = np.load(path, allow_pickle=True).item()
    eeg = np.array(raw['preprocessed_eeg_data'], dtype=np.float64)

    if eeg.ndim == 4:
        eeg = eeg.mean(axis=1)          # average repetitions
    if eeg.shape[1] == 64:
        eeg = eeg[:, :63, :]            # drop reference channel

    # Bandpass filter 0.1–40 Hz
    nyq  = 100 / 2
    b, a = signal.butter(4, [0.1 / nyq, 40.0 / nyq], btype='band')
    for i in range(len(eeg)):
        eeg[i] = signal.filtfilt(b, a, eeg[i], axis=-1)

    # Baseline correction
    eeg -= eeg[:, :, :10].mean(axis=2, keepdims=True)

    # Subject-specific normalization
    mean = eeg.mean(axis=(0, 2), keepdims=True)
    std  = eeg.std(axis=(0, 2),  keepdims=True)
    std  = np.where(std < 1e-8, 1.0, std)

    return ((eeg - mean) / std).astype(np.float32), mean, std


def preprocess_subject_with_stats(path, mean, std):
    """
    Preprocess using pre-computed training statistics.
    Use this for test subjects seen during training.
    """
    raw = np.load(path, allow_pickle=True).item()
    eeg = np.array(raw['preprocessed_eeg_data'], dtype=np.float64)

    if eeg.ndim == 4:
        eeg = eeg.mean(axis=1)
    if eeg.shape[1] == 64:
        eeg = eeg[:, :63, :]

    nyq  = 100 / 2
    b, a = signal.butter(4, [0.1 / nyq, 40.0 / nyq], btype='band')
    for i in range(len(eeg)):
        eeg[i] = signal.filtfilt(b, a, eeg[i], axis=-1)

    eeg -= eeg[:, :, :10].mean(axis=2, keepdims=True)

    return ((eeg - mean) / std).astype(np.float32), mean, std


def augment_eeg(x):
    """
    Online augmentation during training.
    x: (B, 63, 100) torch tensor on device
    """
    import torch
    B, C, T = x.shape

    x = x + torch.randn_like(x) * 0.008

    shift = torch.randint(-4, 5, (1,)).item()
    x = torch.roll(x, shift, dims=2)

    mask = torch.ones(B, C, 1, device=x.device)
    drop = torch.randperm(C)[:max(1, C // 10)]
    mask[:, drop] = 0
    x = x * mask

    return x
