import os, glob, random
import torch, torchaudio
from torch.utils.data import Dataset
import librosa

class InstrumentDataset(Dataset):
    def __init__(self, root, split='train', sr=22050, n_mels=64,
                 train_ratio=0.8, val_ratio=0.1):
        self.sr, self.n_mels = sr, n_mels
        wavs, labels = [], []
        for label, folder in enumerate(sorted(os.listdir(root))):
            files = glob.glob(os.path.join(root, folder, '*.wav'))
            random.shuffle(files)
            n = len(files)
            train_end = int(train_ratio*n)
            val_end   = int((train_ratio+val_ratio)*n)
            if split == 'train':
                split_files = files[:train_end]
            elif split == 'val':
                split_files = files[train_end:val_end]
            else:
                split_files = files[val_end:]
            wavs += split_files
            labels += [label]*len(split_files)
        self.files, self.labels = wavs, labels
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sr, n_fft=1024, hop_length=512, n_mels=n_mels,
            f_min=50, f_max=sr//2)

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        y, fs = librosa.load(path, sr=self.sr)
        if y.ndim > 1:
            y = y.mean(1)
        y = torch.tensor(y)
        with torch.no_grad():
            feat = self.melspec(y).clamp(min=1e-5).log()
        return feat.unsqueeze(0), self.labels[idx]
