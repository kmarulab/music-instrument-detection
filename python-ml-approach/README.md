# Instrument Classification – Python Pipeline

End‑to‑end deep‑learning workflow for recognising musical instruments from appx 5‑second WAV clips.

```
repo/
├── config.yaml         # hyper‑parameters
├── dataset.py          # PyTorch Dataset + split logic
├── model.py            # SmallCNN with global‑avg‑pool
├── train.py            # AMP‑enabled training loop (saves best_model.pt)
├── predict.py          # single‑file inference utility
└── README.md    
```

---

## 1  Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install torch torchaudio librosa soundfile tqdm pyyaml
```

---

## 2  Training

1. **Put audio** under `dataset/InstrumentName/*.wav` (any depth works; script glob‑scans).
2. Edit `config.yaml` if you want different Mel bins, learning rate, etc.

```bash
python3 train.py --data_dir ../dataset --lr 5e-4              # uses config.yaml
# override on command line - RECOMMENDED
python3 train.py --data_dir ../dataset --epochs 40 --lr 5e-4
```

### What happens under the hood

| Stage        | Highlights                                                                                        |
| ------------ | ------------------------------------------------------------------------------------------------- |
| Dataset scan | Splits ≈80 % / 10 % / 10 % (train/val/test) per folder, shuffles for reproducibility.             |
| Front‑end    | On‑the‑fly log‑Mel spectrogram (64 bins, 22.05 kHz, 25 ms frame, 10 ms hop).                      |
| Model        | 3× Conv‑BN‑ReLU‑Pool → GlobalAvgPool → FC(64→128→*n*) with dropout. Shape‑agnostic thanks to GAP. |
| Optimiser    | AdamW + mixed‑precision (`torch.amp.autocast` + `GradScaler`).                                    |
| Checkpoint   | Saves `best_model.pt` containing `state_dict`, `n_classes`, `label_names`, and full config.       |

Estimated speed: **≈4 min** on an RTX A1000 with 2 k training clips; scales with GPU.

---

## 3  Inference

```bash
# simplest – checkpoint already stores label_names
audio=dataset/ukulele/42.wav
python3 predict.py $audio --ckpt best_model.pt
```

Output example:

```
Predicted instrument: ukulele  (p=0.93)
```

---

## 4  Hyper‑parameter reference (`config.yaml`)

```yaml
sample_rate: 22050  # Hz
n_mels: 64
batch_size: 32
lr: 5e-4
epochs: 30
```

Change any field or override via CLI.

---

## 5  Extending

* **Data augmentation** – add `torchaudio.transforms.FrequencyMasking`, pitch‑shift with `librosa.effects.pitch_shift`, etc.
* **Larger models** – swap `SmallCNN` with `torch.hub.load('pytorch/audio', 'resnet18')` and fine‑tune.
* **Export** – `torch.onnx.export` then run through TensorRT for sub‑5 ms inference.

