import argparse, torch, torchaudio, librosa, pathlib
from model import SmallCNN

def load_wav(path, sr):
    y, fs = librosa.load(path, sr=sr)
    if y.ndim > 1:
        y = y.mean(0)
    return torch.tensor(y, dtype=torch.float32)

def read_label_file(path):
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]

def main(wav_path, ckpt_path, labels_txt=None, sr=22050, n_mels=64):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ckpt        = torch.load(ckpt_path, map_location=device)
    state_dict  = ckpt.get('state_dict', ckpt)
    n_classes   = state_dict['fc2.weight'].shape[0]
    label_names = ckpt.get('label_names')


    if labels_txt:
        label_names = read_label_file(labels_txt)

    model = SmallCNN(n_classes=n_classes).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    melspec = torchaudio.transforms.MelSpectrogram(
        sr, n_fft=1024, hop_length=512, n_mels=n_mels,
        f_min=50, f_max=sr//2).to(device)

    amp_ctx = torch.amp.autocast if hasattr(torch, 'amp') else torch.cuda.amp.autocast
    with torch.no_grad(), amp_ctx(device_type=device):
        wav  = load_wav(wav_path, sr).to(device)
        feat = melspec(wav).clamp(min=1e-5).log().unsqueeze(0).unsqueeze(0)
        probs = model(feat).softmax(1).cpu()

    idx   = int(probs.argmax(1))
    prob  = probs[0, idx].item()
    if label_names and idx < len(label_names):
        print(f'Predicted instrument: {label_names[idx]}  (p={prob:.3f})')
    else:
        print(f'Predicted class index: {idx}  (p={prob:.3f})')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('wav', help='Path to .wav file')
    ap.add_argument('--ckpt', default='best_model.pt')
    ap.add_argument('--labels', help='Optional labels.txt (one name per line)')
    args = ap.parse_args()
    main(args.wav, args.ckpt, args.labels)
