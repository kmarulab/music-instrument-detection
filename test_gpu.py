import torch, torchaudio, platform, sys
print("PyTorch :", torch.__version__)
print("CUDA OK :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

