# Music-Instrument-Classification Project

This repository contains **two parallel signal-analysis pipelines** for recognising musical instruments from short instrument audio clips.  
The aim is to **compare a classic Digital-Signal-Processing (DSP) workflow in MATLAB with a modern machine-learning approach in Python**:

| Approach | Core Idea | Toolchain |
|----------|-----------|-----------|
| **MATLAB / DSP** | Hand-design acoustic features and feed the aggregated feature table into MATLAB’s *Classification Learner*. | MATLAB (R2021a +) + Audio Toolbox |
| **Python / ML** | Learn features end-to-end from log-Mel spectrograms using a lightweight CNN trained with PyTorch, accelerated on GPU + AMP. | Python 3.9 – 3.12, PyTorch 2.x |

---

## Directory Layout

instrument-classification/
├── dataset/                  # raw audio (kept out of Git since it is bulky) 
│   ├── cello/   *.wav
│   ├── flute/   *.wav
│   └── …/
│
├── matlab-dsp-approach/      
│   ├── extractInstrumentFeatures.m
│   └── README.md
│
├── python-ml-approach/       
│   ├── config.yaml
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   └── README.md
│
└── README.md                 


## Comparisons

| Dimension | MATLAB (DSP) | Python (ML) |
|-----------|--------------|-------------|
| **Feature philosophy** | “Know what to listen for.” You explicitly pick spectro-temporal descriptors grounded in psycho-acoustics. | “Let the network decide.” Convolutional layers learn filters jointly with the classifier. |
| **Interactivity** | Immediate visual feedback via *Classification Learner* charts (ROC, confusion matrix). | Script-driven; metrics printed to console, but scalable to larger datasets and GPUs. |
| **Transparency** | Easy to plot individual MFCCs, centroids, etc. | Requires probing activations / saliency maps to interpret learned features. |
| **Hardware needs** | CPU-only is fine (runs in seconds). | Benefits from NVIDIA GPU (mixed-precision cuts training time). |

The project is intentionally structured so you can **swap datasets** and **replicate experiments** quickly in either branch, then compare accuracy, training time, and interpretability.