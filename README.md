<div align="center">

# VisioLex

### End-to-end lip reading — transcribe speech from silent video

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.11](https://img.shields.io/badge/PyTorch-2.11-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MediaPipe 0.10](https://img.shields.io/badge/MediaPipe-0.10-0097A7?logo=google&logoColor=white)](https://mediapipe.dev/)
[![Apple M4 MPS](https://img.shields.io/badge/Apple%20M4-MPS%20Accelerated-000000?logo=apple&logoColor=white)](https://developer.apple.com/metal/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.13-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![WER 15.5%](https://img.shields.io/badge/Val%20WER-15.5%25-brightgreen)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

VisioLex is an end-to-end visual speech recognition system built from scratch. It reads lip motion from silent video and outputs a text transcript — no microphone, no audio signal, no pretrained backbone. The model combines a 3D convolutional spatial encoder with a bidirectional GRU temporal decoder, trained with CTC loss directly on raw character sequences. MediaPipe Face Landmarker extracts a tight mouth ROI from each frame, giving the model a stable, speaker-agnostic input crop. The entire pipeline — preprocessing, training, and inference — runs natively on Apple M4 via PyTorch's MPS backend. Trained on 10,000 clips across 10 speakers from the GRID Corpus, VisioLex achieves **84.5% word-level accuracy (WER 0.155)** with 11.3M parameters and zero cloud compute.

---

## Key Features

- **3D CNN backbone** — three Conv3d blocks with BatchNorm capture spatiotemporal lip motion across the full frame sequence in a single forward pass
- **BiGRU sequence model** — 2-layer bidirectional GRU maps the CNN feature sequence to character logits with full temporal context
- **CTC decoder** — connectionist temporal classification removes the need for any forced alignment; the model learns directly from transcripts
- **MediaPipe mouth ROI extraction** — 478 facial landmarks locate the mouth precisely each frame; crops are resized to 64×64 px and stacked to a (1, 75, 64, 64) grayscale volume
- **Apple M4 MPS acceleration** — full forward and backward passes run on-chip via Metal; preprocessing runs at ~3 clips/sec on the M4 Neural Engine
- **GRID Corpus** — 10 speakers × 1,000 clips, structured command sentences, 51-word vocabulary

---

## Tech Stack

| Component | Library | Version |
|-----------|---------|---------|
| Model & training | PyTorch | 2.11 |
| Face landmark detection | MediaPipe | 0.10 |
| Video decoding & frame ops | OpenCV | 4.13 |
| Numerical arrays | NumPy | 2.x |
| Demo UI | Gradio | 6.x |

---

## Architecture

```
Video Frames
     │
     ▼
MediaPipe Face Landmarker  →  Mouth ROI crop (64×64 px, 75 frames)
     │
     ▼
┌────────────────────────────┐
│       3D CNN Backbone       │
│  Conv3d(1→32,  k=3×5×5)   │
│  Conv3d(32→64, k=3×5×5)   │
│  Conv3d(64→96, k=3×3×3)   │
│  BatchNorm + MaxPool each  │
└────────────┬───────────────┘
             │  (B, T, 6144)
             ▼
┌────────────────────────────┐
│     BiGRU  ×2 layers        │
│     hidden = 256 × 2        │
└────────────┬───────────────┘
             │  (B, T, 512)
             ▼
┌────────────────────────────┐
│  Linear → vocab (29 chars) │
│  CTC decode → transcript   │
└────────────────────────────┘
             │
             ▼
      Predicted Text
```

| Stage | Module | Output shape |
|-------|--------|-------------|
| Input | Mouth ROI | `(B, 1, 75, 64, 64)` |
| CNN block 1 | Conv3d(1→32, k=3×5×5) + BN + MaxPool | `(B, 32, 75, 32, 32)` |
| CNN block 2 | Conv3d(32→64, k=3×5×5) + BN + MaxPool | `(B, 64, 75, 16, 16)` |
| CNN block 3 | Conv3d(64→96, k=3×3×3) + BN + MaxPool | `(B, 96, 75, 8, 8)` |
| Flatten | — | `(B, 75, 6144)` |
| BiGRU | 2 layers, hidden=256, bidirectional | `(B, 75, 512)` |
| Classifier | Linear + log-softmax | `(T, B, 29)` |
| Decoder | Greedy CTC | `str` |

**11.3M trainable parameters. No pretrained weights. No transfer learning.**

<img width="1889" height="808" alt="Visiolex" src="https://github.com/user-attachments/assets/bf0f0d1d-3b96-4535-bbcd-8170603fca55" />

---

## Results

Trained on 10 speakers × 1,000 clips from the GRID Corpus (10,000 clips total).
Hardware: Apple M4 Mac Mini, MPS backend. Cloud spend: $0.

| Metric | Value |
|--------|-------|
| **Word Accuracy** | **84.5%** |
| **Val WER** | **0.155** |
| Parameters | 11.3M |
| Hardware | Apple M4 MPS |
| Dataset | GRID Corpus (10 speakers) |
| Training time | ~7 hours |
| Total clips | 10,000 |

### Training curve

| Epoch | Val WER | Word Accuracy | Note |
|-------|---------|--------------|------|
| 1 | 1.000 | 0% | Baseline |
| 10 | 0.875 | 12.5% | |
| 16 | 0.395 | 60.5% | Breakthrough epoch |
| 20 | 0.273 | 72.7% | |
| 61 | 0.157 | 84.3% | |
| **65** | **0.155** | **84.5%** | **Best checkpoint** |
| 75 | 0.157 | 84.3% | Final epoch |

---

## How to Run Locally

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/visiolex.git
cd visiolex

conda create -n visiolex python=3.11 -y
conda activate visiolex

# PyTorch first — anchors the dependency graph
pip install torch torchvision torchaudio

# Everything else
pip install numpy pillow opencv-python pyyaml tqdm editdistance \
            pandas matplotlib scikit-learn mediapipe gradio wandb pytest
```

### 2. Download GRID Corpus

Register (free) at https://spandh.dcs.shef.ac.uk/gridcorpus/ and download the video archives. Extract into `data/grid/`:

```bash
mkdir -p data/grid
for i in 1 2 3 4 5 6 7 8 9 10; do
    tar -xf ~/Downloads/s${i}.tar -C data/grid/
done
```

### 3. Preprocess

```bash
python scripts/preprocess_grid.py \
    --grid_root data/grid \
    --processed_dir data/processed \
    --speakers 1 2 3 4 5 6 7 8 9 10
```

Runs MediaPipe on every frame, crops the mouth ROI to 64×64 px, and saves each clip as a `.npy` array. ~4 min/speaker on M4.

### 4. Train

```bash
# Apple Silicon
python scripts/train.py \
    --config configs/train.yaml \
    --grid_root data/grid \
    --processed_dir data/processed \
    --speakers 1 2 3 4 5 6 7 8 9 10 \
    --epochs 75 \
    --batch_size 8 \
    --device mps

# CUDA
python scripts/train.py \
    --config configs/train.yaml \
    --processed_dir data/processed \
    --epochs 75 --batch_size 16 --device cuda
```

Best checkpoint is saved automatically to `checkpoints/best.pt` whenever val WER improves.

### 5. Run inference

```bash
python app/demo.py --checkpoint checkpoints/best.pt
# → http://localhost:7860
```

Upload a video clip or stream from webcam. VisioLex crops the mouth region in real time and outputs the lip-read transcript.

---

## Project Structure

```
visiolex/
├── configs/
│   └── train.yaml              # all hyperparameters
├── src/
│   ├── data/
│   │   ├── dataset.py          # GRIDDataset
│   │   ├── preprocessing.py    # MediaPipe mouth ROI extraction
│   │   ├── augmentation.py     # flip, brightness, crop, temporal jitter
│   │   └── dataloader.py       # build_dataloaders()
│   ├── models/
│   │   └── lipnet.py           # VisioLexModel (3D CNN + BiGRU)
│   ├── training/
│   │   ├── trainer.py          # training loop
│   │   └── ctc_loss.py         # CTCLoss wrapper (MPS-safe)
│   ├── decoding/
│   │   └── decoder.py          # GreedyCTCDecoder / BeamCTCDecoder
│   └── utils/
│       ├── text.py             # vocab, encode, decode
│       └── logging.py          # structured logger
├── scripts/
│   ├── preprocess_grid.py
│   └── train.py
├── app/
│   └── demo.py                 # Gradio demo
├── tests/
├── notebooks/
│   └── visiolex_colab.ipynb    # Colab training notebook
└── requirements.txt
```

---

## Scaling Up

The codebase is modular by design — each component can be upgraded independently.

**Larger datasets**

| Dataset | Clips | Vocabulary | Projected WER |
|---------|-------|-----------|--------------|
| GRID *(current)* | 10,000 | 51 words | 0.155 |
| LRS2 (BBC subtitles) | 45,000+ | Open vocab | ~0.15–0.25 |
| LRS3 (TED talks) | 150,000+ | Open vocab | ~0.08–0.15 |

**Stronger backbone** — swap `src/models/lipnet.py` without touching the training loop:

```
Current:     3D CNN + BiGRU             ~11M params
Next:        ResNet3D-18 + Transformer  ~30–50M params
State-of-art: AV-HuBERT                ~300M params
```

**Cloud training** — the training script already accepts `--device cuda`:

```bash
# Vast.ai RTX 4090 (~$0.30/hr) — 10× faster than M4 MPS
python scripts/train.py --device cuda --batch_size 32 --epochs 100
```

**Deployment**

```
Gradio demo        →  python app/demo.py
Hugging Face Spaces →  permanent public URL, free tier
CoreML export      →  on-device iOS/macOS, no network required
ONNX → TFLite      →  Android deployment
FastAPI            →  REST API wrapper around model.forward()
```

---

## Roadmap

- [x] MediaPipe mouth ROI extraction (Tasks API 0.10)
- [x] 3D CNN + BiGRU architecture (11.3M parameters)
- [x] CTC loss training loop (MPS-safe)
- [x] Data augmentation (flip, brightness, temporal jitter)
- [x] Multi-speaker training (10 speakers, 10,000 clips)
- [x] Apple M4 MPS acceleration
- [x] 75-epoch training run — **WER 0.155, 84.5% word accuracy**
- [x] Gradio demo (upload + webcam)
- [ ] Hugging Face Spaces deployment
- [ ] LRS2 / LRS3 fine-tuning
- [ ] Transformer backbone

---

## Tests

```bash
pytest tests/ -v
```

---

## Citation

```bibtex
@article{assael2016lipnet,
  title   = {LipNet: End-to-End Sentence-level Lipreading},
  author  = {Assael, Yannis M and Shillingford, Brendan and Whiteson, Shimon and de Freitas, Nando},
  journal = {arXiv preprint arXiv:1611.01599},
  year    = {2016}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
