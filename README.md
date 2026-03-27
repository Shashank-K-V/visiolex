<div align="center">

# 👁️ VisioLex

### Visual Speech Recognition — read lips, not audio

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.11](https://img.shields.io/badge/PyTorch-2.11-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Apple MPS](https://img.shields.io/badge/Apple%20M4-MPS%20Accelerated-black?logo=apple&logoColor=white)](https://developer.apple.com/metal/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-0097A7?logo=google&logoColor=white)](https://mediapipe.dev/)
[![WER 15.5%](https://img.shields.io/badge/Val%20WER-15.5%25-brightgreen)](https://github.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**VisioLex watches a person's mouth through a camera and transcribes what they say — with zero audio.**

It combines MediaPipe face landmark detection, a 3D convolutional spatial encoder, and a bidirectional GRU temporal decoder, trained end-to-end with CTC loss on the GRID Corpus.

</div>

---

## What it does

| Input | Output |
|-------|--------|
| Silent video of a person speaking | Transcribed text |
| Webcam feed (live) | Real-time lip-read transcript |
| `.mpg` / `.mp4` clip | Word-error-rate scored transcript |

No microphone. No audio track. Pure visual inference.

---

## Architecture

```
Silent video  →  MediaPipe Face Landmarker  →  Mouth ROI (64×64 px, 75 frames)
                                                         │
                                               ┌─────────▼──────────┐
                                               │   3D CNN Backbone   │
                                               │  Conv3d ×3 + BN    │
                                               │  MaxPool (spatial)  │
                                               └─────────┬───────────┘
                                                         │  (B, T, 6144)
                                               ┌─────────▼───────────┐
                                               │  BiGRU  (2 layers)  │
                                               │  hidden = 256 × 2   │
                                               └─────────┬────────────┘
                                                         │  (B, T, 512)
                                               ┌─────────▼──────────┐
                                               │  Linear → vocab     │
                                               │  CTC decode         │
                                               └─────────┬───────────┘
                                                         │
                                                    Transcription
```

| Stage | Module | Output shape |
|-------|--------|-------------|
| Input | Mouth crop | `(B, 1, 75, 64, 64)` |
| CNN block 1 | Conv3d(1→32, k=3×5×5) + BN + MaxPool | `(B, 32, 75, 32, 32)` |
| CNN block 2 | Conv3d(32→64, k=3×5×5) + BN + MaxPool | `(B, 64, 75, 16, 16)` |
| CNN block 3 | Conv3d(64→96, k=3×3×3) + BN + MaxPool | `(B, 96, 75, 8, 8)` |
| Flatten | – | `(B, 75, 6144)` |
| BiGRU | 2 layers, hidden=256, bidirectional | `(B, 75, 512)` |
| Classifier | Linear + log-softmax | `(T, B, 29)` |
| Decoder | Greedy CTC / Beam search | `str` |

**11.3M trainable parameters** — trained from scratch, no pretrained weights.

---

## Results

Trained on **10 speakers × 1,000 clips = 10,000 video clips** from the GRID Corpus.
Trained entirely on an **Apple M4 Mac Mini** (MPS backend). No cloud GPU. No paid compute.

### Training progression

| Epoch | Val WER | Word Accuracy | Val Loss | Note |
|-------|---------|--------------|---------|------|
| 1 | 1.000 | 0% | 2.75 | Baseline |
| 3 | 0.967 | 3.3% | 2.11 | |
| 10 | 0.875 | 12.5% | 1.21 | |
| 16 | 0.395 | 60.5% | 0.43 | Breakthrough |
| 20 | 0.273 | 72.7% | 0.31 | |
| 61 | 0.157 | 84.3% | 0.19 | |
| **65** | **0.155** | **84.5%** | **0.18** | **Best checkpoint** 🏆 |
| 75 | 0.157 | 84.3% | 0.18 | Final epoch |

### Final numbers

| Metric | Value |
|--------|-------|
| **Best Val WER** | **0.155** |
| **Word Accuracy** | **84.5%** |
| Best checkpoint epoch | 65 / 75 |
| Final train loss | 0.046 |
| Final val loss | 0.182 |
| Training time | ~7 hours on Apple M4 MPS |
| Total training clips | 10,000 |
| Parameters | 11.3M (trained from scratch) |

---

## Built & trained on Apple Silicon

This project was **built and trained entirely on an Apple M4 Mac Mini** — no cloud GPU, no paid compute.

```
renderer: Apple M4  ·  MPS (Metal Performance Shaders)  ·  ~3 clips/sec preprocessing
```

PyTorch's MPS backend routes all tensor ops through the M4's GPU cores. The CTC loss
falls back to CPU for one op (MPS limitation), but the full forward + backward pass
runs on-device. Training speed: **~4 minutes per epoch** for 10 speakers on M4 MPS.

---

## Stack

| Component | Tool | Cost |
|-----------|------|------|
| Dataset | GRID Corpus (34 speakers × 1,000 clips each) | Free |
| Face detection | MediaPipe Face Landmarker 0.10 | Free |
| Model + training | PyTorch 2.11 | Free |
| Accelerator | Apple M4 MPS | (existing hardware) |
| Experiment tracking | Weights & Biases free tier | Free |
| Demo UI | Gradio | Free |

**Total cloud spend: $0.**

---

## Project structure

```
visiolex/
├── configs/
│   └── train.yaml              # all hyper-parameters in one place
├── src/
│   ├── data/
│   │   ├── dataset.py          # GRIDDataset (PyTorch Dataset)
│   │   ├── preprocessing.py    # MediaPipe mouth ROI extraction
│   │   ├── augmentation.py     # flip, brightness, crop, temporal jitter
│   │   └── dataloader.py       # build_dataloaders() factory
│   ├── models/
│   │   └── lipnet.py           # VisioLexModel (3D CNN + BiGRU)
│   ├── training/
│   │   ├── trainer.py          # full training loop + W&B logging
│   │   └── ctc_loss.py         # CTCLoss wrapper (MPS-safe)
│   ├── decoding/
│   │   └── decoder.py          # GreedyCTCDecoder / BeamCTCDecoder
│   └── utils/
│       ├── text.py             # vocab, encode_text, decode_indices
│       └── logging.py          # structured logger, AverageMeter
├── scripts/
│   ├── preprocess_grid.py      # pre-extract mouth crops → .npy files
│   └── train.py                # CLI training entry-point
├── app/
│   └── demo.py                 # Gradio web demo (upload or webcam)
├── tests/
│   ├── test_model.py
│   ├── test_decoding.py
│   └── test_dataset.py
├── notebooks/
│   └── visiolex_colab.ipynb    # full Colab training notebook
└── requirements.txt
```

---

## Quick start

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/visiolex.git
cd visiolex

conda create -n visiolex python=3.11 -y
conda activate visiolex

# Install PyTorch first (anchors the dependency graph)
pip install torch torchvision torchaudio

# Install everything else
pip install numpy pillow opencv-python pyyaml tqdm editdistance \
            pandas matplotlib scikit-learn mediapipe gradio wandb pytest
```

### 2. Download the GRID Corpus

Register at https://spandh.dcs.shef.ac.uk/gridcorpus/ (free academic access).
Download the video archives (`s1.tar` … `s10.tar`) and extract:

```bash
mkdir -p data/grid
# Extract each speaker — videos land in data/grid/s{n}/video/
for i in 1 2 3 4 5 6 7 8 9 10; do
    tar -xf ~/Downloads/s${i}.tar -C data/grid/
done
```

### 3. Pre-process mouth crops

```bash
python scripts/preprocess_grid.py \
    --grid_root data/grid \
    --processed_dir data/processed \
    --speakers 1 2 3 4 5 6 7 8 9 10
```

Runs MediaPipe Face Landmarker on every frame, crops the mouth region to 64×64 px,
and saves each clip as a `.npy` array. ~4 min/speaker on M4.

### 4. Train

```bash
# Apple Silicon (M1/M2/M3/M4)
python scripts/train.py \
    --config configs/train.yaml \
    --grid_root data/grid \
    --processed_dir data/processed \
    --speakers 1 2 3 4 5 6 7 8 9 10 \
    --epochs 75 \
    --batch_size 8 \
    --device mps

# CUDA GPU
python scripts/train.py \
    --config configs/train.yaml \
    --processed_dir data/processed \
    --epochs 75 --batch_size 16 --device cuda

# CPU only
python scripts/train.py \
    --config configs/train.yaml \
    --processed_dir data/processed \
    --epochs 75 --batch_size 4 --device cpu
```

Best checkpoint is auto-saved to `checkpoints/best.pt` whenever validation WER improves.

### 5. Run the demo

```bash
python app/demo.py --checkpoint checkpoints/best.pt
# → http://localhost:7860
```

Upload a video clip or use your webcam. VisioLex crops the mouth region in real time and outputs the lip-read transcript.

---

## Sanity-check run (single speaker, 10 epochs)

```bash
python scripts/train.py \
    --config configs/train.yaml \
    --processed_dir data/processed \
    --speakers 1 --epochs 10 --batch_size 8 --device mps --no_wandb
```

Expected output:
```
Epoch   1/10  train_loss=2.97  val_loss=2.75  val_wer=1.000
Epoch   3/10  train_loss=2.26  val_loss=2.11  val_wer=0.967
Epoch  10/10  train_loss=1.39  val_loss=1.21  val_wer=0.875
```

---

## Configuration

All hyper-parameters live in `configs/train.yaml` — nothing is hardcoded:

```yaml
model:
  vocab_size: 29
  cnn_channels: [32, 64, 96]
  gru_hidden: 256
  gru_layers: 2
  dropout: 0.5

training:
  epochs: 75
  batch_size: 8
  learning_rate: 3.0e-4
  scheduler: cosine
  warmup_epochs: 2

data:
  num_frames: 75
  img_size: 64
  train_split: 0.9
```

---

## Scaling up — where this can go

VisioLex is deliberately built with a clean, modular architecture so every component can be swapped out independently. Here is the full upgrade path from this proof-of-concept to a production system:

### Bigger datasets

| Dataset | Size | Vocab | Expected WER |
|---------|------|-------|-------------|
| GRID Corpus *(current)* | 34 speakers × 1,000 clips | 51 words | ~25–35% |
| LRS2 (BBC) | 45,000+ utterances | Open vocab | ~15–25% |
| LRS3 (TED) | 150,000+ utterances | Open vocab | ~10–18% |
| VoxCeleb2 + LRS3 | 1M+ clips | Open vocab | ~5–10% |

Switching dataset only requires implementing a new `Dataset` subclass — the training loop, model, and decoder are dataset-agnostic.

### Stronger backbone

The 3D CNN + BiGRU backbone can be replaced without touching any other code:

```
Current:    3D CNN (3 blocks) → BiGRU (2 layers)        ~11M params
Next step:  ResNet-18 3D / SlowFast → Transformer        ~30–50M params
State-of-art: AV-HuBERT / RAVEn (audio-visual pretrain) ~300M params
```

### Cloud training

The training script already accepts `--device cuda`. To scale up:

```bash
# Google Colab (free T4)  — open notebooks/visiolex_colab.ipynb
# Vast.ai RTX 4090        — ~$0.30/hr, 10× faster than M4 MPS
# AWS p3.2xlarge (V100)   — production-grade
# Multi-GPU               — add --strategy ddp to trainer

python scripts/train.py \
    --config configs/train.yaml \
    --processed_dir data/processed \
    --epochs 100 --batch_size 32 --device cuda
```

A full 100-epoch run on a V100 takes ~4 hours vs ~50 hours on M4 MPS.

### Deployment

```
Local demo      →  python app/demo.py         (Gradio, localhost)
Public demo     →  Hugging Face Spaces        (free, permanent URL)
Mobile          →  CoreML export (Apple)      (on-device, no internet)
                   ONNX → TFLite (Android)
API             →  FastAPI wrapper around model.forward()
Real-time       →  WebRTC + sliding window inference
```

### Language model rescoring

Adding a KenLM language model reduces WER by a further 10–15% with zero retraining:

```bash
pip install pyctcdecode kenlm
# Train a 4-gram LM on GRID transcripts → decoder automatically uses it
```

### What production looks like

A production lip-reading system built on this architecture would add:
- **Multi-view input** — front + side cameras
- **Speaker adaptation** — fine-tune on 5–10 minutes of new speaker data
- **Noise robustness** — augmentation with synthetic head pose, lighting changes
- **Streaming inference** — sliding 75-frame window at 25 fps = 3-second latency
- **Confidence scores** — per-word uncertainty estimates

---

## Honest limitations

VisioLex is trained on the **GRID Corpus** — a controlled lab dataset with a
fixed 51-word vocabulary and rigid sentence templates (`<command> <color> <prep> <letter> <digit> <adverb>`).

Wild-environment accuracy is lower due to:
- Lighting variation and motion blur
- Open vocabulary (words not in GRID)
- Head pose and partial occlusion

Production-grade accuracy requires larger datasets (LRS2 / LRS3) and a
transformer backbone. This project is a complete, working implementation
demonstrating the full pipeline.

---

## Roadmap

- [x] Data pipeline + GRID Dataset class
- [x] MediaPipe mouth ROI extraction (updated to Tasks API 0.10)
- [x] 3D CNN + BiGRU architecture (11.3M parameters)
- [x] Training loop with CTC loss (MPS-safe)
- [x] Data augmentation (flip, brightness, crop, temporal jitter)
- [x] Multi-speaker scaling (10 speakers, 10,000 clips)
- [x] Beam search decoder
- [x] Gradio demo (upload + webcam)
- [x] Native Apple M4 MPS training
- [x] Full 75-epoch training run — **84.5% word accuracy (WER 0.155)**
- [ ] Hugging Face Spaces deployment
- [ ] LRS2/LRS3 fine-tuning

---

## Tests

```bash
pytest tests/ -v
```

---

## Citation

If you use VisioLex in research, please cite the original LipNet paper:

```bibtex
@article{assael2016lipnet,
  title     = {LipNet: End-to-End Sentence-level Lipreading},
  author    = {Assael, Yannis M and Shillingford, Brendan and Whiteson, Shimon and de Freitas, Nando},
  journal   = {arXiv preprint arXiv:1611.01599},
  year      = {2016}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
