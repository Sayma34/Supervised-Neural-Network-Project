# GNN-Based BERT for Understanding Context from Music

A multimodal neural-network framework for **music context understanding** that combines semantic representations from **DistilBERT** with structural audio representations from **Graph Neural Networks (GNNs)**.

The project was developed for **Neural Networks (CSE425 / EEE474 / CSE715)** and covers four progressively advanced tasks: text-based multi-label music tagging, graph-based audio understanding, GNN–BERT multimodal fusion, and cross-modal audio–text retrieval.

**Authors:** Kazi Adib Haq, Sayma Shahid, Touhid Ahmed

---

## Table of Contents

- [Project Overview](#project-overview)
- [Motivation](#motivation)
- [System Architecture](#system-architecture)
- [Project Tasks](#project-tasks)
- [Datasets](#datasets)
- [Audio Graph Construction](#audio-graph-construction)
- [Model Components](#model-components)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Configuration](#configuration)
- [Evaluation Metrics](#evaluation-metrics)
- [Demo and EDA](#demo-and-eda)
- [Reproducibility](#reproducibility)
- [Limitations and Notes](#limitations-and-notes)
- [Future Improvements](#future-improvements)
- [Acknowledgements](#acknowledgements)

---

## Project Overview

Music carries several forms of context simultaneously: genre, mood, instrumentation, timbre, rhythm, harmony, semantic tags, and natural-language descriptions. A single representation is often insufficient to capture all of these relationships.

This project therefore combines two complementary views of music:

1. **Semantic context** from text, tags, or captions using DistilBERT.
2. **Structural context** from audio segments represented as a graph and processed using GraphSAGE or GAT.

The two modalities are later joined through **cross-attention**, while a final contrastive-learning stage aligns graph-based audio embeddings with text embeddings in a shared representation space.

The complete project pipeline is:

```text
                         ┌──────────────────────────┐
                         │      Music Context       │
                         └────────────┬─────────────┘
                                      │
                     ┌────────────────┴────────────────┐
                     │                                 │
              Audio / Structure                    Text / Tags
                     │                                 │
            22.05 kHz resampling                 DistilBERT tokenizer
                     │                                 │
             3-second segments                    DistilBERT encoder
                     │                                 │
        Chroma + MFCC + Spectral                  Token embeddings
              Contrast features                   H_text and CLS
                     │                                 │
            Music structure graph                     │
                     │                                 │
              GraphSAGE / GAT                         │
                     │                                 │
              Graph embedding g                       │
                     └──────────────┬──────────────────┘
                                    │
                          Cross-Attention Fusion
                                    │
                          Fused representation z
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
             Multi-label tagging              Emotion regression
                    │
                    └───────────────┬───────────────┘
                                    │
                         Contrastive alignment
                              (InfoNCE)
                                    │
                         Audio ↔ Text retrieval
```

---

## Motivation

Conventional CNN- or RNN-based music models are effective at learning local spectral or temporal patterns, but they do not explicitly model long-range relationships between repeated or acoustically similar parts of a song.

This project addresses that limitation by representing each track as a graph:

- **Nodes** represent fixed-length music segments.
- **Temporal edges** connect consecutive segments.
- **Similarity edges** connect acoustically related non-adjacent segments.
- **GNN message passing** propagates information across those relationships.

BERT contributes a complementary semantic representation, allowing the system to connect structural audio information with text-based music context.

---

## System Architecture

The repository implements the complete four-task architecture defined in the project specification.

| Task | Model | Main Objective |
|---|---|---|
| **Task 1** | DistilBERT multi-label classifier | Understand music context from text/tags |
| **Task 2** | GraphSAGE / GAT | Learn structural audio representations |
| **Baseline B2** | 4-layer 2D CNN | Compare graph learning against mel-spectrogram learning |
| **Task 3** | GNN + BERT cross-attention | Fuse structural and semantic representations |
| **Task 4** | GNN–BERT dual encoder + InfoNCE | Align audio and text for cross-modal retrieval |

---

## Project Tasks

### Task 1 — DistilBERT Multi-Label Music Tagging

Task 1 uses a BERT-family encoder to predict multiple music-context tags from textual tag information.

The implementation uses:

```text
distilbert-base-uncased
```

The classifier consists of:

```text
Text
  ↓
Tokenizer
  ↓
DistilBERT
  ↓
CLS representation
  ↓
Dropout
  ↓
Linear classification head
  ↓
50 tag logits
  ↓
Sigmoid probabilities
```

The model uses `BCEWithLogitsLoss`, which is appropriate because a music clip may have multiple active labels simultaneously.

The BERT module also exposes:

- the full contextual token representation `H_text`
- the CLS embedding

These outputs are reused by Task 3 and Task 4.

---

### Task 2 — GNN on Music Structure Graphs

Task 2 models each track as a graph rather than a flat spectrogram.

The repository supports:

- **GraphSAGE**
- **Graph Attention Network (GAT)**

The default GNN configuration is:

```text
Input dimension:   39
Hidden dimension:  128
Graph output:      64
Number of layers:  2
Dropout:            0.2
Readout:            Global mean pooling
```

For each graph:

```text
Node features
    ↓
GNN layer(s)
    ↓
Batch normalization
    ↓
ReLU + dropout
    ↓
Global mean pooling
    ↓
Graph representation g
    ↓
Classification head
```

The implementation supports both:

- single-label genre classification
- multi-label tag prediction

---

### Baseline B2 — 2D CNN on Log-Mel Spectrograms

A four-block convolutional network is included as an audio baseline.

The CNN operates directly on:

```text
128 × T log-mel spectrograms
```

Architecture:

```text
Conv Block 1:  1   → 32 channels
Conv Block 2:  32  → 64 channels
Conv Block 3:  64  → 128 channels
Conv Block 4:  128 → 256 channels
        ↓
Adaptive average pooling
        ↓
Fully connected classifier
```

This baseline provides a direct comparison between conventional spectrogram learning and graph-based structural learning.

---

### Task 3 — GNN–BERT Cross-Attention Fusion

Task 3 combines the audio graph representation with contextual BERT token embeddings.

The graph representation produces the **query**, while the text sequence provides the **keys and values**:

```text
Q = g W_Q
K = H_text W_K
V = H_text W_V
```

Scaled dot-product attention is then computed as:

```text
Attention(Q, K, V) = softmax(QKᵀ / √d) V
```

The resulting text-aware context vector is concatenated with the graph embedding:

```text
z = [g ; context]
```

The fused representation is used for:

- multi-label tag prediction
- optional valence/arousal emotion regression

The code also supports an **early-concatenation ablation** using:

```text
[g ; t_CLS]
```

instead of cross-attention.

---

### Task 4 — Cross-Modal Contrastive Alignment

Task 4 learns a shared embedding space between:

- graph-based audio representations
- BERT-based text representations

Both modalities are projected to the same dimensionality and L2-normalized.

The model optimizes a symmetric **InfoNCE** objective:

```text
Audio → Text loss
        +
Text → Audio loss
        ↓
      / 2
```

The default contrastive configuration uses:

```text
Projection dimension: 64
Temperature:          0.07
```

Retrieval is evaluated in both directions using:

- Recall@1
- Recall@5
- Recall@10

---

## Datasets

The project uses two primary data sources.

### FMA-Small

Used for audio-based graph learning and genre classification.

Repository/report setup:

- **8,000 tracks**
- **8 root genres**
- approximately **30 seconds per track**
- used for audio features, graph construction, GNN training, and the CNN baseline

### MagnaTagATune

Used for multi-label music-context learning.

The project works with the most frequent music tags and uses a **top-50 tag subset** for classification.

The final report records:

```text
Filtered context examples: 11,108
Training samples:            8,886
Validation samples:          1,111
Test samples:                1,111
```

The split follows an approximately **80/10/10** train/validation/test partition.

---

## Audio Graph Construction

### Preprocessing

Audio is standardized using:

```text
Sample rate:        22,050 Hz
Target duration:    30 seconds
Segment duration:   3 seconds
Mel bins:           128
Chroma bins:        12
MFCC coefficients:  20
FFT size:           2048
Hop length:         512
```

Short tracks are padded and longer tracks are trimmed to the target duration.

---

### Node Features

Each 3-second segment becomes one graph node.

The node representation is a **39-dimensional vector**:

```text
12 Chroma
+ 20 MFCC
+ 7 Spectral Contrast
----------------------
= 39 dimensions
```

The segment vectors are normalized before graph construction.

---

### Graph Edges

Three edge types are supported:

1. **Temporal adjacency**

```text
segment i ↔ segment i+1
```

2. **Acoustic similarity**

Non-adjacent segments are connected when cosine similarity exceeds the configured threshold.

Default:

```text
cosine similarity ≥ 0.70
```

3. **Self-loops**

Optional self-connections are added to each node.

The graph builder exports PyTorch Geometric `Data` objects and can save them as:

- `.pt`
- `.json`

---

## Model Components

The source code is modularized so that individual components can be trained or replaced independently.

### `audio_features.py`

Handles:

- audio loading
- resampling
- normalization
- padding/truncation
- log-mel spectrogram extraction
- chroma extraction
- MFCC extraction
- spectral contrast
- fixed-length segmentation
- 39-dimensional graph-node feature generation

### `graph_builder.py`

Constructs PyTorch Geometric graphs using:

- temporal edges
- acoustic-similarity edges
- optional self-loops
- cosine-similarity edge weights

### `bert_encoder.py`

Implements:

- DistilBERT/BERT text encoding
- multi-label classification
- token-level contextual embeddings
- CLS representations

### `gnn_model.py`

Implements:

- GraphSAGE
- GAT
- graph-level pooling
- genre classification
- multi-label classification
- graph embedding extraction

### `cnn_baseline.py`

Implements the 2D CNN spectrogram baseline.

### `fusion_model.py`

Implements:

- multi-head cross-attention
- early-concatenation ablation
- tag classification
- optional valence/arousal prediction
- multi-task loss

### `contrastive.py`

Implements:

- audio projection head
- text projection head
- normalized shared embedding space
- symmetric InfoNCE loss
- Audio → Text retrieval
- Text → Audio retrieval
- Recall@K

### `dataset.py`

Provides multimodal PyTorch/PyG data loading for:

- audio files
- text descriptions
- tag labels
- genre labels
- emotion labels
- precomputed graph loading
- multimodal mini-batch collation

### `train.py`

Contains unified train/evaluation loops for:

- Task 1 BERT
- Task 2 GNN
- Task 3 fusion
- Task 4 contrastive learning

### `evaluate.py`

Computes:

- Macro-F1
- Micro-F1
- Macro Precision
- Macro Recall
- mean AUC-PR
- multiclass accuracy
- weighted F1
- MAE
- R²
- graph coherence

---

## Results

The repository includes `metrics.json` and a final report containing the following model comparison.

> **Important:** The table below reproduces the results recorded in the repository artifacts. Task 1 also has detailed held-out test results documented separately in the report.

| Model | Macro-F1 | Micro-F1* | AUC-PR | MAE (Emotion) | R@5 |
|---|---:|---:|---:|---:|---:|
| Random Baseline | 0.05 | 0.08 | 0.12 | — | 0.02 |
| CNN Mel-Spectrogram | 0.41 | 0.46 | 0.38 | 1.25 | — |
| Task 1: DistilBERT | **0.315** | **0.427** | **0.455** | — | — |
| Task 2: GNN-Only | 0.520 | 0.585 | 0.470 | 1.10 | — |
| Early Concat Ablation | 0.560 | 0.630 | 0.510 | — | — |
| Task 3: Cross-Attention Fusion | **0.610** | **0.700** | **0.550** | **0.92** | — |
| Task 4: Contrastive Dual Encoder | 0.550 | 0.640 | 0.500 | — | **0.380** |

\*Micro-F1 values are reported in the final report; `metrics.json` stores Macro-F1, AUC-PR, emotion MAE, and R@5.

### Task 1 Detailed Test Results

The DistilBERT baseline was trained for three epochs.

Validation progression:

| Epoch | Validation Loss | Macro-F1 |
|---:|---:|---:|
| 1 | 0.1999 | 0.1501 |
| 2 | 0.1865 | 0.2446 |
| 3 | 0.1804 | 0.2936 |

Held-out test results:

| Metric | Score |
|---|---:|
| Macro-F1 | **0.3150** |
| Micro-F1 | **0.4268** |
| AUC-PR | **0.4547** |

Selected high-performing tags reported in the project:

| Tag | F1 |
|---|---:|
| Indian | 0.7619 |
| Classical | 0.6927 |
| Harpsichord | 0.6792 |
| Female | 0.6667 |
| Rock | 0.6637 |

### Cross-Modal Retrieval

The repository reports:

```text
Recall@1  = 0.16
Recall@5  = 0.38
Recall@10 = 0.52
```

for the contrastive audio–text retrieval stage.

---

## Repository Structure

```text
gnn-bert-music-context/
│
├── README.md
├── requirements.txt
├── config.yaml
│
├── notebooks/
│   ├── colab_master_pipeline.ipynb
│   ├── eda.ipynb
│   └── demo_context.ipynb
│
├── src/
│   ├── __init__.py
│   ├── audio_features.py
│   ├── graph_builder.py
│   ├── bert_encoder.py
│   ├── gnn_model.py
│   ├── cnn_baseline.py
│   ├── fusion_model.py
│   ├── contrastive.py
│   ├── dataset.py
│   ├── train.py
│   └── evaluate.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
│
├── results/
│   ├── metrics.json
│   ├── plots/
│   ├── checkpoints/
│   └── retrieval_examples/
│
└── report/
    └── final_report.tex
```

If your GitHub repository keeps the uploaded files at the root rather than in `notebooks/`, `src/`, and `report/`, either move them into the structure above or update the paths in this section.

---

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd gnn-bert-music-context
```

### 2. Create a virtual environment

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Main dependencies include:

- PyTorch
- PyTorch Geometric
- Hugging Face Transformers
- Accelerate
- Librosa
- SoundFile
- NetworkX
- scikit-learn
- pandas
- NumPy
- SciPy
- Matplotlib
- Seaborn
- PyYAML
- Jupyter

---

## Running the Project

### Recommended: Google Colab

The master notebook is designed to provide an end-to-end project walkthrough.

Open:

```text
notebooks/colab_master_pipeline.ipynb
```

Use a GPU runtime:

```text
Runtime → Change runtime type → T4 GPU
```

The notebook is organized into:

```text
Section 0  Environment setup
Section 1  Dataset initialization
Section 2  Task 1 — BERT
Section 3  Task 2 — GNN / CNN
Section 4  Task 3 — GNN–BERT fusion
Section 5  Task 4 — Contrastive alignment
Section 6  Metrics and visualizations
```

The environment cell installs the required PyTorch Geometric, Transformers, audio-processing, and evaluation packages.

---

### Run EDA

Open:

```text
notebooks/eda.ipynb
```

The notebook covers:

- FMA-Small genre distribution
- MagnaTagATune tag-frequency analysis
- music-graph statistics

---

### Run the Demo

Open:

```text
notebooks/demo_context.ipynb
```

The demo walks through:

```text
music segments
    ↓
graph construction
    ↓
graph visualization
    ↓
context prediction visualization
```

The current notebook uses a simulated segment pattern for its graph demonstration and illustrative prediction scores, making it useful for explaining the architecture and graph behaviour without requiring a full dataset run.

---

## Configuration

The main hyperparameters are stored in `config.yaml`.

### Audio

```yaml
sample_rate: 22050
duration: 30
n_mels: 128
n_chroma: 12
n_mfcc: 20
hop_length: 512
n_fft: 2048
segment_duration: 3.0
similarity_threshold: 0.70
```

### Text

```yaml
model_name: distilbert-base-uncased
max_token_length: 128
num_tags: 50
```

### GNN

```yaml
type: GraphSAGE
hidden_dim: 128
output_dim: 64
num_layers: 2
dropout: 0.2
readout: mean
```

### Fusion

```yaml
d_model: 128
n_heads: 4
dropout: 0.1
fusion_type: cross_attention
```

### Contrastive Learning

```yaml
projection_dim: 64
temperature: 0.07
```

### Training

```yaml
batch_size: 32
learning_rate: 1.0e-4
weight_decay: 1.0e-4
epochs_bert: 5
epochs_gnn: 15
epochs_fusion: 15
epochs_contrastive: 15
patience: 4
```

Individual notebooks may override these values for specific experiments.

---

## Evaluation Metrics

### Multi-Label Classification

The project evaluates music-tag prediction using:

```text
Macro-F1
Micro-F1
Macro Precision
Macro Recall
Mean AUC-PR
```

Macro-F1 gives every tag equal importance, while Micro-F1 aggregates predictions globally.

---

### Genre Classification

For single-label classification:

```text
Accuracy
Macro-F1
Weighted-F1
```

---

### Emotion Regression

For valence and arousal:

```text
Mean Absolute Error (MAE)
R²
```

---

### Cross-Modal Retrieval

For audio–text alignment:

```text
Audio → Text: Recall@1, Recall@5, Recall@10
Text → Audio: Recall@1, Recall@5, Recall@10
```

---

### Graph Coherence

The evaluation module also provides a graph-coherence score measuring the fraction of graph edges whose connected nodes exceed a specified cosine-similarity threshold.

---

## Demo and EDA

The repository contains three notebooks serving different purposes:

| Notebook | Purpose |
|---|---|
| `colab_master_pipeline.ipynb` | Main architecture and project pipeline |
| `eda.ipynb` | Dataset and graph-structure exploration |
| `demo_context.ipynb` | Single-track-style graph and prediction demonstration |

The EDA/demo notebooks include several fixed or simulated values to make the visual workflow reproducible without requiring a complete dataset download. They should therefore be interpreted as architecture/demonstration notebooks rather than independent experimental verification of every reported benchmark.

---

## Reproducibility

The project uses a fixed random seed:

```python
seed = 42
```

For the closest reproduction:

1. use the same dataset versions
2. retain the same train/validation/test split
3. use the supplied `config.yaml`
4. install dependencies from `requirements.txt`
5. use a CUDA-enabled PyTorch environment where available
6. keep random seeds fixed

The code automatically falls back to CPU when CUDA is unavailable.

---

## Limitations and Notes

- Music tags are naturally imbalanced, which can reduce Macro-F1 for rare labels.
- A fixed classification threshold may not be optimal for every tag.
- Graph construction quality depends on segment duration and similarity threshold.
- Cross-modal performance depends strongly on the quality of audio–text alignment.
- The multimodal dataset loader contains a synthetic fallback path when audio feature extraction fails; production experiments should validate input files rather than rely on fallback data.
- The demo and some visualization cells use simulated values for interpretability.
- Metrics in `metrics.json` and the final report should be treated as the repository's recorded project results; independently reproducing them requires the original datasets, preprocessing state, checkpoints, and training runs.

---

## Future Improvements

Possible extensions include:

- per-label threshold optimization
- class-weighted BCE or focal loss
- stronger audio encoders
- pretrained music-specific transformers
- more advanced graph construction using beat/chord boundaries
- edge-conditioned or heterogeneous GNNs
- larger-scale MusicCaps alignment
- hard-negative contrastive mining
- learned temperature scheduling
- artist-disjoint evaluation
- calibration analysis
- attention-map inspection
- retrieval case studies with real audio examples

---

## Acknowledgements

This project uses open-source libraries and pretrained models from:

- [PyTorch](https://pytorch.org/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [Librosa](https://librosa.org/)
- [scikit-learn](https://scikit-learn.org/)
- [NetworkX](https://networkx.org/)

The project architecture is built around the FMA and MagnaTagATune datasets, with the broader project specification also considering datasets such as MusicCaps and DEAM for multimodal and emotion-related extensions.

---

## Academic Use

This repository was created as an academic neural-networks project. If you reuse or extend the code, please cite the original datasets, pretrained models, and libraries used in your experiments.
