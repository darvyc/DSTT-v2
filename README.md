<p align="center">
  <h1 align="center">DSTT-T: Dynamic Semi-Trained Topology — Transformer</h1>
  <p align="center">
    <em>A hybrid transformer trained like a GPT — combinatorial partitioning, dual-flow attention, and evolutionary meta-optimisation fused into every layer.</em>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="Python">
    <img src="https://img.shields.io/badge/pytorch-2.0%2B-orange" alt="PyTorch">
    <img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License">
    <img src="https://img.shields.io/badge/version-3.0.0-brightgreen" alt="Version">
  </p>
</p>

---

## What is DSTT-T?

DSTT-T is a **hybrid transformer architecture** that replaces or augments every core transformer subsystem with a mechanism from the Dynamic Semi-Trained Topology (DSTT) framework — and trains exactly like GPT-2/3 using **next-token prediction**.

The model is a causal language model. You feed it tokens, it predicts the next token. The training objective, optimiser, LR schedule, and generation loop are identical to GPT. What's different is what happens *inside* each transformer block.

### The 6 Innovations Inside Each Block

| Component | Replaces | What it does |
|-----------|----------|--------------|
| **FDMP-E** | Embedding layer | Modality-structured embeddings with partition-aware positional encoding |
| **RP-MHA** | Multi-head attention | Ramsey-coherence determines head count and variable-width scope |
| **Dual-Flow Attention** | QKᵀ scoring | `score = QKᵀ/√d + α·CFM − β·AFM` — boosts relevant keys, suppresses incoherent ones |
| **ARM-FFN** | Feed-forward network | Partition-gated mixture-of-experts with CFM-AFM routing |
| **WCG** | Residual connection | Wittgenstein contextual gate — only lets useful signals through |
| **EML** | Hyperparameter tuning | Evolutionary meta-optimisation co-evolving architecture with weights |

### How a Block Works

```
Input x
  │
  ├─→ LayerNorm → RP-MHA (Ramsey-partitioned dual-flow attention)
  │                  │
  │   Wittgenstein Gate (w₁)
  │   ▼
  ├─→ x' = x + w₁ ⊙ attn_output
  │
  ├─→ LayerNorm → ARM-FFN (partition-gated mixture of experts)
  │                  │
  │   Wittgenstein Gate (w₂)
  │   ▼
  └─→ x'' = x' + w₂ ⊙ ffn_output
```

---

## Quick Start: Train on Shakespeare in 5 Minutes

### 1. Install

```bash
git clone https://github.com/your-org/dstt-transformer.git
cd dstt-transformer
pip install -e .
```

### 2. Download and prepare data

```bash
python prepare_data.py --download shakespeare
```

This downloads Tiny Shakespeare (~1MB), tokenizes it with a character-level tokenizer, and writes `data/train.bin` and `data/val.bin`.

### 3. Train

```bash
python train.py \
    --data_dir data \
    --config tiny \
    --max_steps 5000 \
    --batch_size 32 \
    --block_size 256 \
    --learning_rate 1e-3
```

You'll see output like:

```
12:00:01 | INFO | Model: DSTT-T tiny
12:00:01 | INFO |   Parameters: 15.2M (15,203,456)
12:00:01 | INFO | Training: 5000 steps, effective batch=32, ~40.9M tokens total
12:00:05 | INFO | step      10 | loss 4.2341 | ppl    68.98 | lr 1.50e-04 | tok/s 204,800
12:00:09 | INFO | step      20 | loss 3.8127 | ppl    45.27 | lr 3.00e-04 | tok/s 211,200
...
12:05:12 | INFO |   → val loss 1.5823 | val ppl 4.87
12:05:14 | INFO |   → sample: ROMEO: O, what light through yonder window...
```

### 4. Generate text

```bash
python generate.py \
    --checkpoint checkpoints/best.pt \
    --prompt "ROMEO:" \
    --max_tokens 500 \
    --temperature 0.8
```

### Or do it all in one script

```bash
python examples/train_shakespeare.py
```

---

## How Training Works (GPT-Style)

DSTT-T trains with the same autoregressive next-token prediction objective as GPT-2 and GPT-3. Here's the complete pipeline:

### Training Objective

Given a sequence of tokens `[t₀, t₁, ..., tₙ]`, the model learns to predict `[t₁, t₂, ..., tₙ₊₁]`. The loss is cross-entropy between the predicted and actual next tokens, averaged across all positions.

```
Input:   [The] [cat] [sat] [on]  [the]
Target:  [cat] [sat] [on]  [the] [mat]
```

This is identical to GPT — the only difference is the architecture producing the predictions.

### Optimiser and Schedule

| Setting | Value | Notes |
|---------|-------|-------|
| Optimiser | AdamW | Decoupled weight decay |
| Weight decay | 0.1 | Applied to 2D+ params only (not biases, norms) |
| β₁, β₂ | 0.9, 0.95 | Standard GPT values |
| LR schedule | Warmup → Cosine decay | Linear warmup, then cosine to `min_lr` |
| Gradient clipping | 1.0 | Global norm clipping |
| Gradient accumulation | Configurable | Effective batch = micro_batch × accumulation_steps |

### Learning Rate Schedule

```
LR
 │  ╱‾‾‾‾‾‾‾‾‾‾‾‾‾‾╲
 │ ╱                  ╲
 │╱                    ╲
 │                      ╲______
 └─────────────────────────────→ steps
   warmup    cosine decay    min_lr
```

The schedule is defined by three parameters:
- `warmup_steps`: Linear ramp from 0 to `learning_rate`
- `learning_rate`: Peak LR
- `min_lr`: Floor LR after decay

### Training Phases

**Standard training** (most common):
```bash
python train.py --config base --max_steps 100000
```

**With gradient accumulation** (for larger effective batches on limited hardware):
```bash
python train.py --config base --batch_size 8 --gradient_accumulation_steps 8
# effective batch size = 64
```

**Resume from checkpoint**:
```bash
python train.py --resume_from checkpoints/step_50000.pt
```

**Mixed precision** (faster on modern GPUs):
```bash
python train.py --config base --dtype bfloat16
```

### What Happens During Training

Every `log_interval` steps, you see:
```
step    500 | loss 2.1234 | ppl   8.37 | lr 3.00e-04 | tok/s 125,000 | tokens 16,000,000
```

Every `eval_interval` steps, validation runs:
```
  → val loss 2.3456 | val ppl 10.42
```

Every `sample_interval` steps, the model generates a sample:
```
  → sample: ROMEO: What light through yonder...
```

Checkpoints are saved every `save_interval` steps, and the best (lowest validation loss) is always saved as `checkpoints/best.pt`.

---

## Generation

DSTT-T generates text exactly like GPT: autoregressive sampling one token at a time.

### Sampling Methods

```python
from dstt import DSTTTransformer, DSTTConfig
from dstt.generate import generate_text
from dstt.tokenizer import CharTokenizer

# Load your trained model and tokenizer
model = ...
tokenizer = ...

# Greedy decoding (most likely token at each step)
text = generate_text(model, tokenizer, prompt="Hello", temperature=0.0)

# Standard sampling with temperature
text = generate_text(model, tokenizer, prompt="Hello", temperature=0.8)

# Top-k: only sample from the k most likely tokens
text = generate_text(model, tokenizer, prompt="Hello", top_k=50)

# Top-p (nucleus): sample from smallest set with cumulative prob > p
text = generate_text(model, tokenizer, prompt="Hello", top_p=0.95)

# Reduce repetition
text = generate_text(model, tokenizer, prompt="Hello", repetition_penalty=1.2)
```

### Command-Line Generation

```bash
# Interactive
python generate.py --checkpoint checkpoints/best.pt --prompt "Once upon"

# Multiple samples
python generate.py --checkpoint checkpoints/best.pt --num_samples 5

# Long, creative generation
python generate.py --checkpoint checkpoints/best.pt \
    --max_tokens 2000 --temperature 1.0 --top_k 100
```

---

## Data Preparation

### Option A: Character-Level (simple, zero dependencies)

```bash
# Download Tiny Shakespeare
python prepare_data.py --download shakespeare

# Or use your own text file
python prepare_data.py --input mydata.txt --tokenizer char
```

This creates:
```
data/
├── train.bin         # tokenized training data (memory-mapped)
├── val.bin           # tokenized validation data
└── tokenizer.json    # character vocabulary
```

### Option B: GPT-2 BPE (production, requires tiktoken)

```bash
pip install tiktoken
python prepare_data.py --input mydata.txt --tokenizer gpt2
```

Uses the GPT-2 BPE tokenizer (50,257 tokens). Better for large-scale training.

### Option C: Use any text directly (no prepare step)

Just put a `.txt` file in the data directory. The trainer will tokenize it on the fly:

```bash
mkdir data
cp my_corpus.txt data/
python train.py --data_dir data --config tiny
```

---

## Configuration Reference

### Model Configurations

| Config | Layers | d_model | Heads | Experts | Params | Use case |
|--------|--------|---------|-------|---------|--------|----------|
| `tiny` | 4 | 256 | 8 | 4 | ~15M | Debugging, learning, CPU training |
| `base` | 12 | 768 | 16 | 8 | ~125M | Research, single GPU |
| `large` | 24 | 1024 | 32 | 16 | ~350M | Multi-GPU |
| `xl` | 32 | 1536 | 48 | 32 | ~1.3B | Cluster training |

### All Training Arguments

```bash
python train.py --help
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `tiny` | Model size preset |
| `--data_dir` | `data` | Directory with train.bin/val.bin or .txt files |
| `--block_size` | `256` | Context window (sequence length) |
| `--batch_size` | `32` | Micro-batch size |
| `--gradient_accumulation_steps` | `1` | Accumulation steps (effective batch = batch × this) |
| `--max_steps` | `5000` | Total training steps |
| `--learning_rate` | `3e-4` | Peak learning rate |
| `--min_lr` | `3e-5` | Minimum learning rate |
| `--warmup_steps` | `200` | Linear warmup steps |
| `--weight_decay` | `0.1` | AdamW weight decay |
| `--grad_clip` | `1.0` | Gradient norm clipping |
| `--dtype` | `float32` | Training precision: float32, float16, bfloat16 |
| `--eval_interval` | `250` | Steps between validation |
| `--eval_steps` | `20` | Batches per validation |
| `--log_interval` | `10` | Steps between log prints |
| `--save_interval` | `1000` | Steps between checkpoints |
| `--sample_interval` | `500` | Steps between sample generation |
| `--resume_from` | `None` | Checkpoint path to resume from |
| `--compile` | `False` | Use torch.compile() |

---

## Project Structure

```
dstt-transformer/
├── README.md                    ← You are here
├── LICENSE
├── setup.py
├── pyproject.toml
├── requirements.txt
│
├── dstt/                        ← Core library
│   ├── __init__.py              ← Public API
│   ├── config.py                ← DSTTConfig (model architecture)
│   ├── train_config.py          ← TrainConfig (training hyperparameters)
│   ├── model.py                 ← DSTTTransformer, DSTTBlock
│   ├── attention.py             ← RP-MHA (Ramsey-partitioned dual-flow attention)
│   ├── embedding.py             ← FDMP-E (hybrid embedding layer)
│   ├── routing.py               ← ARM-FFN (partition-gated MoE)
│   ├── gating.py                ← WCG (Wittgenstein contextual gate)
│   ├── flow_matrices.py         ← CFM, AFM, DualFlowScoring
│   ├── partitioning.py          ← RamseyPartitioner
│   ├── evolution.py             ← EML (evolutionary meta-optimiser)
│   ├── losses.py                ← DSTTLoss, LoadBalanceLoss
│   ├── data.py                  ← TextDataset, MemmapDataset
│   ├── tokenizer.py             ← CharTokenizer, GPT2Tokenizer
│   ├── trainer.py               ← GPT-style Trainer class
│   ├── generate.py              ← Autoregressive text generation
│   └── utils.py                 ← Math helpers, RMSNorm, top-k softmax
│
├── train.py                     ← Training entry point
├── generate.py                  ← Generation entry point
├── prepare_data.py              ← Data preparation script
│
├── examples/
│   ├── quickstart.py            ← Minimal forward-pass example
│   ├── train_shakespeare.py     ← End-to-end: download → train → generate
│   ├── language_model.py        ← Training loop with dummy data
│   └── custom_config.py         ← Advanced configuration
│
├── tests/                       ← Unit tests (pytest)
│   ├── test_model.py
│   ├── test_attention.py
│   ├── test_flow_matrices.py
│   ├── test_partitioning.py
│   ├── test_routing.py
│   └── test_training.py         ← Tests for trainer, data, generation
│
└── scripts/
    └── benchmark.py             ← Throughput measurement
```

---

## Key Concepts for Newcomers

### "Trained like a GPT" — What does that mean?

GPT (Generative Pre-trained Transformer) models are trained with a simple objective: **predict the next word**. Given the sequence "The cat sat on the", the model should predict "mat" (or whatever comes next in the training data).

DSTT-T uses this exact same objective. The difference is the architecture that makes the prediction. A standard GPT uses vanilla multi-head attention and a fixed feed-forward network. DSTT-T replaces these with Ramsey-partitioned attention, dual-flow scoring, and partition-gated expert routing.

The training loop is identical to GPT:
1. Sample a random chunk of text from the training data
2. Feed tokens 1..N to the model
3. Compare the model's predictions against the actual tokens 2..N+1
4. Compute cross-entropy loss
5. Backpropagate and update weights
6. Repeat

### What is Dual-Flow Attention?

Standard attention asks: "How similar is my query to each key?" (dot product).

Dual-flow attention asks three questions:
1. "How similar?" (standard dot product — alignment)
2. "How relevant is this key to the current context?" (CFM — boost)
3. "Would attending here introduce contradictions?" (AFM — penalty)

The final score combines all three: `score = alignment + α·relevance − β·penalty`

### What is ARM-FFN?

Standard transformers apply the same feed-forward network to every token. ARM-FFN maintains multiple expert sub-networks and routes each token to the most appropriate experts based on coherence scores. This is similar to Mixture-of-Experts (MoE) architectures like Switch Transformer, but the routing decision is governed by DSTT's CFM-AFM scoring rather than a learned gate.

### What is the Wittgenstein Gate?

Named after Wittgenstein's principle that meaning comes from use in context. The gate is a simple sigmoid function that controls how much of each sub-layer's output actually enters the residual stream. If a token's attention or FFN output isn't contextually useful, the gate closes and the residual stream passes through unchanged. Think of it as a learned "relevance filter".

---

## Checkpoints

Checkpoints contain everything needed to resume training or run inference:

```python
checkpoint = {
    "model_state_dict": ...,     # Model weights
    "optimiser_state_dict": ..., # Optimiser state (for resume)
    "step": 50000,               # Training step
    "best_val_loss": 1.823,      # Best validation loss seen
    "tokens_seen": 819200000,    # Total tokens processed
    "config": DSTTConfig(...),   # Model architecture
    "train_config": TrainConfig(...),  # Training settings
}
```

### Load for inference

```python
import torch
from dstt import DSTTTransformer

ckpt = torch.load("checkpoints/best.pt", weights_only=False)
model = DSTTTransformer(ckpt["config"])
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
```

### Resume training

```bash
python train.py --resume_from checkpoints/step_50000.pt
```

---

## Testing

```bash
pip install pytest
pytest tests/ -v
```

---

## Citation

```bibtex
@article{dstt2026,
  title={DSTT-T: Dynamic Semi-Trained Topology as a Hybrid Transformer Architecture},
  year={2026},
  note={Technical Specification, Revision 3.0}
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
