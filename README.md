# Clu-L8 — Mini Language Model (MLM)

A character-level transformer built entirely from scratch in PyTorch, designed for mathematical and symbolic reasoning. No HuggingFace abstractions — every component hand-written as a learning and portfolio artifact.

---

## What is an MLM?

A **Mini Language Model** is a small transformer trained on a specific domain rather than general language. Clu-L8 is trained exclusively on mathematical expressions and symbolic tasks, making it fast, focused, and interpretable. Think of it as a calculator that learned arithmetic the same way GPT learned language — by predicting the next character, one at a time.

---

## Architecture

| Parameter | Value |
|---|---|
| Layers | 12 |
| Attention heads | 12 |
| Hidden dimension | 768 |
| FFN dimension | 3072 |
| Context window | 512 tokens |
| Total parameters | ~85.2M |
| Tokenizer | Character-level (1 char = 1 token) |
| Vocabulary size | 93 tokens |
| Normalization | Pre-LayerNorm (GPT-2 style) |
| Activation | GELU |

Pre-LayerNorm (`x = x + dropout(sublayer(norm(x)))`) was essential for training stability at this depth. Post-LN caused gradient instability at 12 layers, manifesting as output repetition collapse. Switching to Pre-LN immediately reduced loss from a ~2.7 plateau to ~1.4.

---

## Task Tokens

Clu-L8 uses **task tokens** as inline system prompts — a prefix that tells the model what kind of output to produce. This design mirrors the `[INST]` token convention in instruction-tuned models, arrived at independently.

| Token | Task | Example |
|---|---|---|
| `cl:` | Arithmetic (+ - *) | `cl: 42 + 58 = 100` |
| `hex` | Decimal → hexadecimal | `hex 255 = FF` |
| `oct` | Decimal → octal | `oct 8 = 10` |
| `bin` | Decimal → binary | `bin 10 = 1010` |
| `dec` | Base → decimal | `dec 0xFF = 255` |
| `bf:` | Brainfuck interpreter | `bf: +++++. = 5` |

---

## Training

### Phase 1 — Base pretraining

- **Dataset:** ~1,000,000 lines of synthetic math expressions
- **Result:** ~10% answer accuracy across all tasks (continual training, multiple sessions)

This phase established the model's understanding of arithmetic structure, number representation, and task-token semantics.

### Phase 2 — Reasoning experiment (A/B/C)

To improve multi-digit addition accuracy, three output encoding strategies were tested against each other. Each variant was trained on **50,000 lines, 5 epochs, cosine LR decay (3e-4 → 1e-5)**.

#### Training loss

| Mode | Epoch 1 | Epoch 5 |
|---|---|---|
| cols | 0.4510 | 0.2495 |
| place | 0.6282 | 0.3063 |
| reverse | 1.2394 | 1.0087 |

#### Evaluation (500 test cases each)

| Mode | Accuracy | Description |
|---|---|---|
| **cols** | **100.0%** 🏆 | Column-carry intermediate steps |
| place | 79.8% | Place-value digit split |
| reverse | 1.2% | Silent digit reversal |

**Winner: `cols` — 500/500 correct.**

The `cols` approach encodes intermediate carry information alongside the result, giving the model a scratchpad for multi-digit arithmetic — independently arriving at the core insight behind Chain-of-Thought reasoning.

---

## Results

```
cl: explain 6618 + 6238 = 12856  ✅
cl: explain 3278 + 9821 = 13099  ✅
cl: explain 8824 + 1471 = 10295  ✅
cl: explain 6908 + 9640 = 16548  ✅

📊 Accuracy: 500/500 = 100.0%
```

---

## Project Structure

```
clu-l8-mlm/
├── vocab.json          # 93-token character vocabulary
├── tokenizer.py        # Encode / decode
├── embeddings.py       # Token + sinusoidal positional encoding
├── attention.py        # Multi-head causal self-attention
├── transformer_block.py # Pre-LN transformer block
├── model.py            # Full MLM stack (~85.2M params)
├── data_generator.py   # Synthetic dataset generator (CLI)
├── train.py            # Training loop with continual learning + cosine decay
└── generate.py         # Evaluation (hardcoded prompts or file-based accuracy %)
```

---

## Usage

### Requirements

```bash
pip install torch
python --version  # 3.11 recommended
```

### Generate training data

```bash
python data_generator.py 200000
# With reversed outputs (A/B experiment):
python data_generator.py 50000 --reverse
```

### Train

```bash
python train.py data.txt --epochs 5 --lr 3e-4 --cosine --lr_min 1e-5
# Continual training (resumes from mlm.pt automatically):
python train.py new_data.txt --epochs 3 --lr 1e-4
```

### Evaluate

```bash
# Hardcoded prompts:
python generate.py

# File-based accuracy:
python generate.py test_data.txt --lines 500
```

---

## Roadmap

- [x] Base model — arithmetic (`cl:`)
- [x] Number conversion — `hex`, `oct`, `bin`, `dec`
- [x] Brainfuck interpreter — `bf:`
- [x] Reasoning experiment — `cols` / `place` / `reverse` A/B/C
- [ ] Step-by-step explain mode
- [ ] 5-digit arithmetic
- [ ] Extended Brainfuck (loops)

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

