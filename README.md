# Multi-Label Toxic Comment Classifier — RoBERTa Fine-tuning

Fine-tuned RoBERTa-base for multi-label toxic comment classification on the Jigsaw dataset (159,571 real-world comments), handling severe class imbalance across 6 toxicity categories with weighted loss, optimal threshold tuning, and mixed precision training.

**Final Macro F1: 0.6804 across 6 toxicity labels**

huggingface model deployed live - https://huggingface.co/spaces/hf-garv/roberta_multilabel_text_classifier

---

## Table of Contents

- [Project Overview](#project-overview)
- [Why This Problem is Hard](#why-this-problem-is-hard)
- [Dataset](#dataset)
- [Technical Architecture](#technical-architecture)
- [Training Pipeline](#training-pipeline)
- [Results](#results)
- [Key Engineering Decisions](#key-engineering-decisions)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [What's Next](#whats-next)

---

## Project Overview

Content moderation at scale is one of the most operationally critical NLP problems in industry — every social platform, forum, and comments section needs it. This project fine-tunes RoBERTa-base (a robustly optimized BERT pretraining approach) for **multi-label** toxic comment detection, where a single comment can simultaneously belong to multiple toxicity categories.

The project covers the complete ML pipeline from raw data through EDA, class imbalance handling, tokenization, fine-tuning with mixed precision, threshold optimization, per-label evaluation, and model persistence — mirroring a production-grade fine-tuning workflow.

---

## Why This Problem is Hard

**Multi-label, not multi-class:** A comment can be simultaneously toxic, obscene, AND insulting. This rules out softmax (which assumes mutual exclusivity) — each label requires an independent sigmoid output.

**Severe class imbalance:** The dataset is heavily skewed toward clean comments:

```
Clean comments:    ~90% of dataset
Toxic:             ~10%
Obscene:           ~8%
Insult:            ~5%
Severe toxic:      ~1%
Identity hate:     ~0.9%
Threat:            ~0.3%
```

A naive model predicting "clean" for everything achieves 90% accuracy while being completely useless. Standard cross-entropy loss fails here — the model learns to ignore minority classes.

**Rare classes are the most important ones:** Threats and severe toxicity are the rarest categories but the highest-priority for real content moderation systems — exactly the ones most vulnerable to being ignored by an imbalanced model.

---

## Dataset

**Source:** Jigsaw Toxic Comment Classification Challenge (Kaggle)

**Size:** 159,571 labeled comments from Wikipedia talk page edits

**Labels (6, binary each):**
- `toxic` — general toxicity
- `severe_toxic` — extreme toxicity
- `obscene` — obscene language
- `threat` — threatening content
- `insult` — insulting language
- `identity_hate` — hate based on identity characteristics

**Split strategy:** Iterative stratification (via `iterstrat` library) for multi-label aware train/validation splitting, preserving label distribution in both splits. Standard `train_test_split` with stratify fails for multi-label problems — iterative stratification handles co-occurring labels correctly.

```
Train: ~127,657 samples (80%)
Val:   ~31,914 samples  (20%)
```

---

## Technical Architecture

### Model

```
RoBERTa-base (pretrained, 12 layers, 768 hidden, 125M parameters)
    ↓
<s> token final hidden state  [batch, 768]
    ↓
Dropout(0.1)                  [batch, 768]
    ↓
Linear(768 → 6)               [batch, 6]
    ↓
Raw logits (sigmoid applied   [batch, 6]
at inference only)
```

**Why RoBERTa over BERT:**
RoBERTa removes Next Sentence Prediction (found to hurt rather than help), trains with dynamic masking (different tokens masked each epoch), more data, and larger batch sizes. Consistently outperforms BERT-base on classification benchmarks with identical architecture and inference cost.

**Why `<s>` token (not `[CLS]`):**
RoBERTa uses `<s>` (token ID 0) as its sequence summary token, equivalent to BERT's `[CLS]` (token ID 101). Its final hidden state encodes a summary representation of the entire input sequence, used as the classification input.

**Why sigmoid not softmax:**
Sigmoid applies an independent probability (0→1) to each label with no constraint that they sum to 1. Softmax forces probabilities to sum to 1, implying only one class can be true — fundamentally wrong for multi-label classification.

### Tokenization

```python
RobertaTokenizer.from_pretrained('roberta-base')
max_length = 128
padding = 'max_length'
truncation = True
```

**Why max_length=128 not 512:**
EDA showed >90% of comments fall under 100 words. Attention complexity is O(n²) in sequence length — halving sequence length from 256 to 128 quarters attention computation. Negligible information loss for this dataset with 4× training speed improvement.

**Why RobertaTokenizer specifically:**
RoBERTa uses BPE (Byte Pair Encoding) with a 50,265 token vocabulary. Using BertTokenizer with a RoBERTa model would map tokens to incorrect embedding indices — the vocabulary tables are incompatible. Always match tokenizer to pretrained model.

---

## Training Pipeline

### Class Imbalance Handling — Weighted BCE Loss

```python
pos_weight[i] = num_negative_examples[i] / num_positive_examples[i]
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

For the "threat" class (0.3% positive rate):
```
pos_weight = 99.7 / 0.3 ≈ 332
```

This tells the loss function that missing a "threat" prediction costs 332× more than a false "non-threat" prediction — forcing the model to attend to rare positive examples rather than ignoring them to minimize overall loss.

### Optimizer — AdamW

```python
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
```

**Why AdamW not Adam:**
Standard Adam applies weight decay incorrectly for adaptive optimizers — the decay gets scaled by the adaptive learning rate, making it ineffective. AdamW decouples weight decay from the gradient update, applying it directly to weights. For Transformer fine-tuning this correction matters and AdamW consistently outperforms Adam.

### Learning Rate Schedule — Linear Warmup + Decay

```
Steps 0 → warmup_steps (10% of total):   lr increases linearly 0 → 2e-5
Steps warmup_steps → end:                lr decreases linearly 2e-5 → 0
```

Warmup prevents the first few updates (when gradients can be noisy from the newly initialized classifier head) from destroying pretrained RoBERTa weights before training stabilizes.

### Mixed Precision Training

```python
scaler = GradScaler()
with autocast():
    logits = model(input_ids, attention_mask)
    loss = criterion(logits, labels)
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
```

Uses float16 for forward pass (2× memory reduction, 2-3× faster on modern GPUs), float32 for gradient accumulation. GradScaler prevents gradient underflow by scaling loss before backward pass, then unscaling before optimizer step. Gradient clipping happens after unscaling to ensure correct norm comparison.

### Threshold Optimization — Per-Label Tuning

Default threshold of 0.5 is rarely optimal for imbalanced data. After training, per-label optimal thresholds are found by sweeping [0.1, 0.9] on validation set and maximizing F1 per label:

```python
for thresh in np.arange(0.1, 0.9, 0.05):
    preds = (probs[:, i] >= thresh).astype(int)
    f1 = f1_score(labels[:, i], preds, zero_division=0)
```

Optimal thresholds are saved alongside model weights for consistent inference.

---

## Results

### Training Summary

```
Train Loss (final): 0.2437
Val Loss (final):   0.4657
Epochs:             3
```

### Per-Label F1 Scores

| Label | F1 Score | Optimal Threshold | Class Frequency |
|-------|----------|-------------------|-----------------|
| Toxic | 0.8389 | 0.85 | ~10% |
| Obscene | 0.8300 | 0.85 | ~8% |
| Insult | 0.7535 | 0.85 | ~5% |
| Identity Hate | 0.5838 | 0.85 | ~0.9% |
| Threat | 0.5680 | 0.85 | ~0.3% |
| Severe Toxic | 0.5079 | 0.85 | ~1% |

**Macro F1: 0.6804**

### Analysis

Higher-frequency labels (toxic, obscene, insult) achieve strong F1 scores (0.75-0.84) with sufficient training examples. Rarer labels (severe_toxic, threat, identity_hate) achieve lower F1 (0.51-0.58) — expected given the extreme class imbalance even after pos_weight correction. All optimal thresholds converged to 0.85, indicating the model learned to be appropriately conservative — only predicting positive when highly confident, which is the correct behavior for content moderation systems where false positives (incorrectly flagging clean content) have significant user experience costs.

The train/val loss gap (0.24 vs 0.47) indicates mild overfitting — expected for fine-tuning a 125M parameter model. Addressed through dropout (0.1), weight decay (0.01), and limiting to 3 epochs.

---

## Key Engineering Decisions

**Multi-label stratified splitting** — standard sklearn stratify fails for multi-label problems. Iterative stratification correctly handles label co-occurrence, ensuring rare classes appear proportionally in both splits.

**BCEWithLogitsLoss over BCELoss** — numerically more stable (combines sigmoid and BCE in one operation using log-sum-exp trick), and directly accepts pos_weight for class balancing.

**Collecting all validation logits before computing metrics** — per-batch F1 is unreliable for rare classes (a batch may have zero positive examples for "threat"). Full-dataset metrics are the only meaningful evaluation.

**Saving optimal thresholds with model checkpoint** — inference without saved thresholds would default to 0.5, degrading performance on rare classes significantly. Checkpoint includes model weights, thresholds, label columns, and best macro F1.

---

## Installation

```bash
git clone https://github.com/GarvGupta25/your-repo-name
cd toxic-classifier
pip install torch transformers scikit-learn pandas numpy matplotlib seaborn iterstrat
```

---

## Usage

**Training:**
```python
# Set up dataset, dataloaders, model as in notebook
# Run training loop for 3 epochs
# Best model auto-saved to best_model.pt
```

**Inference:**
```python
import torch
from transformers import RobertaTokenizer

# Load checkpoint
checkpoint = torch.load('best_model.pt', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
thresholds = checkpoint['thresholds']
model.eval()

# Tokenize and predict
def predict(text):
    encoding = tokenizer(text, max_length=128, padding='max_length',
                         truncation=True, return_tensors='pt')
    with torch.no_grad():
        logits = model(encoding['input_ids'], encoding['attention_mask'])
    probs = torch.sigmoid(logits).numpy()[0]
    predictions = {col: (float(prob), bool(prob >= thresh)) 
                   for col, prob, thresh in zip(label_cols, probs, thresholds)}
    return predictions

print(predict("You are an absolute idiot"))
```

---

## File Structure

```
├── toxic_classifier.ipynb     # Full training notebook
├── best_model.pt              # Saved model weights + thresholds
├── train_split.csv            # Training split (iterative stratification)
├── val_split.csv              # Validation split
├── confusion_matrices.png     # Per-label confusion matrix visualization
└── README.md
```

---

## What's Next

- **REST API deployment** — FastAPI endpoint wrapping inference, returning per-label toxicity scores and binary predictions
- **Docker + AWS deployment** — containerized serving with CI/CD pipeline
- **QLoRA fine-tuning** — extend to GPT-style generative model fine-tuning with parameter-efficient training
- **Threshold calibration** — Platt scaling or isotonic regression for better-calibrated probability outputs
- **Ensemble** — combine RoBERTa predictions with a character-level model for better handling of intentional misspellings used to evade detection
