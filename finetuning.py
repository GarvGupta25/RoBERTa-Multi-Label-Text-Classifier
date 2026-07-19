from transformers import RobertaTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Quick test before building full dataset
sample = "You are an idiot and I hate you"
encoded = tokenizer(
    sample,
    max_length=128,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)

print("input_ids shape:", encoded['input_ids'].shape)
print("attention_mask shape:", encoded['attention_mask'].shape)
print("input_ids:", encoded['input_ids'])
print("decoded back:", tokenizer.decode(encoded['input_ids'][0]))

class ToxicDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.texts = df['comment_text'].values
        self.labels = df[label_cols].values.astype('float32')
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        labels = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(labels, dtype=torch.float32)
        }

# Load your already-split CSVs
train_df = pd.read_csv('train_split.csv')
val_df = pd.read_csv('val_split.csv')

train_dataset = ToxicDataset(train_df, tokenizer, max_length=128)
val_dataset = ToxicDataset(val_df, tokenizer, max_length=128)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

batch = next(iter(train_loader))
print("input_ids shape:", batch['input_ids'].shape)       # should be [32, 128]
print("attention_mask shape:", batch['attention_mask'].shape)  # should be [32, 128]
print("labels shape:", batch['labels'].shape)              # should be [32, 6]
print("labels sample:", batch['labels'][0])                # should be 6 floats of 0.0 or 1.0
print("label columns:", label_cols)

import torch
import numpy as np

label_counts = train_df[label_cols].sum()
total = len(train_df)

pos_weight = torch.tensor(
    [(total - label_counts[col]) / label_counts[col] for col in label_cols],
    dtype=torch.float32
)

print("pos_weight per label:")
for col, w in zip(label_cols, pos_weight):
    print(f"  {col}: {w:.1f}")

from transformers import RobertaModel
import torch
import torch.nn as nn

class RobertaToxicClassifier(nn.Module):
    def __init__(self, model_name='roberta-base', num_labels=6, dropout=0.1):
        super(RobertaToxicClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits

from transformers import get_linear_schedule_with_warmup
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = RobertaToxicClassifier(
    model_name='roberta-base',
    num_labels=6,
    dropout=0.1
).to(device)

pos_weight = pos_weight.to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = optim.AdamW(
    model.parameters(),
    lr=2e-5,
    weight_decay=0.01
)

num_epochs = 3
total_steps = len(train_loader) * num_epochs
warmup_steps = int(0.1 * total_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()  # for mixed precision training

def train_epoch(model, loader, optimizer, criterion, scheduler, device, scaler):
    model.train()
    epoch_loss = 0

    for batch_idx, batch in enumerate(loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        with autocast():
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        epoch_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}/{len(loader)} — Loss: {loss.item():.4f}")

    return epoch_loss / len(loader)

import numpy as np

def validate_epoch(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            epoch_loss += loss.item()
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return epoch_loss / len(loader), all_logits, all_labels

from sklearn.metrics import f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def find_optimal_thresholds(logits, labels):
    probs = 1 / (1 + np.exp(-logits))  # sigmoid
    thresholds = np.arange(0.1, 0.9, 0.05)
    optimal_thresholds = []

    for i, col in enumerate(label_cols):
        best_thresh = 0.5
        best_f1 = 0

        for thresh in thresholds:
            preds = (probs[:, i] >= thresh).astype(int)
            f1 = f1_score(labels[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        optimal_thresholds.append(best_thresh)
        print(f"  {col}: optimal threshold = {best_thresh:.2f}, F1 = {best_f1:.4f}")

    return optimal_thresholds

def compute_metrics(logits, labels, thresholds):
    probs = 1 / (1 + np.exp(-logits))

    per_label_f1 = []
    for i, (col, thresh) in enumerate(zip(label_cols, thresholds)):
        preds = (probs[:, i] >= thresh).astype(int)
        f1 = f1_score(labels[:, i], preds, zero_division=0)
        per_label_f1.append(f1)
        print(f"  {col} F1: {f1:.4f}")

    macro_f1 = np.mean(per_label_f1)
    print(f"\n  Macro F1: {macro_f1:.4f}")

    return per_label_f1, macro_f1, probs, thresholds

def plot_confusion_matrices(logits, labels, thresholds):
    probs = 1 / (1 + np.exp(-logits))
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, (col, thresh) in enumerate(zip(label_cols, thresholds)):
        preds = (probs[:, i] >= thresh).astype(int)
        cm = confusion_matrix(labels[:, i], preds)

        sns.heatmap(
            cm, annot=True, fmt='d', ax=axes[i],
            cmap='Blues',
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1']
        )
        axes[i].set_title(f'{col}\nF1: {f1_score(labels[:,i], preds, zero_division=0):.3f}')

    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=150)
    plt.show()

import os

best_macro_f1 = 0
best_thresholds = [0.5] * 6

for epoch in range(num_epochs):
    print(f"\n{'='*50}")
    print(f"EPOCH {epoch+1}/{num_epochs}")
    print(f"{'='*50}")

    train_loss = train_epoch(
        model, train_loader, optimizer,
        criterion, scheduler, device, scaler
    )
    print(f"\nTrain Loss: {train_loss:.4f}")

    val_loss, val_logits, val_labels = validate_epoch(
        model, val_loader, criterion, device
    )
    print(f"Val Loss: {val_loss:.4f}")

    print("\nFinding optimal thresholds...")
    thresholds = find_optimal_thresholds(val_logits, val_labels)

    print("\nPer-label metrics:")
    per_label_f1, macro_f1, _, _ = compute_metrics(
        val_logits, val_labels, thresholds
    )

    # Save best model
    if macro_f1 > best_macro_f1:
        best_macro_f1 = macro_f1
        best_thresholds = thresholds
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'macro_f1': macro_f1,
            'thresholds': thresholds,
        }, 'best_model.pt')
        print(f"\n  New best model saved! Macro F1: {macro_f1:.4f}")

print(f"\nTraining complete. Best Macro F1: {best_macro_f1:.4f}")
print(f"Best thresholds: {best_thresholds}")

# Plot confusion matrices for best model
print("\nGenerating confusion matrices...")
plot_confusion_matrices(val_logits, val_labels, best_thresholds)

import pickle

# Save model weights
torch.save({
    'model_state_dict': model.state_dict(),
    'thresholds': best_thresholds,
    'label_cols': label_cols,
    'macro_f1': best_macro_f1,
}, 'roberta_toxic_classifier.pt')

# Download
from google.colab import files
files.download('roberta_toxic_classifier.pt')

