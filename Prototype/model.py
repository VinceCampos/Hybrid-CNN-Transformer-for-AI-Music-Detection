import torch
import torch.nn as nn
import os
import pandas as pd
from transformers import ASTModel, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from dataset import MusicDetectionDataset

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE      = 2
EPOCHS          = 15       # increased: 10 epochs was not enough to converge
LR_CNN_HEAD     = 3e-4     # higher LR for randomly-init CNN + classifier
LR_AST          = 5e-6     # very low LR for fine-tuning pretrained AST backbone
WARMUP_EPOCHS   = 2        # cosine schedule warmup
NUM_WORKERS     = 2
CHECKPOINT_PATH = "model_checkpoint.pth"
BEST_CKPT_PATH  = "best_model.pth"

# Label smoothing prevents overconfident predictions that don't generalise.
# 0.1 means positive labels become 0.9, negative become 0.1.
LABEL_SMOOTHING = 0.1

# Freeze AST backbone for first N epochs — only train CNN + classifier.
# This prevents the pretrained AST weights from being corrupted early
# before the randomly-init CNN has learned meaningful features.
AST_FREEZE_EPOCHS = 2

# ── IMPROVED MODEL ────────────────────────────────────────────────────────────
#
# Changes from original:
#
# 1. DEEPER CNN (4 layers + BatchNorm + residual)
#    Original 2-layer CNN without normalisation couldn't generalise to diverse
#    audio. BatchNorm stabilises training across different mastering styles.
#    Residual preserves the original mel signal so CNN artifacts don't corrupt
#    the AST input.
#
# 2. MULTI-TOKEN POOLING (CLS + mean pool → 1536 dims)
#    Original only used CLS token (global summary). Concatenating mean of all
#    sequence tokens adds local temporal context — critical for detecting
#    subtle AI generation artifacts that appear in specific frequency bands.
#
# 3. DEEPER CLASSIFIER HEAD (3 layers + Dropout)
#    Single linear layer cannot learn non-linear boundary between AI audio
#    and heavily mastered human music — they overlap in feature space.
#    Dropout(0.4) prevents overfitting to training distribution.
#
class HybridASTDetector(nn.Module):
    def __init__(self, ast_backbone):
        super().__init__()
        self.ast = ast_backbone

        # Deeper CNN with BatchNorm and residual connection
        self.cnn_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.cnn_block2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        # Downsample residual to match 2x MaxPool2d spatial reduction
        self.residual_downsample = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1),
            nn.MaxPool2d(kernel_size=4),
        )

        # Deeper classifier: 1536 → 512 → 128 → 1
        self.classifier = nn.Sequential(
            nn.Linear(768 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        residual = x.unsqueeze(1)

        cnn_out = self.cnn_block1(residual)
        cnn_out = self.cnn_block2(cnn_out)

        res = self.residual_downsample(residual)
        if res.shape != cnn_out.shape:
            res = torch.nn.functional.interpolate(
                res, size=cnn_out.shape[2:], mode='bilinear', align_corners=False
            )
        cnn_out = cnn_out + res
        cnn_out = cnn_out.squeeze(1)

        if cnn_out.shape[-1] != 1024 or cnn_out.shape[-2] != 128:
            cnn_out = torch.nn.functional.interpolate(
                cnn_out.unsqueeze(1), size=(128, 1024),
                mode='bilinear', align_corners=False
            ).squeeze(1)

        ast_out   = self.ast(cnn_out).last_hidden_state   # (B, seq, 768)
        cls_token = ast_out[:, 0, :]                       # (B, 768)
        mean_pool = ast_out[:, 1:, :].mean(dim=1)          # (B, 768)
        combined  = torch.cat([cls_token, mean_pool], dim=1)  # (B, 1536)

        return self.classifier(combined)                   # (B, 1)

    def freeze_ast(self):
        for p in self.ast.parameters():
            p.requires_grad = False

    def unfreeze_ast(self):
        for p in self.ast.parameters():
            p.requires_grad = True


# ── LABEL SMOOTHING LOSS ──────────────────────────────────────────────────────
class BCEWithLogitsLossSmoothed(nn.Module):
    """
    BCEWithLogitsLoss with label smoothing.
    Prevents the model from becoming overconfident on training examples,
    which is a major cause of poor generalisation to diverse real-world audio.
    smoothing=0.1 → positive targets become 0.9, negative become 0.1.
    """
    def __init__(self, smoothing=0.1, pos_weight=None):
        super().__init__()
        self.smoothing   = smoothing
        self.pos_weight  = pos_weight

    def forward(self, logits, targets):
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return nn.functional.binary_cross_entropy_with_logits(
            logits, targets_smooth,
            pos_weight=self.pos_weight
        )


def get_accuracy(logits, labels):
    preds = (torch.sigmoid(logits) > 0.5).float()
    return (preds == labels).float().mean()


# ── TRAINING ──────────────────────────────────────────────────────────────────
def train_model():
    print(f"[*] Device: {device}")
    print(f"[*] Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | AST freeze for first {AST_FREEZE_EPOCHS} epochs")

    # Build model
    ast_backbone = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    model        = HybridASTDetector(ast_backbone).to(device)

    # Freeze AST initially — train only CNN + classifier first
    model.freeze_ast()
    print(f"[*] AST backbone frozen for first {AST_FREEZE_EPOCHS} epochs")

    # Two param groups: different LRs for AST vs CNN+head
    # AST starts frozen so its group has no effect until unfreeze
    optimizer = AdamW([
        {'params': model.ast.parameters(),        'lr': LR_AST},
        {'params': model.cnn_block1.parameters(), 'lr': LR_CNN_HEAD},
        {'params': model.cnn_block2.parameters(), 'lr': LR_CNN_HEAD},
        {'params': model.residual_downsample.parameters(), 'lr': LR_CNN_HEAD},
        {'params': model.classifier.parameters(), 'lr': LR_CNN_HEAD},
    ], weight_decay=0.01)

    # Cosine LR schedule with warmup — prevents training instability early on
    train_master  = pd.read_csv("train_80.csv")
    steps_per_epoch = max(1, int(len(train_master) * 0.20) // BATCH_SIZE)
    total_steps     = EPOCHS * steps_per_epoch
    warmup_steps    = WARMUP_EPOCHS * steps_per_epoch

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    scaler    = GradScaler()
    criterion = BCEWithLogitsLossSmoothed(
        smoothing=LABEL_SMOOTHING,
        pos_weight=torch.tensor([1.42]).to(device)
    )

    best_val_acc  = 0.0
    best_epoch    = 0

    for epoch in range(EPOCHS):

        # Unfreeze AST after freeze period — now fine-tune end-to-end
        if epoch == AST_FREEZE_EPOCHS:
            model.unfreeze_ast()
            print(f"\n[*] Epoch {epoch+1}: AST backbone unfrozen — fine-tuning end-to-end")

        # ── Train ──────────────────────────────────────────────────────────
        model.train()

        # Sample 20% of training data each epoch (keeps your original strategy)
        train_sub = train_master.sample(frac=0.15, random_state=epoch).reset_index(drop=True)
        train_sub.to_csv("current_run.csv", index=False)

        train_loader = DataLoader(
            MusicDetectionDataset("current_run.csv"),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )

        total_loss = 0.0
        total_acc  = 0.0
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [TRAIN]", colour='cyan')

        for mels, labels in train_loop:
            mels, labels = mels.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast():
                logits = model(mels).squeeze(-1)
                loss   = criterion(logits, labels)

            scaler.scale(loss).backward()

            # Gradient clipping — prevents exploding gradients with deep network
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            acc = get_accuracy(logits, labels).item()
            total_loss += loss.item()
            total_acc  += acc
            train_loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.2%}")

        avg_loss = total_loss / len(train_loader)
        avg_acc  = total_acc  / len(train_loader)
        print(f"  Epoch {epoch+1} train — loss: {avg_loss:.4f} | acc: {avg_acc:.2%}")

        # ── Validate every epoch (not just epoch 10) ───────────────────────
        # Validating only at the end means you have no visibility into
        # overfitting or underfitting until it's too late to act.
        val_loader = DataLoader(
            MusicDetectionDataset("val_10.csv"),
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )

        model.eval()
        val_acc  = 0.0
        val_loss = 0.0
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [VAL]", colour='green')

        with torch.no_grad():
            for v_mels, v_labels in val_loop:
                v_mels, v_labels = v_mels.to(device, non_blocking=True), v_labels.to(device, non_blocking=True)
                with autocast():
                    v_logits = model(v_mels).squeeze(-1)
                    v_loss   = criterion(v_logits, v_labels)

                acc      = get_accuracy(v_logits, v_labels).item()
                val_acc  += acc
                val_loss += v_loss.item()
                val_loop.set_postfix(acc=f"{acc:.2%}")

        final_val_acc  = val_acc  / len(val_loader)
        final_val_loss = val_loss / len(val_loader)
        print(f"  Epoch {epoch+1} val   — loss: {final_val_loss:.4f} | acc: {final_val_acc:.2%}")

        # Save checkpoint every epoch
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'val_acc': final_val_acc,
        }, CHECKPOINT_PATH)

        # Save separate best checkpoint — always keep the best model
        if final_val_acc > best_val_acc:
            best_val_acc = final_val_acc
            best_epoch   = epoch + 1
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_acc': final_val_acc,
            }, BEST_CKPT_PATH)
            print(f"  ✅ New best model saved (epoch {best_epoch}, val_acc={best_val_acc:.2%})")

    print(f"\n✨ Training complete. Best val acc: {best_val_acc:.2%} at epoch {best_epoch}")
    print(f"   Best model saved to: {BEST_CKPT_PATH}")
    print(f"   Use {BEST_CKPT_PATH} in app.py — not {CHECKPOINT_PATH}")


if __name__ == '__main__':
    train_model()