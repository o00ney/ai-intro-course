#!/usr/bin/env python3
from pathlib import Path
import json
import sys
import csv
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image


# -----------------------------------------------------------------------------
# Tee — dual console + file logging
# -----------------------------------------------------------------------------

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


# -----------------------------------------------------------------------------
# Configurable parameters
# -----------------------------------------------------------------------------

SEED = 42
EPOCHS = 100
BATCH_SIZE = 8
LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
MIN_LR = 1e-5
NUM_CLASSES = 1
INPUT_H = 96
INPUT_W = 96
INPUT_CHANNELS = 3
DEEP_SUPERVISION = False
NB_FILTER = [32, 64, 128, 256, 512]
VAL_RATIO = 0.2
NUM_WORKERS = 4
IMG_EXT = ".png"
MASK_EXT = ".png"

DATA_DIR = Path(__file__).resolve().parent / "inputs" / "dsb2018_96"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"

# -----------------------------------------------------------------------------
# Seed & device
# -----------------------------------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# Run directory & logging
# -----------------------------------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
run_dir = OUTPUT_DIR / run_name
run_dir.mkdir(parents=True, exist_ok=True)

console_log = (run_dir / "console.log").open("w", encoding="utf-8")
_orig_stdout = sys.stdout
_orig_stderr = sys.stderr
sys.stdout = Tee(sys.stdout, console_log)
sys.stderr = Tee(sys.stderr, console_log)

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
print(f"Run dir: {run_dir}")
print(f"Data dir: {DATA_DIR}")


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

class CellDataset(Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, num_classes=1,
                 img_ext=".png", mask_ext=".png", augment=False):
        self.img_ids = sorted(img_ids)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.num_classes = num_classes
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.augment = augment

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        img = Image.open(self.img_dir / (img_id + self.img_ext)).convert("RGB")
        mask = Image.open(self.mask_dir / "0" / (img_id + self.mask_ext)).convert("L")

        if self.augment:
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
            rot = random.choice([0, 90, 180, 270])
            if rot != 0:
                img = img.rotate(rot, expand=True)
                mask = mask.rotate(rot, expand=True)

        img = np.array(img, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32) / 255.0

        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        return img, mask


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, 3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class NestedUNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, deep_supervision=False,
                 nb_filter=None):
        super().__init__()
        if nb_filter is None:
            nb_filter = [32, 64, 128, 256, 512]
        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            return [self.final1(x0_1), self.final2(x0_2),
                    self.final3(x0_3), self.final4(x0_4)]
        return self.final(x0_4)


# -----------------------------------------------------------------------------
# Loss — BCEDiceLoss
# -----------------------------------------------------------------------------

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target)
        smooth = 1e-5
        pred_sig = torch.sigmoid(pred)
        num = target.size(0)
        pred_flat = pred_sig.view(num, -1)
        target_flat = target.view(num, -1)
        intersection = (pred_flat * target_flat).sum(1)
        dice = (2. * intersection + smooth) / (pred_flat.sum(1) + target_flat.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def iou_score(pred, target):
    smooth = 1e-5
    if isinstance(pred, torch.Tensor):
        pred = torch.sigmoid(pred).data.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.data.cpu().numpy()
    pred_bin = pred > 0.5
    target_bin = target > 0.5
    intersection = (pred_bin & target_bin).sum()
    union = (pred_bin | target_bin).sum()
    return (intersection + smooth) / (union + smooth)


def dice_coef(pred, target):
    smooth = 1e-5
    if isinstance(pred, torch.Tensor):
        pred = torch.sigmoid(pred).view(-1).data.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.view(-1).data.cpu().numpy()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


# -----------------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------------

if not DATA_DIR.exists():
    raise SystemExit(f"Data directory not found: {DATA_DIR}")

img_dir = DATA_DIR / "images"
mask_dir = DATA_DIR / "masks"

all_ids = sorted([
    p.stem for p in img_dir.glob(f"*{IMG_EXT}")
])
print(f"Total samples: {len(all_ids)}")

rng = np.random.default_rng(SEED)
shuffled = all_ids.copy()
rng.shuffle(shuffled)
val_count = max(1, int(len(shuffled) * VAL_RATIO))
val_ids = shuffled[:val_count]
train_ids = shuffled[val_count:]
print(f"Train: {len(train_ids)}, Val: {len(val_ids)}")

train_dataset = CellDataset(train_ids, img_dir, mask_dir,
                            num_classes=NUM_CLASSES, img_ext=IMG_EXT,
                            mask_ext=MASK_EXT, augment=True)
val_dataset = CellDataset(val_ids, img_dir, mask_dir,
                          num_classes=NUM_CLASSES, img_ext=IMG_EXT,
                          mask_ext=MASK_EXT, augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

# -----------------------------------------------------------------------------
# Save config
# -----------------------------------------------------------------------------

config = {
    "seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE, "momentum": MOMENTUM,
    "weight_decay": WEIGHT_DECAY, "min_lr": MIN_LR,
    "num_classes": NUM_CLASSES, "input_h": INPUT_H, "input_w": INPUT_W,
    "input_channels": INPUT_CHANNELS, "deep_supervision": DEEP_SUPERVISION,
    "nb_filter": NB_FILTER, "val_ratio": VAL_RATIO,
    "num_workers": NUM_WORKERS, "optimizer": "SGD",
    "scheduler": "CosineAnnealingLR", "loss": "BCEDiceLoss",
    "train_size": len(train_dataset), "val_size": len(val_dataset),
    "device": str(DEVICE), "run_dir": str(run_dir),
}
(run_dir / "config.json").write_text(
    json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

# -----------------------------------------------------------------------------
# Model, optimizer, scheduler
# -----------------------------------------------------------------------------

model = NestedUNet(num_classes=NUM_CLASSES, input_channels=INPUT_CHANNELS,
                   deep_supervision=DEEP_SUPERVISION, nb_filter=NB_FILTER)
model = model.to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model: NestedUNet (deep_supervision={DEEP_SUPERVISION})")
print(f"Total params: {total_params:,}")
print(f"Trainable params: {trainable_params:,}")

with (run_dir / "model_summary.txt").open("w", encoding="utf-8") as f:
    f.write(str(model))
    f.write(f"\n\nTotal params: {total_params:,}\n")
    f.write(f"Trainable params: {trainable_params:,}\n")

criterion = BCEDiceLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE,
                            momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS,
                                                        eta_min=MIN_LR)

# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------

print("\n=== Training Start ===\n")

best_val_iou = 0
best_epoch = 0
train_losses, val_losses, val_ious, val_dices, lrs = [], [], [], [], []

for epoch in range(1, EPOCHS + 1):
    # ---- Train ----
    model.train()
    train_loss = 0.0
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * imgs.size(0)

    train_loss /= len(train_dataset)
    train_losses.append(train_loss)

    # ---- Validate ----
    model.eval()
    val_loss = 0.0
    val_iou_sum = 0.0
    val_dice_sum = 0.0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            val_loss += loss.item() * imgs.size(0)
            val_iou_sum += iou_score(outputs, masks)
            val_dice_sum += dice_coef(outputs, masks)

    val_loss /= len(val_dataset)
    val_iou = val_iou_sum / len(val_dataset)
    val_dice = val_dice_sum / len(val_dataset)
    val_losses.append(val_loss)
    val_ious.append(val_iou)
    val_dices.append(val_dice)
    lrs.append(optimizer.param_groups[0]["lr"])

    scheduler.step()

    print(f"Epoch {epoch:3d}/{EPOCHS} | "
          f"lr {lrs[-1]:.2e} | "
          f"loss {train_loss:.4f} | "
          f"val_loss {val_loss:.4f} | "
          f"val_iou {val_iou:.4f} | "
          f"val_dice {val_dice:.4f}", end="")

    if val_iou > best_val_iou:
        best_val_iou = val_iou
        best_epoch = epoch
        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_iou": best_val_iou, "config": config,
        }, run_dir / "model_best.pth")
        print(" *", end="")
    print()

print("\n=== Training Complete ===\n")

best_val_loss = min(val_losses)
print(f"Best val_iou: {best_val_iou:.4f} at epoch {best_epoch}")
print(f"Best val_loss: {best_val_loss:.4f}")

# Save last model
torch.save({
    "epoch": EPOCHS, "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "best_val_iou": best_val_iou, "config": config,
}, run_dir / "model_last.pth")

# -----------------------------------------------------------------------------
# Epoch metrics CSV
# -----------------------------------------------------------------------------

csv_path = run_dir / "epoch_metrics.csv"
with csv_path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "val_loss", "val_iou", "val_dice", "lr"])
    for i in range(EPOCHS):
        writer.writerow([i + 1, train_losses[i], val_losses[i],
                         val_ious[i], val_dices[i], lrs[i]])

# -----------------------------------------------------------------------------
# Final metrics JSON
# -----------------------------------------------------------------------------

final_metrics = {
    "train_loss": float(train_losses[-1]),
    "val_loss": float(val_loss),
    "val_iou": float(val_iou),
    "val_dice": float(val_dice),
    "best_val_iou": float(best_val_iou),
    "best_val_loss": float(best_val_loss),
    "best_epoch": best_epoch,
    "epochs_ran": EPOCHS,
    "run_dir": str(run_dir),
    "best_model_path": str(run_dir / "model_best.pth"),
    "last_model_path": str(run_dir / "model_last.pth"),
}
(run_dir / "final_metrics.json").write_text(
    json.dumps(final_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

# -----------------------------------------------------------------------------
# Training curves
# -----------------------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

epochs_range = range(1, EPOCHS + 1)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(epochs_range, train_losses, label="train_loss")
axes[0].plot(epochs_range, val_losses, label="val_loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Loss")
axes[0].legend()

axes[1].plot(epochs_range, val_ious, label="val_iou")
axes[1].plot(epochs_range, val_dices, label="val_dice")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Score")
axes[1].set_title("IoU / Dice")
axes[1].legend()

fig.tight_layout()
curve_path = run_dir / "training_curves.png"
fig.savefig(curve_path, dpi=150)
plt.close(fig)
print(f"Saved curves: {curve_path}")

# -----------------------------------------------------------------------------
# Copy latest to results/ root
# -----------------------------------------------------------------------------

import shutil

shutil.copy(run_dir / "model_best.pth", OUTPUT_DIR / "model_best.pth")
shutil.copy(run_dir / "model_last.pth", OUTPUT_DIR / "model_last.pth")
shutil.copy(run_dir / "final_metrics.json", OUTPUT_DIR / "final_metrics.json")
shutil.copy(run_dir / "training_curves.png", OUTPUT_DIR / "training_curves.png")
(OUTPUT_DIR / "latest_run.txt").write_text(str(run_dir) + "\n", encoding="utf-8")

print(f"Saved latest best model: {OUTPUT_DIR / 'model_best.pth'}")
print(f"Saved latest last model: {OUTPUT_DIR / 'model_last.pth'}")
print(f"Saved latest metrics: {OUTPUT_DIR / 'final_metrics.json'}")
print(f"Saved latest curves: {OUTPUT_DIR / 'training_curves.png'}")

# -----------------------------------------------------------------------------
# Teardown
# -----------------------------------------------------------------------------

sys.stdout = _orig_stdout
sys.stderr = _orig_stderr
console_log.close()
print("Done.")
