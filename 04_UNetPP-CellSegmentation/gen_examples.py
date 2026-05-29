#!/usr/bin/env python3
"""Generate segmentation examples using the best model for the report."""
import sys, random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Model definition (same as train.py) ----
import torch.nn as nn

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
    def __init__(self, num_classes=1, input_channels=3, deep_supervision=False, nb_filter=None):
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
        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])
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
            return [self.final1(x0_1), self.final2(x0_2), self.final3(x0_3), self.final4(x0_4)]
        return self.final(x0_4)

# ---- Load best model ----
BASE = Path("/home/zhao/ai-intro-course/04_UNetPP-CellSegmentation")
ckpt = torch.load(BASE / "results" / "model_best.pth", map_location=DEVICE, weights_only=False)
model = NestedUNet(num_classes=1, input_channels=3, deep_supervision=False)
model.load_state_dict(ckpt["model_state_dict"])
model = model.to(DEVICE)
model.eval()
print(f"Loaded model from epoch {ckpt['epoch']}, best val_iou: {ckpt['best_val_iou']:.4f}")

# ---- Load val split (same seed as training) ----
img_dir = BASE / "inputs" / "dsb2018_96" / "images"
mask_dir = BASE / "inputs" / "dsb2018_96" / "masks"
all_ids = sorted([p.stem for p in img_dir.glob("*.png")])
rng = np.random.default_rng(42)
shuffled = all_ids.copy()
rng.shuffle(shuffled)
val_count = max(1, int(len(shuffled) * 0.2))
val_ids = shuffled[:val_count]

# Pick 6 random val samples
sample_ids = random.sample(val_ids, min(6, len(val_ids)))

# ---- Generate predictions ----
rows = []
for img_id in sample_ids:
    img = Image.open(img_dir / (img_id + ".png")).convert("RGB")
    mask = Image.open(mask_dir / "0" / (img_id + ".png")).convert("L")
    img_arr = np.array(img, dtype=np.float32) / 255.0
    mask_arr = np.array(mask, dtype=np.float32) / 255.0

    with torch.no_grad():
        inp = torch.from_numpy(img_arr.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)
        pred = torch.sigmoid(model(inp)).squeeze().cpu().numpy()

    rows.append((img_arr, mask_arr, pred))

# ---- Plot: 6 rows x 3 columns ----
fig, axes = plt.subplots(6, 3, figsize=(9, 18), dpi=150)
titles = ["Original", "Ground Truth", "Prediction"]

for r, (img, mask, pred) in enumerate(rows):
    axes[r][0].imshow(img)
    axes[r][1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[r][2].imshow(pred > 0.5, cmap="gray", vmin=0, vmax=1)
    for c in range(3):
        axes[r][c].set_xticks([])
        axes[r][c].set_yticks([])
        if r == 0:
            axes[r][c].set_title(titles[c], fontsize=10)

fig.tight_layout()
out_path = BASE / "results" / "segmentation_examples.png"
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"Saved {out_path}")
