"""
Generates model_plot.png — a hand-drawn CNN architecture diagram.
Run from the repo root: python 06/draw_architecture.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# colour palette 
C_INPUT  = "#CFE2FF"
C_AUG    = "#FFF3CD"
C_CONV   = "#4A90D9"
C_BN     = "#85C1E9"
C_POOL   = "#E67E22"
C_DROP   = "#E74C3C"
C_FLAT   = "#8E44AD"
C_DENSE  = "#27AE60"
C_OUT    = "#1A5276"

LIGHT_BOXES = {C_INPUT, C_AUG, C_BN}   # use dark text on these

C_GROUP_BG = {
    "Input":      "#EBF5FB",
    "Augment":    "#FEFDE7",
    "Block 1":    "#EAF4FB",
    "Block 2":    "#E8F8F5",
    "Block 3":    "#F4ECF7",
    "Classifier": "#EAFAF1",
}

# layer table: (name, annotation, box_color, group)
LAYERS = [
    ("Input",             "32 × 32 × 3",
     C_INPUT, "Input"),
    ("RandomFlip",        "horizontal  (p = 0.5)  — train only",
     C_AUG,   "Augment"),
    ("RandomTranslation", "height & width  ± 10 %  — train only",
     C_AUG,   "Augment"),
    ("Conv2D  ×2",        "32 filters · 3×3 · same padding · ReLU  →  32×32×32",
     C_CONV,  "Block 1"),
    ("BatchNorm",         "normalise activations  →  32×32×32",
     C_BN,    "Block 1"),
    ("MaxPooling2D",      "2×2 stride  →  16×16×32",
     C_POOL,  "Block 1"),
    ("Dropout",           "p = 0.20",
     C_DROP,  "Block 1"),
    ("Conv2D  ×2",        "64 filters · 3×3 · same padding · ReLU  →  16×16×64",
     C_CONV,  "Block 2"),
    ("BatchNorm",         "normalise activations  →  16×16×64",
     C_BN,    "Block 2"),
    ("MaxPooling2D",      "2×2 stride  →  8×8×64",
     C_POOL,  "Block 2"),
    ("Dropout",           "p = 0.30",
     C_DROP,  "Block 2"),
    ("Conv2D  ×2",        "128 filters · 3×3 · same padding · ReLU  →  8×8×128",
     C_CONV,  "Block 3"),
    ("BatchNorm",         "normalise activations  →  8×8×128",
     C_BN,    "Block 3"),
    ("MaxPooling2D",      "2×2 stride  →  4×4×128",
     C_POOL,  "Block 3"),
    ("Dropout",           "p = 0.40",
     C_DROP,  "Block 3"),
    ("Flatten",           "4×4×128  =  2 048 units",
     C_FLAT,  "Classifier"),
    ("Dense",             "512 units · ReLU  (he_uniform init)",
     C_DENSE, "Classifier"),
    ("BatchNorm",         "normalise activations  →  512",
     C_BN,    "Classifier"),
    ("Dropout",           "p = 0.50",
     C_DROP,  "Classifier"),
    ("Dense  (output)",   "10 units · Softmax  →  class probabilities",
     C_OUT,   "Classifier"),
]

# layout constants 
FIG_W   = 11.5
ROW_H   = 0.66
GAP     = 0.20
LEFT    = 0.78
BOX_W   = 10.30
LABEL_W = 2.10
GRP_X   = 0.05
GRP_W   = 0.52

n_rows  = len(LAYERS)
FIG_H   = n_rows * (ROW_H + GAP) + 1.80


def yc(i):
    return FIG_H - 1.10 - i * (ROW_H + GAP)


fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")
fig.patch.set_facecolor("#F7F9FC")
ax.set_facecolor("#F7F9FC")

# title 
ax.text(FIG_W / 2, FIG_H - 0.40,
        "CIFAR-10 CNN  —  Model Architecture",
        ha="center", va="center", fontsize=15, fontweight="bold", color="#1A252F")
ax.text(FIG_W / 2, FIG_H - 0.78,
        "3 convolutional blocks  ·  in-model data augmentation  ·  "
        "batch normalisation  ·  he_uniform init",
        ha="center", va="center", fontsize=8.5, color="#566573")

# group background bands & sidebar labels 
group_rows: dict[str, list[int]] = {}
for i, (*_, grp) in enumerate(LAYERS):
    group_rows.setdefault(grp, []).append(i)

for grp, rows in group_rows.items():
    top    = yc(rows[0])  + ROW_H / 2 + GAP / 2
    bottom = yc(rows[-1]) - ROW_H / 2 - GAP / 2
    height = top - bottom
    ax.add_patch(FancyBboxPatch(
        (GRP_X, bottom), FIG_W - GRP_X - 0.05, height,
        boxstyle="round,pad=0.05", linewidth=0,
        facecolor=C_GROUP_BG[grp], zorder=0,
    ))
    ax.add_patch(FancyBboxPatch(
        (GRP_X, bottom), GRP_W, height,
        boxstyle="round,pad=0.03", linewidth=0,
        facecolor="#BFC9CA", zorder=1,
    ))
    ax.text(GRP_X + GRP_W / 2, bottom + height / 2, grp,
            ha="center", va="center", fontsize=7.5, fontweight="bold",
            color="#1A252F", rotation=90, zorder=2)

# layer rows 
for i, (name, annot, color, _) in enumerate(LAYERS):
    cy = yc(i)
    yt = cy - ROW_H / 2

    text_color = "#1A252F" if color in LIGHT_BOXES else "white"

    # name pill
    ax.add_patch(FancyBboxPatch(
        (LEFT, yt), LABEL_W, ROW_H,
        boxstyle="round,pad=0.06", linewidth=0.6,
        edgecolor="#BDC3C7", facecolor=color, zorder=3,
    ))
    ax.text(LEFT + LABEL_W / 2, cy, name,
            ha="center", va="center", fontsize=9, fontweight="bold",
            color=text_color, zorder=4)

    # annotation cell
    ann_left = LEFT + LABEL_W
    ann_w    = BOX_W - LABEL_W
    ax.add_patch(FancyBboxPatch(
        (ann_left, yt), ann_w, ROW_H,
        boxstyle="round,pad=0.06", linewidth=0.6,
        edgecolor="#BDC3C7", facecolor="#FDFEFE", zorder=3,
    ))
    ax.text(ann_left + 0.20, cy, annot,
            ha="left", va="center", fontsize=8.5, color="#2C3E50", zorder=4)

# arrows 
arrow_x = LEFT + BOX_W / 2
for i in range(n_rows - 1):
    y_tail = yc(i)     - ROW_H / 2 - 0.02
    y_head = yc(i + 1) + ROW_H / 2 + 0.02
    ax.annotate("",
                xy=(arrow_x, y_head), xytext=(arrow_x, y_tail),
                arrowprops=dict(arrowstyle="-|>", color="#839192",
                                lw=1.3, mutation_scale=12),
                zorder=5)

# legend 
legend_defs = [
    (C_CONV,  "Convolutional"),
    (C_BN,    "Batch Norm"),
    (C_POOL,  "Max Pooling"),
    (C_DROP,  "Dropout"),
    (C_FLAT,  "Flatten"),
    (C_DENSE, "Dense"),
    (C_OUT,   "Output Dense"),
    (C_AUG,   "Augmentation"),
]
lx, ly = LEFT, 0.18
for k, (c, label) in enumerate(legend_defs):
    px = lx + k * 1.27
    ax.add_patch(mpatches.Rectangle((px, ly), 0.22, 0.26, facecolor=c, zorder=5,
                                    linewidth=0.5, edgecolor="#999"))
    ax.text(px + 0.28, ly + 0.13, label,
            va="center", fontsize=7.2, color="#2C3E50", zorder=6)

plt.tight_layout(pad=0.3)
plt.savefig("model_plot.png", dpi=160, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("model_plot.png saved.")
