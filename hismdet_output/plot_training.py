"""
HiSMD-Net Training Curve Visualizer
Generates publication-quality training analysis charts from training_log.csv
"""

import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# ── Load CSV ────────────────────────────────────────────────────────────────
CSV_PATH = os.path.join(os.path.dirname(__file__), "training_log.csv")
OUT_DIR  = os.path.dirname(__file__)

epochs, lr, loss, cls_loss, box_loss = [], [], [], [], []
ap_ep, ap_val, ap50_val, aps_val, apm_val = [], [], [], [], []

with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        e = int(row["epoch"])
        epochs.append(e)
        lr.append(float(row["lr"]))
        loss.append(float(row["loss"]))
        cls_loss.append(float(row["cls_loss"]))
        box_loss.append(float(row["box_loss"]))
        if row["AP"].strip():
            ap_ep.append(e)
            ap_val.append(float(row["AP"]))
            ap50_val.append(float(row["AP50"]))
            aps_val.append(float(row["APS"]))
            apm_val.append(float(row["APM"]))

epochs   = np.array(epochs)
lr       = np.array(lr)
loss     = np.array(loss)
cls_loss = np.array(cls_loss)
box_loss = np.array(box_loss)
ap_ep    = np.array(ap_ep)
ap_val   = np.array(ap_val)
ap50_val = np.array(ap50_val)
aps_val  = np.array(aps_val)
apm_val  = np.array(apm_val)

# ── Style ────────────────────────────────────────────────────────────────────
DARK_BG  = "#0d1117"
CARD_BG  = "#161b22"
GRID_COL = "#21262d"
TEXT_COL = "#e6edf3"
ACCENT   = ["#58a6ff", "#3fb950", "#f78166", "#d2a8ff", "#ffa657"]

plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    CARD_BG,
    "axes.edgecolor":    GRID_COL,
    "axes.labelcolor":   TEXT_COL,
    "xtick.color":       TEXT_COL,
    "ytick.color":       TEXT_COL,
    "text.color":        TEXT_COL,
    "grid.color":        GRID_COL,
    "legend.facecolor":  CARD_BG,
    "legend.edgecolor":  GRID_COL,
    "font.family":       "DejaVu Sans",
})

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — 2×2 Training Dashboard
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=150)
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("HiSMD-Net — 100-Epoch Training Dashboard  |  VisDrone2019-DET",
             fontsize=15, fontweight="bold", color=TEXT_COL, y=0.98)

# ── Panel A: Total + Component Losses ───────────────────────────────────────
ax = axes[0, 0]
ax.plot(epochs, loss,     color=ACCENT[0], lw=2,   label="Total Loss",      zorder=3)
ax.plot(epochs, cls_loss, color=ACCENT[2], lw=1.5, label="Cls Loss",   ls="--", zorder=3)
ax.plot(epochs, box_loss, color=ACCENT[1], lw=1.5, label="Box Loss",   ls=":",  zorder=3)
ax.set_xlabel("Epoch");  ax.set_ylabel("Loss Value")
ax.set_title("Training Loss Curves", fontweight="bold", color=TEXT_COL)
ax.legend(fontsize=9); ax.grid(True, alpha=0.35); ax.set_xlim(1, 100)
# Annotations
ax.annotate(f"Start: {loss[0]:.2f}", xy=(1, loss[0]),
            xytext=(10, loss[0]+0.1), fontsize=8, color=ACCENT[0],
            arrowprops=dict(arrowstyle="->", color=ACCENT[0], lw=0.8))
ax.annotate(f"End: {loss[-1]:.4f}", xy=(100, loss[-1]),
            xytext=(75, loss[-1]+0.2), fontsize=8, color=ACCENT[0],
            arrowprops=dict(arrowstyle="->", color=ACCENT[0], lw=0.8))

# ── Panel B: AP Metrics ──────────────────────────────────────────────────────
ax = axes[0, 1]
ax.plot(ap_ep, ap50_val, color=ACCENT[0], lw=2.5, marker="o", ms=5, label="AP50")
ax.plot(ap_ep, ap_val,   color=ACCENT[1], lw=2.5, marker="s", ms=5, label="mAP (0.50:0.95)")
ax.plot(ap_ep, aps_val,  color=ACCENT[2], lw=2.0, marker="^", ms=5, label="APS (Small)")
ax.plot(ap_ep, apm_val,  color=ACCENT[3], lw=2.0, marker="D", ms=5, label="APM (Medium)")
# UAVDet baselines (dashed horizontal)
ax.axhline(64.20, color=ACCENT[0], lw=1.0, ls="--", alpha=0.5, label="UAVDet AP50 (64.20%)")
ax.axhline(43.10, color=ACCENT[1], lw=1.0, ls="--", alpha=0.5, label="UAVDet mAP (43.10%)")
ax.axhline(24.50, color=ACCENT[2], lw=1.0, ls="--", alpha=0.5, label="UAVDet APS (24.50%)")
ax.set_xlabel("Epoch"); ax.set_ylabel("AP (%)")
ax.set_title("Detection Accuracy Metrics vs Epoch", fontweight="bold", color=TEXT_COL)
ax.legend(fontsize=7.5, ncol=2); ax.grid(True, alpha=0.35); ax.set_xlim(1, 100)
# Final value annotations
for val, col, name in zip(
        [ap50_val[-1], ap_val[-1], aps_val[-1], apm_val[-1]],
        ACCENT[:4],
        ["68.43%", "47.80%", "30.23%", "69.72%"]):
    ax.annotate(name, xy=(100, val), xytext=(101, val), fontsize=7.5,
                color=col, va="center")

# ── Panel C: Learning Rate Schedule ─────────────────────────────────────────
ax = axes[1, 0]
ax.plot(epochs, lr * 1e4, color=ACCENT[4], lw=2)
ax.fill_between(epochs, 0, lr * 1e4, color=ACCENT[4], alpha=0.15)
ax.set_xlabel("Epoch"); ax.set_ylabel("Learning Rate (×10⁻⁴)")
ax.set_title("Cosine Annealing LR Schedule (with 3-Epoch Warmup)",
             fontweight="bold", color=TEXT_COL)
ax.grid(True, alpha=0.35); ax.set_xlim(1, 100)
ax.axvspan(1, 3, color=ACCENT[0], alpha=0.12, label="Warmup (3 epochs)")
ax.legend(fontsize=9)
ax.annotate(f"Peak: {lr[0]*1e4:.4f}×10⁻⁴", xy=(1, lr[0]*1e4),
            xytext=(5, lr[0]*1e4*0.97), fontsize=8, color=ACCENT[4])
ax.annotate(f"Min: {lr[-1]*1e4:.4f}×10⁻⁴", xy=(100, lr[-1]*1e4),
            xytext=(70, lr[-1]*1e4+0.005), fontsize=8, color=ACCENT[4])

# ── Panel D: APS Improvement Bar Chart ──────────────────────────────────────
ax = axes[1, 1]
models   = ["Faster\nR-CNN", "RT-DETR\nR50", "YOLOv8-m", "YOLOv9-c",
            "UAVDet\n(Base)", "HiSMD-Net\n(Ours)"]
aps_bars = [9.10, 14.20, 17.40, 18.10, 24.50, 30.23]
colors   = [GRID_COL]*5 + [ACCENT[1]]
bars     = ax.bar(models, aps_bars, color=colors, width=0.6,
                  edgecolor=GRID_COL, linewidth=0.5, zorder=3)
ax.set_ylabel("APS — Small Object AP (%)")
ax.set_title("APS Comparison: HiSMD-Net vs SOTA  (VisDrone Val)", fontweight="bold", color=TEXT_COL)
ax.grid(True, axis="y", alpha=0.35); ax.set_ylim(0, 38)
ax.axhline(24.50, color=ACCENT[2], lw=1.2, ls="--", alpha=0.7,
           label="UAVDet APS = 24.50%")
ax.legend(fontsize=8)
for bar, val in zip(bars, aps_bars):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.4,
            f"{val:.2f}%", ha="center", va="bottom",
            fontsize=8, color=TEXT_COL, fontweight="bold")
# Improvement arrow
ax.annotate("", xy=(5, 30.23), xytext=(4, 24.50),
            arrowprops=dict(arrowstyle="->", color=ACCENT[1], lw=2))
ax.text(4.55, 27.5, "+5.73pp", color=ACCENT[1], fontsize=9, fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.97])
out1 = os.path.join(OUT_DIR, "training_dashboard.png")
plt.savefig(out1, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print(f"Saved: {out1}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Training Convergence & Loss Breakdown (Wide)
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("HiSMD-Net — Loss Breakdown & Convergence Analysis",
             fontsize=13, fontweight="bold", color=TEXT_COL, y=1.01)

# Loss reduction rate
ax = axes[0]
loss_drop = np.diff(loss)
ax.bar(epochs[1:], -loss_drop, color=[ACCENT[1] if d < 0 else ACCENT[2] for d in loss_drop],
       width=0.8, alpha=0.8)
ax.axhline(0, color=TEXT_COL, lw=0.8, alpha=0.5)
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss Reduction per Epoch")
ax.set_title("Per-Epoch Loss Reduction", fontweight="bold", color=TEXT_COL)
ax.grid(True, alpha=0.25, axis="y"); ax.set_xlim(1, 100)
ax.text(0.05, 0.92, "Green = Improving\nRed = Regression",
        transform=ax.transAxes, fontsize=8, color=TEXT_COL, va="top")

# Cls vs Box loss breakdown
ax = axes[1]
ax.stackplot(epochs, cls_loss, box_loss,
             labels=["Classification Loss", "Box Regression Loss"],
             colors=[ACCENT[2]+"55", ACCENT[1]+"55"],
             edgecolor=None)
ax.plot(epochs, cls_loss, color=ACCENT[2], lw=1.5, alpha=0.9)
ax.plot(epochs, box_loss, color=ACCENT[1], lw=1.5, alpha=0.9)
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss Value")
ax.set_title("Classification vs Box Loss Breakdown", fontweight="bold", color=TEXT_COL)
ax.legend(fontsize=9, loc="upper right"); ax.grid(True, alpha=0.25); ax.set_xlim(1, 100)

# AP convergence with UAVDet threshold shading
ax = axes[2]
ax.fill_between(ap_ep, 0, ap50_val, color=ACCENT[0], alpha=0.15)
ax.fill_between(ap_ep, 0, aps_val,  color=ACCENT[2], alpha=0.20)
ax.plot(ap_ep, ap50_val, color=ACCENT[0], lw=2.5, label="AP50")
ax.plot(ap_ep, aps_val,  color=ACCENT[2], lw=2.0, label="APS (Small)")
ax.axhline(64.20, color=ACCENT[0], lw=1.2, ls="--", alpha=0.6)
ax.axhline(24.50, color=ACCENT[2], lw=1.2, ls="--", alpha=0.6)
ax.text(2, 65.0, "UAVDet AP50 = 64.20%", fontsize=7.5, color=ACCENT[0], alpha=0.8)
ax.text(2, 25.3, "UAVDet APS = 24.50%",  fontsize=7.5, color=ACCENT[2], alpha=0.8)

# Mark epoch where we surpass UAVDet
for i, (e, v) in enumerate(zip(ap_ep, ap50_val)):
    if v >= 64.20:
        ax.axvline(e, color=ACCENT[0], lw=1.0, ls=":", alpha=0.7)
        ax.text(e+1, 30, f"Surpasses\nUAVDet\n@ Ep {e}", fontsize=7,
                color=ACCENT[0], alpha=0.9)
        break
for i, (e, v) in enumerate(zip(ap_ep, aps_val)):
    if v >= 24.50:
        ax.axvline(e, color=ACCENT[2], lw=1.0, ls=":", alpha=0.7)
        ax.text(e+1, 14, f"APS > UAVDet\n@ Ep {e}", fontsize=7,
                color=ACCENT[2], alpha=0.9)
        break

ax.set_xlabel("Epoch"); ax.set_ylabel("AP (%)")
ax.set_title("Convergence vs UAVDet Baselines", fontweight="bold", color=TEXT_COL)
ax.legend(fontsize=9); ax.grid(True, alpha=0.25); ax.set_xlim(1, 100)

plt.tight_layout()
out2 = os.path.join(OUT_DIR, "training_convergence.png")
plt.savefig(out2, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print(f"Saved: {out2}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Publication-Ready Comparison Chart
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("HiSMD-Net vs State-of-the-Art  |  VisDrone2019-DET Validation Set",
             fontsize=13, fontweight="bold", color=TEXT_COL)

models_full = ["Faster R-CNN\n(41.5M)", "RT-DETR-R50\n(42.0M)",
               "YOLOv8-m\n(25.9M)", "YOLOv9-c\n(25.3M)",
               "UAVDet\n(19.8M)", "HiSMD-Net\n(16.5M) ★"]
ap50_bars = [47.20, 59.10, 60.80, 62.40, 64.20, 68.43]
map_bars  = [28.70, 38.90, 39.20, 41.60, 43.10, 47.80]
bar_cols  = [GRID_COL, GRID_COL, GRID_COL, GRID_COL, "#30363d", ACCENT[1]]

x = np.arange(len(models_full))
w = 0.42

# AP50
ax = axes[0]
bars1 = ax.bar(x - w/2, ap50_bars, w, color=bar_cols, edgecolor=GRID_COL, lw=0.5, label="AP50", zorder=3)
ax.set_xticks(x); ax.set_xticklabels(models_full, fontsize=8)
ax.set_ylabel("AP50 (%)"); ax.set_title("AP50 Comparison", fontweight="bold", color=TEXT_COL)
ax.grid(True, axis="y", alpha=0.3, zorder=0); ax.set_ylim(0, 80)
for bar, val in zip(bars1, ap50_bars):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.5,
            f"{val:.2f}", ha="center", va="bottom", fontsize=8.5, color=TEXT_COL, fontweight="bold")

# mAP
bars2 = ax.bar(x + w/2, map_bars, w, color=bar_cols, edgecolor=GRID_COL, lw=0.5,
               alpha=0.65, label="mAP (0.50:0.95)", zorder=3)
for bar, val in zip(bars2, map_bars):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.5,
            f"{val:.2f}", ha="center", va="bottom", fontsize=8.5, color=TEXT_COL)
ax.legend(fontsize=9)

# APS vs Params bubble chart
ax = axes[1]
params = [41.53, 42.00, 25.90, 25.30, 19.80, 16.49]
aps_cmp = [9.10, 14.20, 17.40, 18.10, 24.50, 30.23]
bubble_cols = [GRID_COL]*5 + [ACCENT[1]]
bubble_sizes = [p * 30 for p in params]
sc = ax.scatter(params, aps_cmp, s=bubble_sizes, c=bubble_cols,
                edgecolors=[TEXT_COL]*5 + [ACCENT[1]], linewidths=1.5, zorder=4, alpha=0.85)
for i, (p, a, m) in enumerate(zip(params, aps_cmp, models_full)):
    label = m.split("\n")[0]
    offset = (0.4, 0.5) if i < 5 else (-3.5, 1.2)
    ax.annotate(label, (p, a), xytext=(p + offset[0], a + offset[1]),
                fontsize=7.5, color=TEXT_COL, fontweight="bold" if i == 5 else "normal")
ax.set_xlabel("Parameters (M)  [bubble size ∝ params]")
ax.set_ylabel("APS — Small Object AP (%)")
ax.set_title("Accuracy vs Efficiency (APS vs Params)", fontweight="bold", color=TEXT_COL)
ax.grid(True, alpha=0.3); ax.invert_xaxis()
# Best region annotation
ax.text(0.03, 0.92, "← Better (fewer params,\n    higher APS)",
        transform=ax.transAxes, fontsize=8, color=ACCENT[1], va="top")

plt.tight_layout()
out3 = os.path.join(OUT_DIR, "sota_comparison.png")
plt.savefig(out3, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print(f"Saved: {out3}")

print("\n✅ All 3 figures generated successfully.")
print(f"   1. training_dashboard.png")
print(f"   2. training_convergence.png")
print(f"   3. sota_comparison.png")
