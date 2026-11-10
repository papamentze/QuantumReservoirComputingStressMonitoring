import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                     "mathtext.fontset": "dejavuserif"})

BLUE = "#2E5A87"; BLUEF = "#DCE7F1"; ORANGE = "#B5651D"; ORANGEF = "#F5E3CE"
GREY = "#555555"; GREYF = "#ECECEC"; QBG = "#EAF1F8"
FS_TITLE, FS_DET, FS_NOTE = 6.6, 5.6, 5.6        # point sizes at final print size

XL, YL = (0, 100), (3.5, 34.0)
WIDTH = 4.8                                       # llncs text width (12.2 cm), printed at 100%
fig, ax = plt.subplots(figsize=(WIDTH, WIDTH * (YL[1] - YL[0]) / (XL[1] - XL[0])))
ax.axis("off"); ax.set_xlim(*XL); ax.set_ylim(*YL); ax.set_aspect("equal")

N = 6; margin = 0.4; gap = 2.0
W = (100 - 2 * margin - (N - 1) * gap) / N
H = 17.5; y = 13.0
xs = [margin + i * (W + gap) for i in range(N)]

# container behind the two untrained quantum stages
cx0 = xs[2] - 0.8; cx1 = xs[3] + W + 0.8
ax.add_patch(FancyBboxPatch((cx0, y - 2.6), cx1 - cx0, H + 5.4,
             boxstyle="round,pad=0.2,rounding_size=1.8", fc=QBG, ec=BLUE, lw=0.8, alpha=0.55, zorder=0))
ax.text((cx0 + cx1) / 2, y + H + 1.6, "untrained quantum reservoir",
        ha="center", va="center", fontsize=FS_NOTE, color=BLUE, style="italic", zorder=3)

nodes = [
    (GREYF, GREY,     "Wearable\nsignals",  "60-s windows\n" + r"$\to$ features $x$"),
    (GREYF, GREY,     "Angle\nencoding",    r"$\theta=\pi\tanh(\gamma g)$" + "\nEq. (1)"),
    (BLUEF, BLUE,     "Quantum\nreservoir", "Eq. (2)\n" + r"ring $ZZ$, noise"),
    (BLUEF, BLUE,     "Readout\nfeatures",  r"$\langle X,Y,Z\rangle,\langle ZZ\rangle$" + "\n" + r"$V{=}3$ sub-steps"),
    (ORANGEF, ORANGE, "Ridge\nreadout",     "linear score\n" + r"$\lambda{=}1$"),
    (GREYF, GREY,     "Conformal\nalarm",   r"$\alpha{=}0.10$" + "\n" + r"$m{=}3$ windows"),
]
for (fc, ec, title, det), x in zip(nodes, xs):
    ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.13,rounding_size=1.3",
                 fc=fc, ec=ec, lw=1.0, zorder=2))
    ax.text(x + W / 2, y + H - 4.3, title, ha="center", va="center", fontsize=FS_TITLE,
            weight="bold", color=ec, linespacing=0.98, zorder=3)
    ax.text(x + W / 2, y + 3.7, det, ha="center", va="center", fontsize=FS_DET,
            color="#222", linespacing=1.05, zorder=3)

gy = y + H / 2                             # glyph band, between title and details
# waveform (box 1)
gx = xs[0] + W / 2; t = np.linspace(0, 1, 60)
wave = 0.9 * np.sin(2 * np.pi * 3 * t) * np.exp(-0.6 * t) + 0.4 * np.sin(2 * np.pi * 7 * t)
ax.plot(gx - 3.2 + 6.4 * t, gy + 1.1 * wave, color=GREY, lw=0.7, zorder=3)
# qubit ring (box 3)
rx = xs[2] + W / 2; R = 1.9
pts = [(rx + R * np.cos(a), gy + R * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 6, endpoint=False)]
for i in range(len(pts)):
    a, b = pts[i], pts[(i + 1) % len(pts)]
    ax.plot([a[0], b[0]], [a[1], b[1]], color=BLUE, lw=0.5, alpha=0.7, zorder=3)
for px, py in pts:
    ax.add_patch(Circle((px, py), 0.5, fc="white", ec=BLUE, lw=0.7, zorder=4))
# alarm glyph (box 6)
bx = xs[5] + W / 2
ax.plot([bx - 1.6, bx + 1.6], [gy - 1.2, gy - 1.2], color=GREY, lw=0.7, zorder=3)
ax.plot([bx - 1.4, bx, bx + 1.4], [gy - 1.2, gy + 1.8, gy - 1.2], color=GREY, lw=0.7, zorder=3)
ax.add_patch(Circle((bx, gy - 1.7), 0.35, fc=GREY, ec=GREY, zorder=4))

# flow arrows
for i in range(N - 1):
    ax.add_patch(FancyArrowPatch((xs[i] + W + 0.12, y + H / 2), (xs[i + 1] - 0.12, y + H / 2),
                 arrowstyle="-|>", mutation_scale=7, lw=1.0, color="#3a3a3a", zorder=5))

# trained / untrained annotations
ax.plot([cx0 + 0.6, cx1 - 0.6], [y - 4.4, y - 4.4], color=BLUE, lw=0.8, zorder=3)
ax.text((cx0 + cx1) / 2, y - 6.6, r"fixed $\cdot$ no gradients", ha="center", va="center",
        fontsize=FS_NOTE, color=BLUE, style="italic")
ax.plot([xs[4] + 1, xs[4] + W - 1], [y - 4.4, y - 4.4], color=ORANGE, lw=0.8, zorder=3)
ax.text(xs[4] + W / 2, y - 6.6, "only trained\ncomponent", ha="center", va="center",
        fontsize=FS_NOTE, color=ORANGE, style="italic", linespacing=0.95)

plt.subplots_adjust(left=0.002, right=0.998, top=0.998, bottom=0.002)
os.makedirs("Output/figures", exist_ok=True); os.makedirs("figures", exist_ok=True)
for d in ("Output/figures", "figures"):
    fig.savefig(f"{d}/fig_pipeline.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{d}/fig_pipeline.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{d}/fig_pipeline.tiff", dpi=600, bbox_inches="tight", pad_inches=0.02,
                pil_kwargs={"compression": "tiff_lzw"})

# verify every label line fits its box
fig.canvas.draw(); r = fig.canvas.get_renderer()
box_px = W / 100.0 * ax.get_window_extent().width; bad = 0
for (fc, ec, title, det), x in zip(nodes, xs):
    for block, fs in [(title, FS_TITLE), (det, FS_DET)]:
        for line in block.split("\n"):
            tt = ax.text(0, 0, line, fontsize=fs); fig.canvas.draw()
            if tt.get_window_extent(renderer=r).width > box_px - 6:
                bad += 1; print("OVER:", line)
            tt.remove()
print("box_px=%.0f overflows=%d" % (box_px, bad))
print("size (px, 600 dpi):", Image.open("figures/fig_pipeline.png").size) 