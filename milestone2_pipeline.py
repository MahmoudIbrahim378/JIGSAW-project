# milestone2_pipeline.py
# Classical CV (no AI/ML): neighbor matching + puzzle assembly using Milestone-1 outputs
# - Fix A: trim by mask then trim remaining black separators/borders
# - Fix B: ordered RGB edge profiles (no mean-subtraction that can zero-out)
# - Assembly: Beam search + Simulated annealing
# - Visualization: adjacency lines + scores (rubric requirement)

import json
import random
from math import exp
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
META_PATH = Path("puzzle_output/pieces_metadata.json")
assert META_PATH.exists(), f"Missing: {META_PATH}"

# Set grid size to match your puzzle
GRID_R = 2
GRID_C = 2
assert GRID_R * GRID_C > 1

# Profile/solver settings
PROFILE_K = 8      # thickness of edge strip
TOPK = 12          # keep top-K candidates per piece per direction (speed/pruning)
CAND_LIMIT = 25    # candidates per position during beam search
BEAM_WIDTH = 250   # increase for better quality, slower
ANNEAL_ITERS = 15000
ANNEAL_T0 = 10.0
ANNEAL_ALPHA = 0.9994

OUT_DIR = Path("milestone2_output")
OUT_DIR.mkdir(exist_ok=True)

# =========================
# LOAD METADATA
# =========================
with open(META_PATH, "r") as f:
    pieces = json.load(f)

print("Pieces loaded:", len(pieces))
assert GRID_R * GRID_C == len(pieces), f"GRID_R*GRID_C must equal {len(pieces)} pieces"

# =========================
# FIX A: Load crop+mask and trim
# =========================
def load_crop_mask(piece):
    crop = cv2.imread(piece["crop_path"])
    mask = cv2.imread(piece["mask_path"], cv2.IMREAD_GRAYSCALE)
    if crop is None or mask is None:
        raise FileNotFoundError(piece["crop_path"], piece["mask_path"])
    return crop, mask

def trim_by_mask(crop_bgr, mask, pad=1):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return crop_bgr
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, crop_bgr.shape[1] - 1)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, crop_bgr.shape[0] - 1)
    return crop_bgr[y0:y1 + 1, x0:x1 + 1]

def trim_black_border(tile_rgb, thresh=15, pad=2):
    """
    Removes black separators/padding by cropping to the bbox of pixels brighter than 'thresh' in grayscale.
    Works well for puzzles with black gaps/borders.
    """
    gray = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2GRAY)
    m = gray > thresh
    if m.sum() < 50:
        return tile_rgb
    ys, xs = np.where(m)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    h, w = tile_rgb.shape[:2]
    y0 = max(y0 - pad, 0); y1 = min(y1 + pad, h - 1)
    x0 = max(x0 - pad, 0); x1 = min(x1 + pad, w - 1)
    return tile_rgb[y0:y1 + 1, x0:x1 + 1]

tiles = {}  # id -> RGB tile image
for p in pieces:
    crop_bgr, mask = load_crop_mask(p)
    crop_bgr_t = trim_by_mask(crop_bgr, mask, pad=1)
    rgb = cv2.cvtColor(crop_bgr_t, cv2.COLOR_BGR2RGB)
    rgb = trim_black_border(rgb, thresh=15, pad=2)
    tiles[p["id"]] = rgb

ids = list(tiles.keys())
print("Tiles prepared:", len(ids))
print("Example tile shape:", tiles[ids[0]].shape)

# =========================
# FIX B: Ordered RGB edge profiles + distance
# =========================
SIDES = ["top", "right", "bottom", "left"]

def edge_profile_rgb(tile_rgb, side, k=PROFILE_K):
    img = tile_rgb.astype(np.float32)
    h, w = img.shape[:2]
    if side == "top":
        prof = img[0:k, :, :].mean(axis=0)          # (w,3)
    elif side == "bottom":
        prof = img[h-k:h, :, :].mean(axis=0)
    elif side == "left":
        prof = img[:, 0:k, :].mean(axis=1)          # (h,3)
    elif side == "right":
        prof = img[:, w-k:w, :].mean(axis=1)
    else:
        raise ValueError("bad side")
    return prof

def profile_distance(pA, pB):
    if pA is None or pB is None:
        return np.inf
    nA, nB = len(pA), len(pB)
    if nA < 30 or nB < 30:
        return np.inf

    L = min(nA, nB)
    A = cv2.resize(pA.reshape(nA, 1, 3), (1, L), interpolation=cv2.INTER_AREA).reshape(L, 3)
    B = cv2.resize(pB.reshape(nB, 1, 3), (1, L), interpolation=cv2.INTER_AREA).reshape(L, 3)
    return float(np.mean(np.abs(A - B)))

profiles = {pid: {s: edge_profile_rgb(tiles[pid], s) for s in SIDES} for pid in ids}
print("Profiles computed.")

# =========================
# Precompute adjacency costs
# =========================
right_cost = {A: {} for A in ids}  # A.right -> B.left
down_cost  = {A: {} for A in ids}  # A.bottom -> B.top

for A in ids:
    for B in ids:
        if A == B:
            right_cost[A][B] = np.inf
            down_cost[A][B] = np.inf
        else:
            right_cost[A][B] = profile_distance(profiles[A]["right"], profiles[B]["left"])
            down_cost[A][B]  = profile_distance(profiles[A]["bottom"], profiles[B]["top"])

bestR = {A: sorted(ids, key=lambda B: right_cost[A][B])[:TOPK] for A in ids}
bestD = {A: sorted(ids, key=lambda B: down_cost[A][B])[:TOPK]  for A in ids}

print("Sanity costs (random right edges):")
for _ in range(min(5, len(ids))):
    X, Y = random.sample(ids, 2)
    print(" ", right_cost[X][Y])

# =========================
# Scoring + Beam search
# =========================
def total_cost_grid(grid):
    cost = 0.0
    for r in range(GRID_R):
        for c in range(GRID_C):
            idx = r * GRID_C + c
            A = grid[idx]
            if c + 1 < GRID_C:
                B = grid[idx + 1]
                cost += right_cost[A][B]
            if r + 1 < GRID_R:
                B = grid[idx + GRID_C]
                cost += down_cost[A][B]
    return float(cost)

def beam_search():
    states = [(0.0, [None] * (GRID_R * GRID_C), set())]

    for pos in range(GRID_R * GRID_C):
        r, c = divmod(pos, GRID_C)
        new_states = []

        for cost_so_far, grid, used in states:
            left = grid[pos - 1] if c > 0 else None
            up   = grid[pos - GRID_C] if r > 0 else None

            cand = set(ids) - used
            if left is not None:
                cand = cand.intersection(bestR[left])
            if up is not None:
                cand = cand.intersection(bestD[up]) if cand else set(bestD[up])

            if not cand:
                cand = set(ids) - used

            scored = []
            for x in cand:
                inc = 0.0
                if left is not None:
                    inc += right_cost[left][x]
                if up is not None:
                    inc += down_cost[up][x]
                scored.append((inc, x))

            scored.sort(key=lambda t: t[0])
            scored = scored[:CAND_LIMIT]

            for inc, x in scored:
                g2 = grid.copy()
                g2[pos] = x
                u2 = set(used)
                u2.add(x)
                new_states.append((cost_so_far + inc, g2, u2))

        new_states.sort(key=lambda s: s[0])
        states = new_states[:BEAM_WIDTH]
        print(f"pos {pos+1}/{GRID_R*GRID_C} kept={len(states)} best_partial={states[0][0]:.2f}")

    best = min(states, key=lambda s: total_cost_grid(s[1]))
    return best[1], total_cost_grid(best[1])

grid0, cost0 = beam_search()
print("\\nBeam result cost:", cost0)
for r in range(GRID_R):
    print(grid0[r * GRID_C:(r + 1) * GRID_C])

# =========================
# Simulated annealing refinement
# =========================
def anneal(grid, iters=15000, T0=10.0, alpha=0.9994):
    g = grid.copy()
    bestg = g.copy()
    bestc = total_cost_grid(g)
    curc = bestc
    T = T0

    for _ in range(iters):
        i, j = random.sample(range(len(g)), 2)
        g2 = g.copy()
        g2[i], g2[j] = g2[j], g2[i]

        c2 = total_cost_grid(g2)
        dc = c2 - curc

        if dc < 0 or random.random() < exp(-dc / max(T, 1e-9)):
            g = g2
            curc = c2
            if curc < bestc:
                bestc = curc
                bestg = g.copy()

        T *= alpha

    return bestg, bestc

grid, cost = anneal(grid0, iters=ANNEAL_ITERS, T0=ANNEAL_T0, alpha=ANNEAL_ALPHA)
print("\\nAfter annealing cost:", cost)
for r in range(GRID_R):
    print(grid[r * GRID_C:(r + 1) * GRID_C])

# =========================
# Render assembled image
# =========================
def pad_to_same(img, H, W):
    h, w = img.shape[:2]
    out = np.zeros((H, W, 3), dtype=img.dtype)
    out[:h, :w] = img
    return out

H = max(tiles[x].shape[0] for x in grid)
W = max(tiles[x].shape[1] for x in grid)

rows = []
for r in range(GRID_R):
    row_imgs = []
    for c in range(GRID_C):
        pid = grid[r * GRID_C + c]
        row_imgs.append(pad_to_same(tiles[pid], H, W))
    rows.append(np.concatenate(row_imgs, axis=1))

assembled = np.concatenate(rows, axis=0)

plt.figure(figsize=(7, 7))
plt.imshow(assembled)
plt.axis("off")
plt.title("Assembled Puzzle (Beam Search + Annealing)")
plt.tight_layout()
plt.savefig(str(OUT_DIR / "assembled.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved:", OUT_DIR / "assembled.png")

# =========================
# Visualization of matches (lines + scores)
# =========================
plt.figure(figsize=(7, 7))
plt.imshow(assembled)
plt.axis("off")

def center(r, c):
    return (c * W + W // 2, r * H + H // 2)

for r in range(GRID_R):
    for c in range(GRID_C):
        A = grid[r * GRID_C + c]

        if c + 1 < GRID_C:
            B = grid[r * GRID_C + (c + 1)]
            sc = right_cost[A][B]
            x1, y1 = center(r, c)
            x2, y2 = center(r, c + 1)
            plt.plot([x1, x2], [y1, y2], linewidth=3)
            plt.text((x1 + x2) / 2, (y1 + y2) / 2, f"{sc:.1f}", fontsize=10, ha="center", va="center")

        if r + 1 < GRID_R:
            B = grid[(r + 1) * GRID_C + c]
            sc = down_cost[A][B]
            x1, y1 = center(r, c)
            x2, y2 = center(r + 1, c)
            plt.plot([x1, x2], [y1, y2], linewidth=3)
            plt.text((x1 + x2) / 2, (y1 + y2) / 2, f"{sc:.1f}", fontsize=10, ha="center", va="center")

plt.title("Neighbor Matches Visualization (lines + scores)")
plt.tight_layout()
plt.savefig(str(OUT_DIR / "matches_visualization.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved:", OUT_DIR / "matches_visualization.png")

grid_txt = "\\n".join([" ".join(grid[r * GRID_C:(r + 1) * GRID_C]) for r in range(GRID_R)])
(OUT_DIR / "grid_order.txt").write_text(grid_txt)
print("Saved:", OUT_DIR / "grid_order.txt")
