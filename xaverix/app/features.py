"""The 150-dimensional leaf descriptor, implemented in NumPy.

Imported by both the training script and the application, so features are
computed by exactly one implementation. ``FEATURE_VERSION`` is written into
the model file and asserted on load; if this module changes, the model refuses
to load rather than predicting from features it was not fitted on.

Design rationale
----------------
A small convolutional network was rejected for two reasons:

1. Cost. Four convolutional layers are roughly 60-120 M multiply-accumulates
   per image. NumPy in Termux has no tuned GEMM, placing that in the
   200-700 ms range on an Arm CPU. This descriptor measures ~11 ms on x86.
2. Out-of-distribution detection. The descriptor is low-dimensional and each
   axis is physically interpretable (necrotic pixel fraction, texture energy,
   lesion scatter), so a Mahalanobis distance in this space is a usable
   novelty signal. The same statistic over a learned embedding is not. Since
   the classifier is trained on synthetic imagery, that novelty signal is the
   mechanism that keeps it reliable on real input.

All operations are vectorised; there are no per-pixel Python loops.
"""
from __future__ import annotations

import numpy as np

FEATURE_VERSION = "feat-v3-150"
FEATURE_DIM = 150

EPS = 1e-6

# ---------------------------------------------------------------- colour
def rgb_to_hsv(rgb):
    """rgb float32 (H,W,3) in [0,1] -> (h, s, v) each (H,W) in [0,1]."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    delta = mx - mn

    h = np.zeros_like(mx)
    nonzero = delta > EPS

    mask = (mx == r) & nonzero
    h[mask] = ((g - b)[mask] / delta[mask]) % 6.0
    mask = (mx == g) & nonzero
    h[mask] = ((b - r)[mask] / delta[mask]) + 2.0
    mask = (mx == b) & nonzero
    h[mask] = ((r - g)[mask] / delta[mask]) + 4.0

    h /= 6.0
    s = np.where(mx > EPS, delta / np.maximum(mx, EPS), 0.0)
    return h.astype(np.float32), s.astype(np.float32), mx.astype(np.float32)


def majority3(mask):
    """3x3 majority filter: cheap morphological cleanup without scipy."""
    padded = np.pad(mask.astype(np.uint8), 1)
    height, width = mask.shape
    total = np.zeros((height, width), dtype=np.uint8)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            total += padded[dy : dy + height, dx : dx + width]
    return total >= 5


def _fill_holes(grid):
    """Mark cells enclosed by True cells. Flood fill from the border, 14x14."""
    height, width = grid.shape
    outside = np.zeros(grid.shape, dtype=bool)
    stack = []
    for y in range(height):
        for x in (0, width - 1):
            if not grid[y, x]:
                stack.append((y, x))
    for x in range(width):
        for y in (0, height - 1):
            if not grid[y, x]:
                stack.append((y, x))
    while stack:
        y, x = stack.pop()
        if outside[y, x] or grid[y, x]:
            continue
        outside[y, x] = True
        if y > 0:
            stack.append((y - 1, x))
        if y < height - 1:
            stack.append((y + 1, x))
        if x > 0:
            stack.append((y, x - 1))
        if x < width - 1:
            stack.append((y, x + 1))
    return ~outside & ~grid


def leaf_mask(h, s, v):
    """Segment the leaf region, seeded from its green tissue.

    Thresholding on saturation alone is not sufficient: brown soil in the
    background also satisfies the necrosis colour test, which inflates the
    measured damage fraction. Against the renderer ground truth this produced
    a constant +25 to +30 point severity bias across all lesion types.

    The leaf is therefore located by its green tissue, and non-green pixels
    are admitted only where enclosed by it, which is where lesions occur. Hole
    filling runs on a coarse grid, so the flood fill cost is negligible.
    """
    coloured = (s > 0.15) & (v > 0.08) & (v < 0.99)
    green_core = majority3((h >= 0.18) & (h < 0.50) & (s > 0.22) & (v > 0.15))

    if not green_core.any():
        # No green tissue: not a leaf under this definition. Gate 0 abstains.
        return np.zeros_like(coloured)

    # Green tissue already defines the boundary accurately. The coarse grid is
    # used only to decide which non-green pixels are enclosed, and is eroded
    # first so mixed leaf/background blocks are excluded. Dilating instead
    # inflated measured leaf area from a true 41% of frame to 77%.
    block = 8
    coarse = block_mean(green_core, block) > 0.25
    filled = coarse | _fill_holes(coarse)
    interior = majority3(filled)  # erosion: needs 5 of 9 neighbours

    upsampled = np.repeat(np.repeat(interior, block, axis=0), block, axis=1)
    enclosed = np.zeros_like(coloured)
    height = min(upsampled.shape[0], enclosed.shape[0])
    width = min(upsampled.shape[1], enclosed.shape[1])
    enclosed[:height, :width] = upsampled[:height, :width]

    return majority3(green_core | (coloured & enclosed))


def majority3_loose(mask):
    """Dilation: a cell is kept if any 3x3 neighbour is set."""
    padded = np.pad(mask.astype(np.uint8), 1)
    height, width = mask.shape
    total = np.zeros((height, width), dtype=np.uint8)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            total += padded[dy : dy + height, dx : dx + width]
    return total >= 1


# ------------------------------------------------------------ damage masks
def damage_masks(h, s, v, leaf):
    """Tissue-condition masks, shared with severity estimation.

    Brightness bounds were selected by measuring severity against the renderer
    ground-truth lesion mask. An earlier necrosis bound of ``v < 0.55`` missed
    rust pustules, which are bright orange-brown, and a powdery bound of
    ``s < 0.20 & v > 0.75`` missed pale lesion rings and leaf-miner trails.
    The ``pale`` mask replaces the narrow powdery one and covers bleached,
    mined and powdery tissue together.
    """
    brown_hue = (h < 0.14) | (h > 0.93)
    necrotic = leaf & (s > 0.18) & (v < 0.70) & brown_hue
    water_soaked = leaf & (s < 0.35) & (v < 0.45)
    chlorotic = leaf & (h >= 0.12) & (h < 0.20) & (s > 0.28) & (v > 0.45)
    pale = leaf & (s < 0.30) & (v > 0.62)
    healthy_green = leaf & (h >= 0.20) & (h < 0.47) & (s > 0.25) & (v > 0.20)
    return {
        "necrotic": necrotic,
        "water_soaked": water_soaked,
        "chlorotic": chlorotic,
        "pale": pale,
        "healthy_green": healthy_green,
    }


# ------------------------------------------------------------------ texture
def _build_lbp_lut():
    lut = np.empty(256, dtype=np.int64)
    for value in range(256):
        bits = [(value >> i) & 1 for i in range(8)]
        transitions = sum(bits[i] != bits[(i + 1) % 8] for i in range(8))
        lut[value] = sum(bits) if transitions <= 2 else 9
    return lut


_LBP_LUT = _build_lbp_lut()

_LBP_OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))


def lbp_hist(grey, mask):
    """Rotation-tolerant uniform-LBP histogram: 9 uniform bins plus a catch-all."""
    centre = grey[1:-1, 1:-1]
    code = np.zeros(centre.shape, dtype=np.uint8)
    height, width = grey.shape
    for i, (dy, dx) in enumerate(_LBP_OFFSETS):
        neighbour = grey[1 + dy : height - 1 + dy, 1 + dx : width - 1 + dx]
        code |= ((neighbour >= centre).astype(np.uint8) << i)
    inner = mask[1:-1, 1:-1]
    if not inner.any():
        return np.zeros(10, dtype=np.float32)
    counts = np.bincount(_LBP_LUT[code[inner]], minlength=10).astype(np.float32)
    return counts / max(counts.sum(), 1.0)


def _sobel_stats(v, leaf):
    gx = np.zeros_like(v)
    gy = np.zeros_like(v)
    gx[:, 1:-1] = v[:, 2:] - v[:, :-2]
    gy[1:-1, :] = v[2:, :] - v[:-2, :]
    magnitude = np.sqrt(gx * gx + gy * gy)
    inside = magnitude[leaf] if leaf.any() else magnitude.ravel()
    if inside.size == 0:
        return np.zeros(6, dtype=np.float32)
    return np.array(
        [
            inside.mean(),
            inside.std(),
            np.percentile(inside, 90),
            (inside > 0.05).mean(),
            (inside > 0.10).mean(),
            (inside > 0.20).mean(),
        ],
        dtype=np.float32,
    )


# ---------------------------------------------------------------- helpers
def _hist(values, bins):
    if values.size == 0:
        return np.zeros(bins, dtype=np.float32)
    index = np.clip((values * bins).astype(np.int32), 0, bins - 1)
    counts = np.bincount(index, minlength=bins).astype(np.float32)
    return counts / max(counts.sum(), 1.0)


def _joint_hist(h, s, h_bins=8, s_bins=4):
    if h.size == 0:
        return np.zeros(h_bins * s_bins, dtype=np.float32)
    hi = np.clip((h * h_bins).astype(np.int32), 0, h_bins - 1)
    si = np.clip((s * s_bins).astype(np.int32), 0, s_bins - 1)
    counts = np.bincount(hi * s_bins + si, minlength=h_bins * s_bins).astype(np.float32)
    return counts / max(counts.sum(), 1.0)


def _channel_stats(values):
    """mean, std, third moment, p10, p50, p90 -> 6 numbers."""
    if values.size == 0:
        return np.zeros(6, dtype=np.float32)
    mean = float(values.mean())
    std = float(values.std())
    skew = float(((values - mean) ** 3).mean() / (std ** 3 + EPS))
    p10, p50, p90 = np.percentile(values, [10, 50, 90])
    return np.array(
        [mean, std, np.clip(skew, -8.0, 8.0), p10, p50, p90], dtype=np.float32
    )


def _index_stats(values):
    if values.size == 0:
        return np.zeros(4, dtype=np.float32)
    p10, p90 = np.percentile(values, [10, 90])
    return np.array([values.mean(), values.std(), p10, p90], dtype=np.float32)


def block_mean(mask, k):
    """Mean of a boolean mask over k x k blocks. Requires edge % k == 0."""
    height, width = mask.shape
    trimmed = mask[: height - height % k, : width - width % k].astype(np.float32)
    h2, w2 = trimmed.shape
    return trimmed.reshape(h2 // k, k, w2 // k, k).mean(axis=(1, 3))


def _gini(values):
    flat = np.sort(values.ravel())
    total = flat.sum()
    if total <= EPS:
        return 0.0
    n = flat.size
    index = np.arange(1, n + 1, dtype=np.float32)
    return float((2.0 * (index * flat).sum()) / (n * total) - (n + 1.0) / n)


def _lesion_structure(damage):
    """Where the damage sits, not just how much of it there is."""
    blocks = block_mean(damage, 8)  # 224 -> 28 x 28
    if blocks.size == 0 or min(blocks.shape) < 2:
        return np.zeros(10, dtype=np.float32)
    active = blocks > 0.2
    half_h = blocks.shape[0] // 2
    half_w = blocks.shape[1] // 2
    quadrants = [
        blocks[:half_h, :half_w].mean(),
        blocks[:half_h, half_w:].mean(),
        blocks[half_h:, :half_w].mean(),
        blocks[half_h:, half_w:].mean(),
    ]
    return np.array(
        [
            active.mean(),
            blocks.mean(),
            blocks.std(),
            blocks.max(),
            quadrants[0],
            quadrants[1],
            quadrants[2],
            quadrants[3],
            _gini(blocks),
            float(active.sum()),
        ],
        dtype=np.float32,
    )


def _centre_margin(necrotic, leaf):
    """Early blight favours the blade; late blight favours the margin."""
    height, width = leaf.shape
    yy, xx = np.mgrid[0:height, 0:width]
    cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
    radius = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)

    inner = leaf & (radius < 0.5)
    outer = leaf & (radius >= 0.5)
    inner_frac = necrotic[inner].mean() if inner.any() else 0.0
    outer_frac = necrotic[outer].mean() if outer.any() else 0.0
    ratio = float(inner_frac / (outer_frac + EPS))
    return np.array([np.clip(ratio, 0.0, 20.0), outer_frac], dtype=np.float32)


def _geometry(leaf):
    height, width = leaf.shape
    area = float(leaf.mean())
    if not leaf.any():
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    rows = np.where(leaf.any(axis=1))[0]
    cols = np.where(leaf.any(axis=0))[0]
    box_h = float(rows[-1] - rows[0] + 1)
    box_w = float(cols[-1] - cols[0] + 1)
    aspect = box_w / max(box_h, 1.0)
    fill = float(leaf.sum()) / max(box_h * box_w, 1.0)

    # Perimeter estimate: boundary pixels are those losing a 4-neighbour.
    shifted = (
        np.pad(leaf, ((1, 0), (0, 0)))[:-1]
        & np.pad(leaf, ((0, 1), (0, 0)))[1:]
        & np.pad(leaf, ((0, 0), (1, 0)))[:, :-1]
        & np.pad(leaf, ((0, 0), (0, 1)))[:, 1:]
    )
    perimeter = float((leaf & ~shifted).sum())
    compactness = perimeter / max(float(leaf.sum()), 1.0)

    # Row-run count: a crude connectedness proxy; a single leaf gives ~1 run.
    middle = leaf[height // 2]
    runs = float(np.count_nonzero(np.diff(middle.astype(np.int8)) == 1))
    return np.array(
        [area, np.clip(aspect, 0.0, 6.0), fill, compactness, min(runs, 12.0)],
        dtype=np.float32,
    )


# -------------------------------------------------------------- public API
def extract(rgb_u8):
    """Compute the descriptor for one image.

    Parameters
    ----------
    rgb_u8 : uint8 array (H, W, 3), expected 224x224.

    Returns
    -------
    (features, intermediates)
        ``features`` is float32[150]. ``intermediates`` carries the HSV planes
        and masks, which ``severity.py`` reuses instead of recomputing them,
        roughly halving the per-image cost.
    """
    rgb = np.asarray(rgb_u8, dtype=np.float32) / 255.0
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("expected an (H, W, 3) RGB image, got %r" % (rgb.shape,))
    if min(rgb.shape[:2]) < 32:
        raise ValueError(
            "image is %dx%d; too small to describe (expected %d square)"
            % (rgb.shape[0], rgb.shape[1], 224)
        )

    h, s, v = rgb_to_hsv(rgb)
    leaf = leaf_mask(h, s, v)
    leaf_frac = float(leaf.mean())

    # With no leaf-like region, fall back to the whole frame so the descriptor
    # stays well defined. Gate 0 in classifier.py is what declines to answer.
    stats_mask = leaf if leaf.any() else np.ones_like(leaf)

    masks = damage_masks(h, s, v, stats_mask)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    hm, sm, vm = h[stats_mask], s[stats_mask], v[stats_mask]
    rm, gm, bm = r[stats_mask], g[stats_mask], b[stats_mask]

    exg = (2.0 * g - r - b)[stats_mask]
    vari = ((g - r) / (g + r - b + EPS))[stats_mask]
    rg = ((r - g) / (r + g + EPS))[stats_mask]

    grey = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)
    damage = masks["necrotic"] | masks["water_soaked"] | masks["pale"]

    blocks = [
        _hist(hm, 16),                      # 16
        _hist(sm, 8),                       # 8
        _hist(vm, 8),                       # 8
        _joint_hist(hm, sm),                # 32
        _channel_stats(rm),                 # 6
        _channel_stats(gm),                 # 6
        _channel_stats(bm),                 # 6
        _channel_stats(hm),                 # 6
        _channel_stats(sm),                 # 6
        _channel_stats(vm),                 # 6
        _index_stats(exg),                  # 4
        _index_stats(np.clip(vari, -4, 4)), # 4
        _index_stats(rg),                   # 4
        np.array(                           # 5
            [
                masks["necrotic"][stats_mask].mean(),
                masks["chlorotic"][stats_mask].mean(),
                masks["water_soaked"][stats_mask].mean(),
                masks["pale"][stats_mask].mean(),
                masks["healthy_green"][stats_mask].mean(),
            ],
            dtype=np.float32,
        ),
        lbp_hist(grey, stats_mask),         # 10
        _sobel_stats(v, stats_mask),        # 6
        _lesion_structure(damage),          # 10
        _centre_margin(masks["necrotic"], stats_mask),  # 2
        _geometry(leaf),                    # 5
    ]

    features = np.concatenate([np.asarray(x, dtype=np.float32).ravel() for x in blocks])
    if features.size != FEATURE_DIM:
        raise AssertionError(
            "descriptor is %d dims, expected %d (FEATURE_VERSION must be bumped)"
            % (features.size, FEATURE_DIM)
        )
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    intermediates = {
        "h": h,
        "s": s,
        "v": v,
        "leaf": leaf,
        "leaf_frac": leaf_frac,
        "masks": masks,
        "green_frac": float(masks["healthy_green"][stats_mask].mean()),
    }
    return features, intermediates


def extract_only(rgb_u8):
    return extract(rgb_u8)[0]
