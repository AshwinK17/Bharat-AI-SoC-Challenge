"""Procedural leaf renderer and negative-set generator.

The classifier is trained on procedurally generated imagery rather than a
curated field dataset, so it is a pipeline-validation model and no real-world
accuracy is claimed for it. What makes that tractable is pairing it with the
abstention cascade, for which this module supplies the training signal.

Two categories are rendered:

* **Leaves**, healthy and damaged, which train the classifier. Each render
  returns its exact lesion mask, giving ground truth for severity calibration.
* **A negative set** of scenes that are not a single leaf, used to fit the
  out-of-distribution thresholds. It covers cases representative of real user
  error: a leaf photographed on ruled paper, bare soil, sky, cloth, skin,
  printed text, uniform foliage and motion blur.

NumPy and Pillow only.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

SIZE = 224

LESION_KINDS = (
    "target_rings",     # early-blight-like concentric lesions
    "water_soaked",     # late-blight-like marginal patches
    "fine_spots",       # bacterial-spot-like speckling with halos
    "rust_pustules",
    "powdery",
    "miner_trails",
    "chewed_edge",
    "interveinal",      # nutrient-deficiency-like chlorosis
)


# ------------------------------------------------------------------ noise
def value_noise(rng, size=SIZE, cells=8, octaves=4):
    """Smooth multi-octave noise in [0,1] via successive bilinear upsampling."""
    total = np.zeros((size, size), dtype=np.float32)
    amplitude = 1.0
    weight = 0.0
    for octave in range(octaves):
        n = max(2, int(cells * (2 ** octave)))
        grid = rng.random((n, n)).astype(np.float32)
        layer = np.asarray(
            Image.fromarray((grid * 255).astype(np.uint8)).resize(
                (size, size), Image.BILINEAR
            ),
            dtype=np.float32,
        ) / 255.0
        total += amplitude * layer
        weight += amplitude
        amplitude *= 0.5
    return total / max(weight, 1e-6)


def _radial(size=SIZE):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    centre = (size - 1) / 2.0
    dy = (yy - centre) / centre
    dx = (xx - centre) / centre
    radius = np.sqrt(dx * dx + dy * dy)
    theta = np.arctan2(dy, dx)
    return radius, theta, dy, dx


_RADIUS, _THETA, _DY, _DX = _radial()


# -------------------------------------------------------------- background
BACKGROUND_KINDS = ("soil", "sky", "cloth", "wood", "skin", "grass", "flat")


def _background(rng, kind=None):
    """Procedural background. ``kind`` must be honoured when supplied.

    This used to ignore its caller and always pick at random, which meant a
    negative sample labelled "soil" could be rendered as green grass -- and a
    green frame is then classified as a healthy leaf with high confidence.
    A mislabelled negative is worse than no negative, because it corrupts both
    the fitted thresholds and the demo.
    """
    kind = kind or str(rng.choice(BACKGROUND_KINDS))
    noise = value_noise(rng, cells=rng.integers(3, 10), octaves=4)

    if kind == "soil":
        base = np.array([0.42, 0.31, 0.22], dtype=np.float32)
        amount = 0.22
    elif kind == "sky":
        base = np.array([0.62, 0.72, 0.86], dtype=np.float32)
        amount = 0.10
    elif kind == "cloth":
        base = np.array(rng.random(3) * 0.6 + 0.3, dtype=np.float32)
        amount = 0.08
    elif kind == "wood":
        base = np.array([0.55, 0.38, 0.22], dtype=np.float32)
        amount = 0.18
    elif kind == "skin":
        base = np.array([0.76, 0.58, 0.47], dtype=np.float32)
        amount = 0.07
    elif kind == "grass":
        base = np.array([0.28, 0.42, 0.18], dtype=np.float32)
        amount = 0.25
    else:
        base = np.array(rng.random(3) * 0.7 + 0.2, dtype=np.float32)
        amount = 0.03

    canvas = base[None, None, :] + amount * (noise - 0.5)[..., None]
    return np.clip(canvas, 0.0, 1.0)


# ------------------------------------------------------------- leaf shape
def _leaf_shape(rng):
    """Lobed leaf silhouette via a perturbed polar radius."""
    scale = rng.uniform(0.62, 0.94)
    lobes = rng.integers(3, 8)
    phase = rng.uniform(0, 2 * np.pi)
    wobble = rng.uniform(0.08, 0.20)
    fine = rng.uniform(0.03, 0.10)
    squash = rng.uniform(0.70, 1.0)
    rotation = rng.uniform(0, 2 * np.pi)

    theta = _THETA + rotation
    boundary = scale * (
        1.0 + wobble * np.sin(lobes * theta + phase) + fine * np.sin(11 * theta)
    )
    # Elliptical squash, applied in the rotated frame.
    ry = _DY * np.cos(rotation) - _DX * np.sin(rotation)
    rx = _DY * np.sin(rotation) + _DX * np.cos(rotation)
    radius = np.sqrt((rx / squash) ** 2 + ry ** 2)
    return radius <= boundary, rotation


def _veins(rng, rotation):
    """Midrib plus lateral veins, as a brightness modulation field."""
    ry = _DY * np.cos(rotation) - _DX * np.sin(rotation)
    rx = _DY * np.sin(rotation) + _DX * np.cos(rotation)

    midrib = np.exp(-((rx / 0.035) ** 2))
    count = int(rng.integers(5, 10))
    lateral = np.zeros_like(midrib)
    for i in range(count):
        offset = -0.85 + 1.7 * (i + 0.5) / count
        slope = rng.uniform(0.45, 0.85) * (1 if (i % 2 == 0) else -1)
        distance = rx - slope * (ry - offset)
        lateral += np.exp(-((distance / 0.030) ** 2)) * np.exp(
            -((ry - offset) / 0.45) ** 2
        )
    return np.clip(midrib + 0.65 * lateral, 0.0, 1.5)


def _leaf_colour(rng):
    hue = rng.uniform(0.20, 0.38)
    sat = rng.uniform(0.35, 0.80)
    val = rng.uniform(0.32, 0.72)
    # HSV -> RGB for a single colour, written out to avoid a dependency.
    i = int(hue * 6.0) % 6
    f = hue * 6.0 - int(hue * 6.0)
    p = val * (1 - sat)
    q = val * (1 - sat * f)
    t = val * (1 - sat * (1 - f))
    table = [(val, t, p), (q, val, p), (p, val, t), (p, q, val), (t, p, val), (val, p, q)]
    return np.array(table[i], dtype=np.float32)


# ---------------------------------------------------------------- lesions
def _blob_mask(rng, centre_y, centre_x, radius, raggedness=0.35):
    dy = (_DY - centre_y)
    dx = (_DX - centre_x)
    distance = np.sqrt(dy * dy + dx * dx)
    angle = np.arctan2(dy, dx)
    edge = radius * (
        1.0
        + raggedness * np.sin(rng.integers(3, 9) * angle + rng.uniform(0, 6.28))
        + 0.5 * raggedness * np.sin(rng.integers(9, 17) * angle)
    )
    return distance <= edge, distance, edge


def _apply_lesions(rng, canvas, leaf, rotation, kinds):
    """Paint lesions onto the leaf. Returns the damaged-pixel ground truth."""
    lesion = np.zeros(leaf.shape, dtype=bool)
    chlorosis = np.zeros(leaf.shape, dtype=bool)
    inside = np.argwhere(leaf)
    if inside.size == 0:
        return lesion, chlorosis, leaf

    def random_point(margin_bias=False):
        idx = rng.integers(0, len(inside))
        y, x = inside[idx]
        cy = (y - (SIZE - 1) / 2.0) / ((SIZE - 1) / 2.0)
        cx = (x - (SIZE - 1) / 2.0) / ((SIZE - 1) / 2.0)
        if margin_bias:
            norm = np.sqrt(cy * cy + cx * cx) + 1e-6
            scale = rng.uniform(0.75, 1.0)
            cy, cx = cy / norm * scale * 0.85, cx / norm * scale * 0.85
        return cy, cx

    leaf_out = leaf.copy()

    for kind in kinds:
        if kind == "target_rings":
            for _ in range(int(rng.integers(3, 10))):
                cy, cx = random_point()
                radius = rng.uniform(0.045, 0.11)
                blob, distance, edge = _blob_mask(rng, cy, cx, radius, 0.28)
                spot = blob & leaf_out
                brown = np.array([0.30, 0.19, 0.09], dtype=np.float32)
                canvas[spot] = brown
                # concentric rings
                rings = np.cos(distance / max(radius, 1e-6) * 9.0) > 0.45
                canvas[spot & rings] = brown * 0.62
                halo = (distance <= edge * 1.5) & ~blob & leaf_out
                canvas[halo] = np.array([0.72, 0.68, 0.22], dtype=np.float32)
                lesion |= spot
                chlorosis |= halo

        elif kind == "water_soaked":
            for _ in range(int(rng.integers(1, 4))):
                cy, cx = random_point(margin_bias=True)
                blob, distance, edge = _blob_mask(rng, cy, cx, rng.uniform(0.14, 0.30), 0.45)
                patch = blob & leaf_out
                canvas[patch] = np.array([0.33, 0.34, 0.29], dtype=np.float32)
                pale = (distance <= edge * 1.18) & ~blob & leaf_out
                canvas[pale] = np.array([0.70, 0.71, 0.60], dtype=np.float32)
                lesion |= patch | pale

        elif kind == "fine_spots":
            count = int(rng.integers(25, 70))
            for _ in range(count):
                cy, cx = random_point()
                blob, distance, edge = _blob_mask(rng, cy, cx, rng.uniform(0.010, 0.026), 0.2)
                spot = blob & leaf_out
                canvas[spot] = np.array([0.17, 0.13, 0.10], dtype=np.float32)
                halo = (distance <= edge * 2.0) & ~blob & leaf_out
                canvas[halo] = np.array([0.74, 0.70, 0.24], dtype=np.float32)
                lesion |= spot
                chlorosis |= halo

        elif kind == "rust_pustules":
            noise = value_noise(rng, cells=42, octaves=2)
            pustules = (noise > 0.70) & leaf_out
            canvas[pustules] = np.array([0.62, 0.33, 0.11], dtype=np.float32)
            lesion |= pustules

        elif kind == "powdery":
            noise = value_noise(rng, cells=5, octaves=3)
            patch = (noise > 0.58) & leaf_out
            blend = rng.uniform(0.55, 0.85)
            canvas[patch] = (1 - blend) * canvas[patch] + blend * np.array(
                [0.90, 0.90, 0.88], dtype=np.float32
            )
            lesion |= patch

        elif kind == "miner_trails":
            trail = np.zeros(leaf.shape, dtype=bool)
            for _ in range(int(rng.integers(1, 4))):
                y, x = inside[rng.integers(0, len(inside))]
                angle = rng.uniform(0, 6.28)
                for _ in range(int(rng.integers(90, 190))):
                    angle += rng.normal(0, 0.45)
                    y = int(np.clip(y + 1.7 * np.sin(angle), 1, SIZE - 2))
                    x = int(np.clip(x + 1.7 * np.cos(angle), 1, SIZE - 2))
                    trail[y - 1 : y + 2, x - 1 : x + 2] = True
            trail &= leaf_out
            canvas[trail] = np.array([0.82, 0.78, 0.60], dtype=np.float32)
            lesion |= trail

        elif kind == "chewed_edge":
            for _ in range(int(rng.integers(2, 6))):
                cy, cx = random_point(margin_bias=True)
                blob, _, _ = _blob_mask(rng, cy, cx, rng.uniform(0.07, 0.16), 0.5)
                bite = blob & leaf_out
                leaf_out &= ~bite
                # Edges of a chewed hole dry out and brown.
                from app.features import majority3

                ring = majority3(bite) & ~bite & leaf_out
                canvas[ring] = np.array([0.34, 0.24, 0.12], dtype=np.float32)
                lesion |= ring

        elif kind == "interveinal":
            veins = _veins(rng, rotation)
            between = (veins < 0.18) & leaf_out
            noise = value_noise(rng, cells=4, octaves=2)
            between &= noise > 0.35
            canvas[between] = np.array([0.76, 0.73, 0.26], dtype=np.float32)
            chlorosis |= between

    return lesion, chlorosis, leaf_out


# ------------------------------------------------------------- rendering
def render_leaf(infected, rng, severity_hint=None):
    """Render one leaf.

    Returns (rgb_u8, info) where ``info['lesion_fraction']`` is the exact
    fraction of leaf area that is damaged — the ground truth used to calibrate
    the severity heuristic.
    """
    canvas = _background(rng)
    leaf, rotation = _leaf_shape(rng)

    colour = _leaf_colour(rng)
    texture = value_noise(rng, cells=rng.integers(4, 12), octaves=4)
    veins = _veins(rng, rotation)

    leaf_rgb = colour[None, None, :] * (
        0.80 + 0.40 * texture[..., None] - 0.22 * veins[..., None]
    )
    canvas = np.where(leaf[..., None], np.clip(leaf_rgb, 0, 1), canvas)

    kinds = []
    if infected:
        how_many = 1 if severity_hint is None else int(rng.integers(1, 3))
        if severity_hint is not None and severity_hint > 0.35:
            how_many = int(rng.integers(2, 4))
        kinds = list(rng.choice(LESION_KINDS, size=how_many, replace=False))

    lesion, chlorosis, leaf_final = _apply_lesions(rng, canvas, leaf, rotation, kinds)

    # Lighting gradient, then specular highlights: applied after lesions so
    # the illumination is global, as in a real photograph.
    gradient = value_noise(rng, cells=2, octaves=2)
    canvas *= (0.72 + 0.56 * gradient)[..., None]
    for _ in range(int(rng.integers(0, 3))):
        cy, cx = rng.uniform(-0.7, 0.7, size=2)
        spot = np.exp(-(((_DY - cy) ** 2 + (_DX - cx) ** 2) / rng.uniform(0.004, 0.03)))
        canvas += spot[..., None] * rng.uniform(0.15, 0.45)

    canvas = np.clip(canvas, 0.0, 1.0)

    leaf_area = max(float(leaf_final.sum()), 1.0)
    damaged = float((lesion & leaf_final).sum()) + 0.5 * float(
        (chlorosis & leaf_final & ~lesion).sum()
    )
    info = {
        "infected": bool(infected),
        "kinds": [str(k) for k in kinds],
        "leaf_fraction": float(leaf_final.mean()),
        "lesion_fraction": 100.0 * damaged / leaf_area,
    }
    return (canvas * 255).astype(np.uint8), info


# -------------------------------------------------------------- negatives
NEGATIVE_KINDS = (
    "lined_paper",
    "grass",
    "flat_colour",
    "gradient",
    "noise",
    "soil",
    "sky",
    "brick",
    "cloth",
    "wood",
    "skin",
    "printed_text",
    "scrambled_leaf",
    "greyscale_leaf",
    "overexposed",
    "underexposed",
    "motion_blur",
    "multi_leaf_clutter",
)


def render_negative(rng, kind=None):
    """Something that is NOT a single leaf. Used to fit the abstention gates."""
    kind = kind or str(rng.choice(NEGATIVE_KINDS))

    if kind == "lined_paper":
        # A dark leaf-coloured region photographed on ruled notebook paper:
        # a common framing error that must not receive a verdict.
        canvas = np.full((SIZE, SIZE, 3), 0.94, dtype=np.float32)
        canvas += 0.03 * (value_noise(rng, cells=6, octaves=3) - 0.5)[..., None]
        spacing = int(rng.integers(14, 24))
        offset = int(rng.integers(0, spacing))
        for y in range(offset, SIZE, spacing):
            canvas[y : y + 1, :, :] = np.array([0.55, 0.62, 0.80], dtype=np.float32)
        if rng.random() < 0.8:
            blob, _, _ = _blob_mask(rng, rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3),
                                    rng.uniform(0.25, 0.5), 0.4)
            dark = rng.uniform(0.10, 0.30)
            canvas[blob] = np.array([dark, dark * 1.15, dark * 0.8], dtype=np.float32)
        return np.clip(canvas * 255, 0, 255).astype(np.uint8)

    if kind == "flat_colour":
        colour = rng.random(3).astype(np.float32)
        canvas = np.tile(colour, (SIZE, SIZE, 1))
        canvas += 0.02 * rng.standard_normal((SIZE, SIZE, 3)).astype(np.float32)
        return np.clip(canvas * 255, 0, 255).astype(np.uint8)

    if kind == "gradient":
        a, b = rng.random(3).astype(np.float32), rng.random(3).astype(np.float32)
        ramp = np.linspace(0, 1, SIZE, dtype=np.float32)
        if rng.random() < 0.5:
            field = np.repeat(ramp[:, None], SIZE, axis=1)[..., None]
        else:
            field = np.repeat(ramp[None, :], SIZE, axis=0)[..., None]
        canvas = a[None, None, :] * (1 - field) + b[None, None, :] * field
        return np.clip(canvas * 255, 0, 255).astype(np.uint8)

    if kind == "noise":
        if rng.random() < 0.5:
            canvas = rng.random((SIZE, SIZE, 3)).astype(np.float32)
        else:
            canvas = np.stack([value_noise(rng, cells=20, octaves=3) for _ in range(3)], -1)
        return np.clip(canvas * 255, 0, 255).astype(np.uint8)

    if kind == "brick":
        canvas = np.tile(np.array([0.55, 0.28, 0.20], dtype=np.float32), (SIZE, SIZE, 1))
        canvas += 0.10 * (value_noise(rng, cells=12, octaves=3) - 0.5)[..., None]
        for y in range(0, SIZE, 28):
            canvas[y : y + 3, :, :] = 0.82
        for i, x in enumerate(range(0, SIZE, 56)):
            shift = 28 if i % 2 else 0
            canvas[:, (x + shift) % SIZE : (x + shift) % SIZE + 3, :] = 0.82
        return np.clip(canvas * 255, 0, 255).astype(np.uint8)

    if kind == "printed_text":
        canvas = np.full((SIZE, SIZE, 3), 0.96, dtype=np.float32)
        for y in range(10, SIZE - 10, 13):
            x = 8
            while x < SIZE - 20:
                word = int(rng.integers(8, 28))
                canvas[y : y + 5, x : x + word, :] = rng.uniform(0.05, 0.25)
                x += word + int(rng.integers(4, 9))
        return np.clip(canvas * 255, 0, 255).astype(np.uint8)

    if kind in ("scrambled_leaf", "greyscale_leaf"):
        image, _ = render_leaf(bool(rng.random() < 0.5), rng)
        array = image.astype(np.float32) / 255.0
        if kind == "greyscale_leaf":
            grey = (0.299 * array[..., 0] + 0.587 * array[..., 1] + 0.114 * array[..., 2])
            array = np.stack([grey] * 3, axis=-1)
        else:
            block = 28
            blocks = array.reshape(SIZE // block, block, SIZE // block, block, 3)
            blocks = blocks.transpose(0, 2, 1, 3, 4).reshape(-1, block, block, 3)
            rng.shuffle(blocks)
            n = SIZE // block
            array = (
                blocks.reshape(n, n, block, block, 3)
                .transpose(0, 2, 1, 3, 4)
                .reshape(SIZE, SIZE, 3)
            )
        return np.clip(array * 255, 0, 255).astype(np.uint8)

    if kind in ("overexposed", "underexposed", "motion_blur"):
        image, _ = render_leaf(bool(rng.random() < 0.5), rng)
        pil = Image.fromarray(image)
        if kind == "motion_blur":
            pil = pil.filter(ImageFilter.GaussianBlur(rng.uniform(3.5, 7.0)))
            return np.asarray(pil)
        array = np.asarray(pil, dtype=np.float32) / 255.0
        if kind == "overexposed":
            array = np.clip(array * rng.uniform(2.2, 3.5), 0, 1)
        else:
            array = array * rng.uniform(0.05, 0.16)
        return np.clip(array * 255, 0, 255).astype(np.uint8)

    if kind == "multi_leaf_clutter":
        canvas = _background(rng)
        for _ in range(int(rng.integers(4, 9))):
            leaf, rotation = _leaf_shape(rng)
            scale = rng.uniform(0.2, 0.4)
            small = leaf & (_RADIUS < scale)
            shift_y, shift_x = rng.integers(-70, 70, size=2)
            small = np.roll(np.roll(small, shift_y, axis=0), shift_x, axis=1)
            colour = _leaf_colour(rng)
            canvas = np.where(small[..., None], colour[None, None, :], canvas)
        return np.clip(canvas * 255, 0, 255).astype(np.uint8)

    # soil / sky / cloth / wood / skin: forward the kind, do not re-randomise
    canvas = _background(rng, kind=kind if kind in BACKGROUND_KINDS else None)
    return np.clip(canvas * 255, 0, 255).astype(np.uint8)


# ----------------------------------------------------------- augmentation
def augment(image_u8, rng):
    """Photographic augmentation, applied before feature extraction.

    This runs hard on colour on purpose. Without it the classifier learns the
    renderer's palette instead of learning what necrotic tissue looks like.
    """
    pil = Image.fromarray(image_u8)

    if rng.random() < 0.8:
        pil = pil.rotate(float(rng.uniform(0, 360)), resample=Image.BILINEAR)
    if rng.random() < 0.5:
        pil = pil.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < 0.3:
        pil = pil.transpose(Image.FLIP_TOP_BOTTOM)
    if rng.random() < 0.45:
        pil = pil.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.2, 1.2))))
    if rng.random() < 0.3:
        small = max(48, int(rng.integers(64, 140)))
        pil = pil.resize((small, small), Image.BILINEAR).resize(
            (SIZE, SIZE), Image.BILINEAR
        )

    array = np.asarray(pil, dtype=np.float32) / 255.0

    if rng.random() < 0.8:  # gamma
        array = np.clip(array, 1e-4, 1.0) ** float(rng.uniform(0.7, 1.4))
    if rng.random() < 0.8:  # per-channel white balance
        array *= (1.0 + rng.uniform(-0.08, 0.08, size=3)).astype(np.float32)
    if rng.random() < 0.6:  # brightness / contrast
        array = (array - 0.5) * float(rng.uniform(0.82, 1.22)) + 0.5 + float(
            rng.uniform(-0.08, 0.08)
        )
    if rng.random() < 0.6:  # sensor noise
        array += rng.standard_normal(array.shape).astype(np.float32) * float(
            rng.uniform(0.004, 0.030)
        )
    if rng.random() < 0.3:  # coarse quantisation, JPEG-ish
        levels = int(rng.integers(10, 32))
        array = np.round(array * levels) / levels

    return (np.clip(array, 0.0, 1.0) * 255).astype(np.uint8)
