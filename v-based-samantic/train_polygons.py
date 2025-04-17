"""
train_polygons.py (RGB Prediction – semantic‑label version)

This script trains a model on top of CLIP's text encoder to predict RGB colour
values directly from *text* descriptions of shapes (plus polygon geometry and
parent context).  **Training labels are mined from the child description text**
(`rgb(…)`, `#RRGGBB`, or a tiny training‑time list of basic colour words).
No extra fields are added to the JSON files; inference remains colour‑table‑free.
"""

from __future__ import annotations

import os, sys, json, re, logging, math, glob, datetime
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import clip

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOG_DIR, f"train_rgb_polygons_{_ts}.log")

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] [%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE)],
)

# ---------------------------------------------------------------------------
# Geometry helpers (unchanged)
# ---------------------------------------------------------------------------

def process_polygon(polygon: List[List[float]], normalize: bool = True) -> torch.Tensor:
    """Convert polygon coordinates → 10‑d feature vector (optionally normalised)."""
    if not polygon:
        return torch.zeros(10)

    xs, ys = zip(*polygon)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width, height = max_x - min_x, max_y - min_y

    # Area (Shoelace)
    area = 0.5 * abs(
        sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(polygon, polygon[1:] + [polygon[0]]))
    )

    # Centroid
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)

    # Perimeter
    perimeter = sum(
        math.hypot(x1 - x0, y1 - y0)
        for (x0, y0), (x1, y1) in zip(polygon, polygon[1:] + [polygon[0]])
    )

    aspect_ratio = width / height if height else 1.0
    dists = [math.hypot(x - cx, y - cy) for x, y in polygon]
    features = [
        width,
        height,
        area,
        perimeter,
        cx,
        cy,
        aspect_ratio,
        sum(dists) / len(dists),
        max(dists),
        len(polygon),
    ]

    ft = torch.tensor(features, dtype=torch.float32)
    if normalize:
        nz = ft != 0
        ft[nz] = ft[nz].log1p()
        ft = torch.tanh(ft * 0.1)
    return ft

# ---------------------------------------------------------------------------
# Text‑to‑RGB label mining  (training‑time only!)
# ---------------------------------------------------------------------------

_WORD_TO_RGB: Dict[str, Tuple[int, int, int]] = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (255, 0, 0),
    "green": (0, 128, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "pink": (255, 192, 203),
    "brown": (165, 42, 42),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
}

_RGB_RE = re.compile(r"rgb\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)")
_HEX_RE = re.compile(r"#([0-9a-fA-F]{6}|[0-9a-fA-F]{3})")


def extract_rgb_from_text(text: str) -> Optional[Tuple[int, int, int]]:
    """Return (R,G,B) if the description explicitly encodes a colour; else None."""
    text_l = text.lower()

    # rgb(r,g,b)
    if m := _RGB_RE.search(text_l):
        r, g, b = map(int, m.groups())
        if all(0 <= v <= 255 for v in (r, g, b)):
            return r, g, b

    # #RRGGBB or #RGB
    if m := _HEX_RE.search(text_l):
        hexstr = m.group(1)
        if len(hexstr) == 3:
            hexstr = "".join(ch * 2 for ch in hexstr)
        val = int(hexstr, 16)
        return (val >> 16) & 255, (val >> 8) & 255, val & 255

    # basic colour words
    for word, rgb in _WORD_TO_RGB.items():
        if word in text_l:
            return rgb
    return None

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RGBShapesDataset(data.Dataset):
    """Builds samples **only** for shapes whose description yields a colour label."""

    def __init__(self, shapes: List[dict], id_to_shape: Dict[int, dict]):
        self.samples = []
        skipped = 0

        for shape in shapes:
            child_desc = shape.get("description", "")
            rgb = extract_rgb_from_text(child_desc)
            if rgb is None:
                skipped += 1
                continue  # no label ⇒ skip sample

            # Build features
            polygon_features = process_polygon(shape.get("polygon", []))
            parent_desc = ""
            if (pid := shape.get("parent")) is not None and pid in id_to_shape:
                parent_desc = id_to_shape[pid].get("description", "")

            norm_rgb = [c / 255.0 for c in rgb]
            self.samples.append((parent_desc, polygon_features, child_desc, norm_rgb))

        logging.info(
            "Constructed dataset: %d labelled samples (skipped %d shapes without colour)",
            len(self.samples), skipped,
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p_desc, poly_ft, c_desc, rgb = self.samples[idx]
        return p_desc, poly_ft, c_desc, torch.tensor(rgb, dtype=torch.float32)

# ---------------------------------------------------------------------------
# Model (unchanged)
# ---------------------------------------------------------------------------

class RGBPredictionModel(nn.Module):
    def __init__(self, text_dim: int, poly_dim: int = 10):
        super().__init__()
        self.poly = nn.Sequential(nn.Linear(poly_dim, 32), nn.ReLU(), nn.Dropout(0.2))
        total = text_dim * 2 + 32
        self.net = nn.Sequential(
            nn.Linear(total, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 3), nn.Sigmoid(),  # → 0‑1 then ×255 at loss time
        )

    def forward(self, p_txt, poly, c_txt):
        return self.net(torch.cat([p_txt, self.poly(poly), c_txt], dim=1)) * 255.0

# ---------------------------------------------------------------------------
# Training loop (unchanged logic, improved comments)
# ---------------------------------------------------------------------------

def rgb_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(((a - b) ** 2).sum(dim=-1))


def train_model(dataset, clip_model, rgb_model, optimiser, device, epochs=50, batch=16):
    val_split = 0.2
    n_val = int(len(dataset) * val_split)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [len(dataset) - n_val, n_val], generator=torch.Generator().manual_seed(42))
    train_loader = data.DataLoader(train_ds, batch_size=batch, shuffle=True)
    val_loader = data.DataLoader(val_ds, batch_size=batch)

    loss_fn = nn.MSELoss()
    thresh = 30.0  # RGB Euclidean distance for “accuracy”

    logging.info("Training %d samples, validating %d samples", len(train_ds), len(val_ds))

    for epoch in range(1, epochs + 1):
        rgb_model.train()
        t_loss = t_corr = 0
        t_dist = []
        for p_txt, poly, c_txt, tgt in train_loader:
            poly, tgt = poly.to(device), tgt.to(device) * 255.0
            p_emb = clip_model.encode_text(clip.tokenize(p_txt, truncate=True).to(device)).detach()
            c_emb = clip_model.encode_text(clip.tokenize(c_txt, truncate=True).to(device)).detach()
            p_emb /= p_emb.norm(dim=-1, keepdim=True)
            c_emb /= c_emb.norm(dim=-1, keepdim=True)

            optimiser.zero_grad()
            pred = rgb_model(p_emb, poly, c_emb)
            loss = loss_fn(pred, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rgb_model.parameters(), 1.0)
            optimiser.step()

            dist = rgb_distance(pred, tgt)
            t_corr += (dist < thresh).sum().item()
            t_loss += loss.item() * tgt.size(0)
            t_dist.extend(dist.cpu().tolist())

        # ----------------- validation -----------------
        rgb_model.eval()
        v_loss = v_corr = 0
        v_dist = []
        with torch.no_grad():
            for p_txt, poly, c_txt, tgt in val_loader:
                poly, tgt = poly.to(device), tgt.to(device) * 255.0
                p_emb = clip_model.encode_text(clip.tokenize(p_txt, truncate=True).to(device))
                c_emb = clip_model.encode_text(clip.tokenize(c_txt, truncate=True).to(device))
                p_emb /= p_emb.norm(dim=-1, keepdim=True)
                c_emb /= c_emb.norm(dim=-1, keepdim=True)
                pred = rgb_model(p_emb, poly, c_emb)
                loss = loss_fn(pred, tgt)
                dist = rgb_distance(pred, tgt)
                v_corr += (dist < thresh).sum().item()
                v_loss += loss.item() * tgt.size(0)
                v_dist.extend(dist.cpu().tolist())

        # ----------------- logging -----------------
        n_tr, n_val = len(train_ds), len(val_ds)
        log_msg = (
            f"Epoch {epoch}/50: Train Loss = {t_loss/n_tr:.4f}, Val Loss = {v_loss/n_val:.4f}, "
            f"Avg RGB Distance = {np.mean(v_dist):.2f}, "
            f"Accuracy: Train = {100*t_corr/n_tr:.2f}%, Val = {100*v_corr/n_val:.2f}%, "
            f"Overfitting = {100*t_corr/n_tr - 100*v_corr/n_val:.2f}%"
        )
        logging.info(log_msg)

    return rgb_model

# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def main():
    logging.info("Starting RGB colour‑prediction training (semantic labels)")
    root = os.path.dirname(os.path.abspath(__file__))
    json_dir = os.path.join(root, "shapes_jsons") if os.path.isdir(os.path.join(root, "shapes_jsons")) else root
    json_files = glob.glob(os.path.join(json_dir, "**", "*.json"), recursive=True)
    if not json_files:
        logging.error("No JSON files found for training!")
        sys.exit(1)

    # Load shapes
    shapes: List[dict] = []
    for path in json_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                shp = json.load(f)
                shapes.extend(shp)
        except Exception as e:
            logging.warning("Failed to load %s: %s", path, e)
    if not shapes:
        logging.error("Loaded zero shapes – aborting")
        sys.exit(1)

    id_to_shape = {s["id"]: s for s in shapes}
    dataset = RGBShapesDataset(shapes, id_to_shape)
    if len(dataset) == 0:
        logging.error("No shapes contained explicit colour information – cannot train")
        sys.exit(1)

    # Device & CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    text_dim = clip_model.encode_text(clip.tokenize(["dummy"], truncate=True).to(device)).shape[-1]

    model = RGBPredictionModel(text_dim).to(device)
    optimiser = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    model = train_model(dataset, clip_model, model, optimiser, device)

    # Save
    torch.save(model.state_dict(), os.path.join(root, "rgb_model_weights.pth"))
    with open(os.path.join(root, "rgb_model_info.json"), "w", encoding="utf-8") as f:
        json.dump({"embed_dim": text_dim, "poly_feature_dim": 10, "hidden_dims": [256, 128], "output_dim": 3, "dropout_rates": [0.4, 0.3]}, f)
    logging.info("Training complete – model saved.")


if __name__ == "__main__":
    main()
