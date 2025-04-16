"""
 test_polygons.py  –  RGB Prediction (semantic‑label version)
 -----------------------------------------------------------
 Uses a trained model to predict RGB colours for polygons from text descriptions
 (parent + child) plus polygon geometry.  The script now tolerates *both* weight
 files trained with the old layer names (poly_net / main_network) and the new
 ones (poly / net) so you can swap models without editing this file.
"""

from __future__ import annotations

import os, sys, json, logging, math, argparse, datetime, traceback
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import clip
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOG_DIR, f"test_rgb_polygons_{_ts}.log")
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] [%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE)],
)

# ---------------------------------------------------------------------------
# Geometry helper (unchanged from training)
# ---------------------------------------------------------------------------

def process_polygon(polygon: List[List[float]], normalize: bool = True) -> torch.Tensor:
    if not polygon:
        return torch.zeros(10)
    xs, ys = zip(*polygon)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width, height = max_x - min_x, max_y - min_y
    area = 0.5 * abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(polygon, polygon[1:] + [polygon[0]])))
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    perimeter = sum(math.hypot(x1 - x0, y1 - y0) for (x0, y0), (x1, y1) in zip(polygon, polygon[1:] + [polygon[0]]))
    aspect = width / height if height else 1.0
    dists = [math.hypot(x - cx, y - cy) for x, y in polygon]
    feats = [width, height, area, perimeter, cx, cy, aspect, sum(dists) / len(dists), max(dists), len(polygon)]
    ft = torch.tensor(feats, dtype=torch.float32)
    if normalize:
        nz = ft != 0
        ft[nz] = ft[nz].log1p()
        ft = torch.tanh(ft * 0.1)
    return ft

# ---------------------------------------------------------------------------
# Model definition (must match training script names)
# ---------------------------------------------------------------------------

class RGBPredictionModel(nn.Module):
    def __init__(self, text_embed_dim: int, poly_feature_dim: int = 10, hidden_dims=(256, 128), dropout_rates=(0.4, 0.3)):
        super().__init__()
        assert len(hidden_dims) == len(dropout_rates)
        self.poly_net = nn.Sequential(nn.Linear(poly_feature_dim, 32), nn.ReLU(), nn.Dropout(0.2))
        total = text_embed_dim * 2 + 32
        layers = []
        prev = total
        for hd, dr in zip(hidden_dims, dropout_rates):
            layers += [nn.Linear(prev, hd), nn.ReLU(), nn.Dropout(dr)]
            prev = hd
        layers += [nn.Linear(prev, 3), nn.Sigmoid()]
        self.main_network = nn.Sequential(*layers)

    def forward(self, p_txt, poly, c_txt):
        return self.main_network(torch.cat([p_txt, self.poly_net(poly), c_txt], dim=1)) * 255.0

# ---------------------------------------------------------------------------
# Weight‑loading helper with compatibility shim
# ---------------------------------------------------------------------------

def load_model(weights: str, info_json: str, device: str) -> RGBPredictionModel:
    if os.path.exists(info_json):
        with open(info_json, "r", encoding="utf-8") as f:
            info = json.load(f)
        embed_dim = info.get("embed_dim", 512)
        poly_dim = info.get("poly_feature_dim", 10)
        hidden_dims = info.get("hidden_dims", [256, 128])
        dropout_rates = info.get("dropout_rates", [0.4, 0.3])
        logging.info("Loaded model architecture from info file")
    else:
        logging.warning("Model info file missing – using defaults")
        embed_dim, poly_dim, hidden_dims, dropout_rates = 512, 10, [256, 128], [0.4, 0.3]

    model = RGBPredictionModel(embed_dim, poly_dim, hidden_dims, dropout_rates).to(device)
    state = torch.load(weights, map_location=device)

    # Remap keys if weight file uses new short names
    sample_key = next(iter(state))
    if sample_key.startswith("poly.") or sample_key.startswith("net."):
        logging.info("Remapping weight keys for compatibility with test script")
        remap = {}
        for k, v in state.items():
            if k.startswith("poly."):
                remap[k.replace("poly.", "poly_net.")] = v
            elif k.startswith("net."):
                remap[k.replace("net.", "main_network.")] = v
            else:
                remap[k] = v
        state = remap

    model.load_state_dict(state, strict=False)
    return model

# ---------------------------------------------------------------------------
# Core colour‑prediction helper
# ---------------------------------------------------------------------------

def get_color_prediction(parent_desc: str, polygon: List[List[float]], child_desc: str, clip_model, rgb_model, device: str) -> Tuple[int, int, int]:
    polygon_ft = process_polygon(polygon).unsqueeze(0).to(device)
    p_tok = clip.tokenize([parent_desc or ""], truncate=True).to(device)
    c_tok = clip.tokenize([child_desc or ""], truncate=True).to(device)
    with torch.no_grad():
        p_emb = clip_model.encode_text(p_tok); p_emb /= p_emb.norm(dim=-1, keepdim=True)
        c_emb = clip_model.encode_text(c_tok); c_emb /= c_emb.norm(dim=-1, keepdim=True)
        rgb = rgb_model(p_emb, polygon_ft, c_emb)[0]
    r, g, b = (int(round(max(0, min(255, c.item())))) for c in rgb)
    return r, g, b

# ---------------------------------------------------------------------------
# Utility: bounding box for canvas size
# ---------------------------------------------------------------------------

def bounding_box(shapes: List[dict]):
    xs, ys = [], []
    for s in shapes:
        for x, y in s["polygon"]:
            xs.append(x); ys.append(y)
    return min(xs), max(xs), min(ys), max(ys)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser("RGB polygon colour prediction")
    ap.add_argument("input_json")
    ap.add_argument("output_image")
    ap.add_argument("--model", default="rgb_model_weights.pth")
    ap.add_argument("--model-info", default="rgb_model_info.json")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    logging.info("Starting test with %s", args.input_json)
    with open(args.input_json, "r", encoding="utf-8") as f:
        shapes = json.load(f)
    min_x, max_x, min_y, max_y = bounding_box(shapes)
    width, height = int(max_x - min_x + 1), int(max_y - min_y + 1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load("ViT-B/32", device=device)
    rgb_model = load_model(args.model, args.model_info, device).eval()

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    id_to_shape = {s["id"]: s for s in shapes}

    # Draw parents first → simple level sort
    def level(s):
        lvl, p = 0, s.get("parent")
        while p is not None:
            lvl += 1; p = id_to_shape.get(p, {}).get("parent")
        return lvl
    for s in sorted(shapes, key=level):
        colour = get_color_prediction(id_to_shape.get(s.get("parent"), {}).get("description", ""), s["polygon"], s.get("description", ""), clip_model, rgb_model, device)
        pts = [(x - min_x, y - min_y) for x, y in s["polygon"]]
        draw.polygon(pts, fill=colour)

    os.makedirs(os.path.dirname(args.output_image), exist_ok=True)
    canvas.save(args.output_image)
    logging.info("Saved image → %s", args.output_image)
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logging.error("Fatal error: %s", e)
        logging.error(traceback.format_exc())
        sys.exit(1)
