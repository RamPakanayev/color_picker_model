
# =========================
# test_polygons_rgb_from_json.py
# =========================

import os, sys, json, argparse, datetime, logging, traceback
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
import clip

# ---------- Logging ----------
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
logs_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(logs_dir, exist_ok=True)
log_file = os.path.join(logs_dir, f"test_clip_rgb_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] [%(asctime)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode='w', encoding='utf-8')
    ]
)

def close_logging_handlers():
    for handler in logging.getLogger().handlers[:]:
        handler.flush()
        handler.close()
        logging.getLogger().removeHandler(handler)

# ---------- Model ----------
class CLIPToRGB(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# ---------- Prediction ----------
def predict_rgb(text, clip_model, rgb_model, device):
    tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        embed = clip_model.encode_text(tokens)[0]
        embed = embed / embed.norm()
        pred_rgb = rgb_model(embed.unsqueeze(0)).squeeze(0)
    return tuple((pred_rgb.clamp(0, 1) * 255).round().int().tolist())

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json")
    parser.add_argument("output_image")
    parser.add_argument("--model", default="clip_to_rgb_model.pth")
    parser.add_argument("--model-info", default="clip_to_rgb_model_info.json")
    args = parser.parse_args()

    try:
        with open(args.input_json, "r", encoding="utf-8") as f:
            shapes = json.load(f)

        with open(args.model_info, "r") as f:
            embed_dim = json.load(f)["embed_dim"]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, _ = clip.load("ViT-B/32", device=device)
        model = CLIPToRGB(embed_dim).to(device)
        model.load_state_dict(torch.load(args.model, map_location=device))
        model.eval()

        # canvas size
        min_x = min(p[0] for s in shapes for p in s["polygon"])
        max_x = max(p[0] for s in shapes for p in s["polygon"])
        min_y = min(p[1] for s in shapes for p in s["polygon"])
        max_y = max(p[1] for s in shapes for p in s["polygon"])
        width, height = int(max_x - min_x + 1), int(max_y - min_y + 1)

        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)

        for shape in shapes:
            desc = shape.get("description", "")
            polygon = [(x - min_x, y - min_y) for x, y in shape["polygon"]]
            rgb = predict_rgb(desc, clip_model, model, device)
            logging.info(f"{desc!r} -> {rgb}")
            draw.polygon(polygon, fill=rgb)

        os.makedirs(os.path.dirname(args.output_image), exist_ok=True)
        img.save(args.output_image)
        logging.info(f"Saved image to {args.output_image}")

    except Exception as e:
        logging.error(traceback.format_exc())
        sys.exit(1)
    finally:
        close_logging_handlers()

if __name__ == '__main__':
    main()