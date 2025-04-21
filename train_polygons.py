# =========================
# train_polygons_rgb_from_json.py
# =========================

import os, json, glob, datetime, logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import clip
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# ---------------- Logging ------------------
logs_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(logs_dir, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(logs_dir, f"train_from_json_rgb_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] [%(asctime)s] %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler(log_file, mode='w')]
)

# ---------------- Dataset ------------------
class CLIPColorDataset(data.Dataset):
    def __init__(self, shapes, clip_model, device):
        self.samples = []
        for shape in shapes:
            desc = shape.get("description", "")
            rgb = shape.get("rgb")
            if not rgb:
                continue  # skip shapes without true RGB

            tokens = clip.tokenize([desc]).to(device)
            with torch.no_grad():
                embed = clip_model.encode_text(tokens)[0]
                embed = embed / embed.norm()

            rgb_tensor = torch.tensor(rgb, dtype=torch.float32) / 255.0
            self.samples.append((embed, rgb_tensor))

        logging.info(f"Dataset size: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# ---------------- Model ------------------
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

# ---------------- Training ------------------
def train(model, dataset, device, epochs=50, batch_size=16):
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for embeds, targets in loader:
            embeds = embeds.to(device)
            targets = targets.to(device)
            preds = model(embeds)
            loss = loss_fn(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * embeds.size(0)

        avg_loss = total_loss / len(dataset)
        logging.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        scheduler.step()

    torch.save(model.state_dict(), "clip_to_rgb_model.pth")
    logging.info("Model saved as clip_to_rgb_model.pth")

# ---------------- Entry ------------------
def main():
    json_dir = os.path.join(os.path.dirname(__file__), "shapes_jsons")
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    shapes = []
    for fpath in json_files:
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                shapes.extend(json.load(f))
        except Exception as e:
            logging.warning(f"Error loading {fpath}: {e}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load("ViT-B/32", device=device)

    dataset = CLIPColorDataset(shapes, clip_model, device)
    embed_dim = clip_model.encode_text(clip.tokenize(["test"]).to(device)).shape[-1]

    model = CLIPToRGB(embed_dim).to(device)
    train(model, dataset, device)

    with open("clip_to_rgb_model_info.json", "w") as f:
        json.dump({"embed_dim": embed_dim}, f)

if __name__ == "__main__":
    main()
