# =========================
# test_polygons_rgb_with_context.py
# =========================

import os, sys, json, argparse, datetime, logging, traceback
from collections import defaultdict
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
import clip

# ---------- Logging ----------
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
logs_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(logs_dir, exist_ok=True)
log_file = os.path.join(logs_dir, f"test_clip_rgb_context_{timestamp}.log")

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
class ContextualCLIPToRGB(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# ---------- Context Management ----------
def sort_shapes_by_hierarchy(shapes):
    """Sort shapes so parents are processed before their children"""
    # Create a graph representation
    children = defaultdict(list)
    for shape in shapes:
        parent = shape.get("parent")
        children[parent].append(shape)
    
    # Perform BFS traversal
    sorted_shapes = []
    queue = children[None].copy()  # Start with root nodes (no parent)
    
    # If no root nodes found, use all shapes
    if not queue and shapes:
        logging.warning("No root nodes (parent=None) found, using all shapes as roots")
        queue = shapes
        
    while queue:
        node = queue.pop(0)
        sorted_shapes.append(node)
        queue.extend(children[node.get("id")])
    
    return sorted_shapes

def combine_context_and_embedding(context, embed, context_weight=0.3):
    """Combine context vector and shape embedding"""
    # Weighted combination
    w = context_weight
    combined = (1 - w) * embed + w * context
    # Normalize the combined vector
    combined = combined / combined.norm()
    return combined

def update_context(prev_context, new_embed, alpha=0.7):
    """Update context by combining previous context with new embedding"""
    updated = alpha * new_embed + (1 - alpha) * prev_context
    # Normalize the updated context
    updated = updated / (updated.norm() + 1e-8)  # Add small epsilon to avoid division by zero
    return updated

# ---------- Prediction ----------
def predict_rgb_with_context(shape, clip_model, rgb_model, device, contexts, last_sibling_by_parent, context_weight=0.3):
    """Predict RGB color using contextual information"""
    desc = shape.get("description", "")
    shape_id = shape.get("id")
    parent_id = shape.get("parent")
    
    # Determine context to use (sibling context if available, otherwise parent context)
    if parent_id in last_sibling_by_parent:
        # Use the most recent sibling's context
        last_sibling_id = last_sibling_by_parent[parent_id]
        context_to_use = contexts[last_sibling_id]
        context_source = f"sibling {last_sibling_id}"
    else:
        # Fall back to parent context if no siblings have been processed yet
        context_to_use = contexts.get(parent_id, contexts[None])
        context_source = f"parent {parent_id}"
    
    # Log context source decision
    logging.debug(f"Shape {shape_id} using context from {context_source}")
    
    # Get shape embedding
    tokens = clip.tokenize([desc]).to(device)
    with torch.no_grad():
        shape_embed = clip_model.encode_text(tokens)[0]
        shape_embed = shape_embed / shape_embed.norm()
    
    # Combine context with shape embedding
    combined_embed = combine_context_and_embedding(context_to_use, shape_embed, context_weight)
    
    # Predict RGB
    with torch.no_grad():
        pred_rgb = rgb_model(combined_embed.unsqueeze(0)).squeeze(0)
    
    # Update context for this shape
    contexts[shape_id] = update_context(context_to_use, shape_embed)
    
    # Update last sibling tracker
    last_sibling_by_parent[parent_id] = shape_id
    
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
            model_info = json.load(f)
            embed_dim = model_info["embed_dim"]
            context_weight = model_info.get("context_weight", 0.3)
            model_type = model_info.get("model_type", "standard")

        logging.info(f"Using model type: {model_type} with context weight: {context_weight}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, _ = clip.load("ViT-B/32", device=device)
        
        # Use the contextual model architecture that matches training
        model = ContextualCLIPToRGB(embed_dim).to(device)
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

        # Initialize context tracking
        contexts = {None: torch.zeros(embed_dim, device=device)}
        last_sibling_by_parent = {}

        # Sort shapes by hierarchy to maintain proper context flow
        sorted_shapes = sort_shapes_by_hierarchy(shapes)
        
        # Log the processing order
        shape_order = [f"{s.get('id')}:{s.get('description', '')[:20]}..." for s in sorted_shapes]
        logging.info(f"Processing shapes in order: {shape_order}")

        # Process each shape in the sorted order
        for shape in sorted_shapes:
            desc = shape.get("description", "")
            shape_id = shape.get("id")
            polygon = [(x - min_x, y - min_y) for x, y in shape["polygon"]]
            
            # Predict RGB with context
            if model_type in ["contextual", "sibling_contextual"]:
                rgb = predict_rgb_with_context(shape, clip_model, model, device, 
                                              contexts, last_sibling_by_parent, context_weight)
            else:
                # Fall back to original predict method for standard models
                tokens = clip.tokenize([desc]).to(device)
                with torch.no_grad():
                    embed = clip_model.encode_text(tokens)[0]
                    embed = embed / embed.norm()
                    pred_rgb = model(embed.unsqueeze(0)).squeeze(0)
                rgb = tuple((pred_rgb.clamp(0, 1) * 255).round().int().tolist())
            
            logging.info(f"Shape {shape_id} - {desc!r} -> {rgb}")
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