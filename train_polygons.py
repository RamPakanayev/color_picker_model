# =========================
# train_polygons_rgb_with_color_feedback.py
# =========================

import os, json, glob, datetime, logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import clip
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from collections import defaultdict

# ---------------- Logging ------------------
logs_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(logs_dir, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(logs_dir, f"train_color_feedback_rgb_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] [%(asctime)s] %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler(log_file, mode='w')]
)

# ---------------- Dataset ------------------
class ColorFeedbackCLIPDataset(data.Dataset):
    def __init__(self, all_shapes, clip_model, device, context_weight=0.3, color_weight=0.1):
        """
        Creates a dataset with contextual information including sibling context and color feedback.
        
        Args:
            all_shapes: List of shapes from all JSON files
            clip_model: CLIP model for text encoding
            device: Device to use (cuda/cpu)
            context_weight: Weight of context in the combined embedding (0-1)
            color_weight: Weight of color information in updated context (0-1)
        """
        self.clip_model = clip_model
        self.device = device
        self.context_weight = context_weight
        self.color_weight = color_weight
        self.samples = []
        self.embed_dim = clip_model.encode_text(clip.tokenize(["test"]).to(device)).shape[-1]
        
        # Group shapes by JSON file to maintain hierarchy within each scene
        shapes_by_file = self._group_shapes_by_file(all_shapes)
        
        # Build and initialize the model we'll use for color prediction during context building
        self.color_model = self._build_color_model()
        
        # Process each file separately to maintain proper context
        for file_name, shapes in shapes_by_file.items():
            logging.info(f"Processing file: {file_name} with {len(shapes)} shapes")
            file_samples = self._process_file_shapes(shapes)
            self.samples.extend(file_samples)
            
        logging.info(f"Total dataset size: {len(self.samples)}")

    def _build_color_model(self):
        """Creates a simple model for initial color predictions"""
        model = nn.Sequential(
            nn.Linear(self.embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid()
        ).to(self.device)
        return model

    def _group_shapes_by_file(self, all_shapes):
        """Group shapes by their source file (if available)"""
        shapes_by_file = defaultdict(list)
        
        for shape in all_shapes:
            file_name = shape.get("_source_file", "unknown")
            shapes_by_file[file_name].append(shape)
            
        return shapes_by_file

    def _process_file_shapes(self, shapes):
        """Process shapes from a single file to build context and create samples"""
        samples = []
        
        # Create a mapping of shape IDs for easy lookup
        shapes_by_id = {shape.get("id"): shape for shape in shapes}
        
        # Initialize context map with empty context for null parent
        contexts = {None: torch.zeros(self.embed_dim, device=self.device)}
        
        # Initialize color context (RGB values for each shape ID)
        color_contexts = {None: torch.zeros(3, device=self.device)}
        
        # Track the last processed sibling for each parent
        last_sibling_by_parent = {}
        
        # Sort shapes by parent-child relationship
        sorted_shapes = self._sort_shapes_by_hierarchy(shapes)
        
        # Log the processing order
        shape_order = [f"{s.get('id')}:{s.get('description', '')[:20]}..." for s in sorted_shapes]
        logging.info(f"Processing shapes in order: {shape_order}")
        
        # Process each shape in the sorted order
        for shape in sorted_shapes:
            shape_id = shape.get("id")
            desc = shape.get("description", "")
            rgb = shape.get("rgb")
            parent_id = shape.get("parent")
            
            if not rgb:
                logging.info(f"Skipping shape {shape_id} without RGB value")
                continue
            
            # Determine context to use (sibling context if available, otherwise parent context)
            if parent_id in last_sibling_by_parent:
                # Use the most recent sibling's context
                last_sibling_id = last_sibling_by_parent[parent_id]
                context_to_use = contexts[last_sibling_id]
                color_context_to_use = color_contexts[last_sibling_id]
                context_source = f"sibling {last_sibling_id}"
            else:
                # Fall back to parent context if no siblings have been processed yet
                context_to_use = contexts.get(parent_id, contexts[None])
                color_context_to_use = color_contexts.get(parent_id, color_contexts[None])
                context_source = f"parent {parent_id}"
            
            # Log context source decision
            logging.debug(f"Shape {shape_id} using context from {context_source}")
            
            # Get shape embedding
            tokens = clip.tokenize([desc]).to(self.device)
            with torch.no_grad():
                shape_embed = self.clip_model.encode_text(tokens)[0]
                shape_embed = shape_embed / shape_embed.norm()
            
            # Combine context with shape embedding
            combined_embed = self._combine_context_and_embedding(context_to_use, shape_embed)
            
            # Create target RGB tensor
            rgb_tensor = torch.tensor(rgb, dtype=torch.float32, device=self.device) / 255.0
            
            # Add to samples
            samples.append((combined_embed, rgb_tensor))
            
            # Predict color for context updating (use a simplified model for prediction)
            with torch.no_grad():
                predicted_color = self.color_model(combined_embed.unsqueeze(0)).squeeze(0)
            
            # Update context with embedding and predicted color
            contexts[shape_id] = self._update_context_with_color(
                context_to_use, 
                shape_embed, 
                predicted_color
            )
            
            # Store color context
            color_contexts[shape_id] = predicted_color
            
            # Update last sibling tracker
            last_sibling_by_parent[parent_id] = shape_id
            
            # Log context update
            logging.debug(f"Shape {shape_id} ({desc[:30]}...) - RGB: {rgb_tensor.tolist()}, Predicted: {predicted_color.tolist()}, Context norm: {contexts[shape_id].norm().item():.4f}")
        
        return samples

    def _sort_shapes_by_hierarchy(self, shapes):
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

    def _combine_context_and_embedding(self, context, embed):
        """Combine context vector and shape embedding"""
        # Weighted combination
        w = self.context_weight
        combined = (1 - w) * embed + w * context
        # Normalize the combined vector
        combined = combined / combined.norm()
        return combined

    def _update_context_with_color(self, prev_context, new_embed, color_rgb, embed_weight=0.6, color_weight=None):
        """
        Update context by combining previous context with new embedding and predicted color.
        
        Args:
            prev_context: Previous context vector
            new_embed: Current shape embedding
            color_rgb: Predicted RGB color (normalized to 0-1)
            embed_weight: Weight for the new embedding (default 0.6)
            color_weight: Weight for color information (default from self.color_weight)
        """
        if color_weight is None:
            color_weight = self.color_weight
            
        # Ensure weights sum to 1
        context_weight = 1.0 - embed_weight - color_weight
        
        # Create a color embedding by expanding color to embedding dimensions
        color_embed = self._expand_color_to_embedding(color_rgb)
        
        # Combine all components
        updated = (
            embed_weight * new_embed + 
            color_weight * color_embed + 
            context_weight * prev_context
        )
        
        # Normalize the updated context
        updated = updated / (updated.norm() + 1e-8)  # Add small epsilon to avoid division by zero
        return updated
    
    def _expand_color_to_embedding(self, color_rgb):
        """
        Expand RGB color to embedding dimensions for context integration.
        This creates a simple color representation in embedding space.
        """
        # Create a repeating pattern of the color values to match embedding dimension
        repeat_factor = self.embed_dim // 3
        
        # Unfold the color
        expanded = torch.cat([
            color_rgb[0].repeat(repeat_factor),
            color_rgb[1].repeat(repeat_factor),
            color_rgb[2].repeat(repeat_factor),
        ])
        
        # Pad if needed
        padding = self.embed_dim - expanded.shape[0]
        if padding > 0:
            expanded = torch.cat([expanded, expanded[:padding]])
        
        # Normalize the expanded color vector
        expanded = expanded / (expanded.norm() + 1e-8)
        return expanded

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# ---------------- Model ------------------
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

# ---------------- Training ------------------
def train(model, dataset, device, epochs=200, batch_size=16):
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    loss_fn = nn.MSELoss()

    logging.info(f"Starting training: {epochs} epochs, batch size {batch_size}")
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_count = 0

        for embeds, targets in loader:
            embeds = embeds.to(device)
            targets = targets.to(device)
            preds = model(embeds)
            loss = loss_fn(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * embeds.size(0)
            batch_count += 1
            
            if batch_count % 10 == 0:
                logging.debug(f"Epoch {epoch+1}, Batch {batch_count}: Loss {loss.item():.4f}")

        avg_loss = total_loss / len(dataset)
        logging.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        scheduler.step()
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "clip_to_rgb_model_best.pth")
            logging.info(f"Saved best model with loss: {best_loss:.4f}")

    # Save final model
    torch.save(model.state_dict(), "clip_to_rgb_model.pth")
    logging.info("Final model saved as clip_to_rgb_model.pth")
    logging.info(f"Best model saved as clip_to_rgb_model_best.pth (loss: {best_loss:.4f})")

# ---------------- Entry ------------------
def main():
    # Load all JSON files
    json_dir = os.path.join(os.path.dirname(__file__), "shapes_jsons")
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    
    logging.info(f"Found {len(json_files)} JSON files in {json_dir}")
    
    # Parse shapes from all files
    shapes = []
    for fpath in json_files:
        try:
            file_name = os.path.basename(fpath)
            with open(fpath, 'r', encoding='utf-8') as f:
                file_shapes = json.load(f)
                # Add source file information
                for shape in file_shapes:
                    shape["_source_file"] = file_name
                shapes.extend(file_shapes)
            logging.info(f"Loaded {len(file_shapes)} shapes from {file_name}")
        except Exception as e:
            logging.warning(f"Error loading {fpath}: {e}")

    # Setup device and model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    
    clip_model, _ = clip.load("ViT-B/32", device=device)
    logging.info("CLIP model loaded")

    # Create dataset with color feedback contextual information
    context_weight = 0.3  # Weight of context in the combined embedding
    color_weight = 0.1    # Weight of color in context updates
    dataset = ColorFeedbackCLIPDataset(shapes, clip_model, device, context_weight, color_weight)
    
    # Get embedding dimension from the dataset
    sample_embed, _ = dataset[0]
    embed_dim = sample_embed.shape[0]
    
    # Create and train the model
    model = ContextualCLIPToRGB(embed_dim).to(device)
    logging.info(f"Created model with input dimension: {embed_dim}")
    
    train(model, dataset, device)
    
    # Save model information
    with open("clip_to_rgb_model_info.json", "w") as f:
        model_info = {
            "embed_dim": embed_dim,
            "context_weight": context_weight,
            "color_weight": color_weight,
            "model_type": "color_feedback_contextual"
        }
        json.dump(model_info, f)
        logging.info(f"Saved model info: {model_info}")

if __name__ == "__main__":
    main()