# =========================
# train_polygons_rgb_with_unconstrained_weights.py
# =========================

import os, json, glob, datetime, logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import clip
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from collections import defaultdict

# ---------------- Logging ------------------
logs_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(logs_dir, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(logs_dir, f"train_unconstrained_weights_rgb_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] [%(asctime)s] %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler(log_file, mode='w')]
)

# ---------------- Dataset ------------------
class ContextualCLIPDataset(data.Dataset):
    def __init__(self, all_shapes, clip_model, device):
        """
        Creates a dataset with hierarchical and contextual information for each shape.
        No assumptions are made about which information source is more important.
        
        Args:
            all_shapes: List of shapes from all JSON files
            clip_model: CLIP model for text encoding
            device: Device to use (cuda/cpu)
        """
        self.clip_model = clip_model
        self.device = device
        self.samples = []
        self.embed_dim = clip_model.encode_text(clip.tokenize(["test"]).to(device)).shape[-1]
        
        # Group shapes by JSON file to maintain hierarchy within each scene
        shapes_by_file = self._group_shapes_by_file(all_shapes)
        
        # Process each file separately to maintain proper context
        for file_name, shapes in shapes_by_file.items():
            logging.info(f"Processing file: {file_name} with {len(shapes)} shapes")
            file_samples = self._process_file_shapes(shapes)
            self.samples.extend(file_samples)
            
        logging.info(f"Total dataset size: {len(self.samples)}")

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
        
        # Initialize context maps for hierarchical information
        text_contexts = {None: torch.zeros(self.embed_dim, device=self.device)}
        color_contexts = {None: torch.zeros(3, device=self.device)}
        
        # Track relationships between shapes
        last_sibling_by_parent = {}
        
        # Sort shapes by parent-child relationship
        sorted_shapes = self._sort_shapes_by_hierarchy(shapes)
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
            
            # Get descriptive embedding from CLIP
            tokens = clip.tokenize([desc]).to(self.device)
            with torch.no_grad():
                text_embed = self.clip_model.encode_text(tokens)[0]
                text_embed = text_embed / text_embed.norm()
            
            # Get parent/sibling context
            text_context, color_context = self._get_context(
                shape_id, parent_id, last_sibling_by_parent, 
                text_contexts, color_contexts
            )
            
            # Store RGB target
            rgb_tensor = torch.tensor(rgb, dtype=torch.float32, device=self.device) / 255.0
            
            # Expand color context to embedding dimensions for proper combination
            expanded_color = self._expand_color_to_embedding(color_context)
            
            # Create sample with all information components
            sample = {
                "id": shape_id,
                "description": desc,
                "text_embed": text_embed,
                "text_context": text_context,
                "color_context": expanded_color,
                "target_rgb": rgb_tensor,
                "parent_id": parent_id
            }
            
            samples.append(sample)
            
            # Update contexts for future shapes
            text_contexts[shape_id] = text_embed
            color_contexts[shape_id] = rgb_tensor
            
            # Update sibling tracking
            last_sibling_by_parent[parent_id] = shape_id
            
        return samples
        
    def _get_context(self, shape_id, parent_id, last_sibling_by_parent, text_contexts, color_contexts):
        """Get context information from parent/sibling relationships"""
        # Prefer sibling context if available, otherwise use parent context
        if parent_id in last_sibling_by_parent:
            sibling_id = last_sibling_by_parent[parent_id]
            text_context = text_contexts[sibling_id]
            color_context = color_contexts[sibling_id]
        else:
            # Fall back to parent context
            text_context = text_contexts.get(parent_id, text_contexts[None])
            color_context = color_contexts.get(parent_id, color_contexts[None])
            
        return text_context, color_context

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
    
    def _expand_color_to_embedding(self, color_rgb):
        """
        Expand RGB color to embedding dimensions for proper combination
        with text embeddings
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
        sample = self.samples[idx]
        return (
            sample["text_embed"],     # Text embedding from description
            sample["text_context"],    # Context embedding from related shapes
            sample["color_context"],   # Color context from related shapes  
            sample["target_rgb"],      # Target RGB color
            sample["id"],              # Shape ID for analysis
            sample["description"]      # Description for analysis
        )

# ---------------- Unconstrained Weight Model ------------------
class UnconstrainedWeightModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        
        # Feature extractors for text embedding
        self.text_processor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Weight prediction network - predicts free weights for the components
        self.weight_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # Weights for [text, context, color]
        )
        
        # Color prediction network
        self.color_predictor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Sigmoid()  # RGB values between 0-1
        )
        
    def forward(self, text_embed, text_context=None, color_context=None):
        """
        Forward pass with completely unconstrained weights
        
        Args:
            text_embed: CLIP embedding of the shape description
            text_context: Context embedding from related shapes
            color_context: Color context from related shapes
            
        Returns:
            predicted_rgb: Predicted RGB color
            weights: Weights used for the three components
        """
        # For testing compatibility, handle case with just text embedding
        if text_context is None or color_context is None:
            # Process text embedding directly if no context is provided
            return self.color_predictor(text_embed)
        
        # Process the text embedding to get features for weight prediction
        text_features = self.text_processor(text_embed)
        
        # Predict weights from text features - no constraints whatsoever
        logits = self.weight_predictor(text_features)
        
        # Apply softmax to ensure weights sum to 1 - but no ordering constraints
        weights = torch.nn.functional.softmax(logits, dim=1)
        
        # Extract individual component weights
        text_weight = weights[:, 0].unsqueeze(1)
        context_weight = weights[:, 1].unsqueeze(1)
        color_weight = weights[:, 2].unsqueeze(1)
        
        # Combine all components using the predicted weights
        combined = (
            text_weight * text_embed + 
            context_weight * text_context + 
            color_weight * color_context
        )
        
        # Normalize the combined representation
        combined = combined / (combined.norm(dim=1, keepdim=True) + 1e-8)
        
        # Predict RGB color from the weighted combination
        color = self.color_predictor(combined)
        
        return color, weights

# ---------------- Training Loop ------------------
def train_model(model, dataset, device, epochs=200, batch_size=16, learning_rate=1e-3):
    """Train the model with purely color prediction loss, no constraints on weights"""
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    color_loss_fn = nn.MSELoss()
    
    logging.info(f"Starting training with unconstrained weights: {epochs} epochs, batch size {batch_size}")
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_count = 0
        
        # Track weights for analysis
        all_weights = []
        all_text_weights = []
        all_context_weights = []
        all_color_weights = []
        
        # Track detailed sample info for analysis
        sample_details = []
        
        for text_embeds, text_contexts, color_contexts, targets, shape_ids, descriptions in loader:
            text_embeds = text_embeds.to(device)
            text_contexts = text_contexts.to(device)
            color_contexts = color_contexts.to(device)
            targets = targets.to(device)
            
            # Forward pass - model predicts colors and weights
            color_preds, weights = model(text_embeds, text_contexts, color_contexts)
            
            # Compute loss - ONLY color prediction accuracy, no constraints on weights
            loss = color_loss_fn(color_preds, targets)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item() * text_embeds.size(0)
            batch_count += 1
            
            # Track weights
            all_weights.append(weights.detach())
            
            # Track individual weight components
            all_text_weights.extend(weights[:, 0].detach().cpu().numpy())
            all_context_weights.extend(weights[:, 1].detach().cpu().numpy())
            all_color_weights.extend(weights[:, 2].detach().cpu().numpy())
            
            # Track sample details for analysis
            for i in range(text_embeds.size(0)):
                sample_details.append({
                    'shape_id': shape_ids[i],
                    'description': descriptions[i],
                    'weights': weights[i].detach().cpu().numpy(),
                    'rgb_target': targets[i].detach().cpu().numpy(),
                    'rgb_pred': color_preds[i].detach().cpu().numpy()
                })
            
            if batch_count % 10 == 0:
                logging.debug(f"Epoch {epoch+1}, Batch {batch_count}: Loss {loss.item():.4f}")
        
        # Calculate average metrics
        avg_loss = total_loss / len(dataset)
        
        # Calculate weight statistics
        all_weights_tensor = torch.cat(all_weights, dim=0)
        avg_weights = torch.mean(all_weights_tensor, dim=0)
        std_weights = torch.std(all_weights_tensor, dim=0)
        min_weights, _ = torch.min(all_weights_tensor, dim=0)
        max_weights, _ = torch.max(all_weights_tensor, dim=0)
        
        logging.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        logging.info(f"Avg weights: text={avg_weights[0].item():.3f}, "
                    f"context={avg_weights[1].item():.3f}, color={avg_weights[2].item():.3f}")
        logging.info(f"Std weights: text={std_weights[0].item():.3f}, "
                    f"context={std_weights[1].item():.3f}, color={std_weights[2].item():.3f}")
        logging.info(f"Min weights: text={min_weights[0].item():.3f}, "
                    f"context={min_weights[1].item():.3f}, color={min_weights[2].item():.3f}")
        logging.info(f"Max weights: text={max_weights[0].item():.3f}, "
                    f"context={max_weights[1].item():.3f}, color={max_weights[2].item():.3f}")
        
        # Log detailed weight distribution every 10 epochs
        if epoch % 10 == 0 or epoch == epochs - 1:
            # Calculate weight distribution statistics
            def weight_distribution(weights):
                low = (weights < 0.2).float().mean().item() * 100
                med = ((weights >= 0.2) & (weights <= 0.5)).float().mean().item() * 100
                high = (weights > 0.5).float().mean().item() * 100
                return f"low(<0.2): {low:.1f}%, med(0.2-0.5): {med:.1f}%, high(>0.5): {high:.1f}%"
            
            logging.info(f"Text weight distribution: {weight_distribution(all_weights_tensor[:,0])}")
            logging.info(f"Context weight distribution: {weight_distribution(all_weights_tensor[:,1])}")
            logging.info(f"Color weight distribution: {weight_distribution(all_weights_tensor[:,2])}")
            
            # Analyze common ordering patterns
            text_gt_context = (all_weights_tensor[:,0] > all_weights_tensor[:,1]).float().mean().item() * 100
            context_gt_color = (all_weights_tensor[:,1] > all_weights_tensor[:,2]).float().mean().item() * 100
            text_gt_color = (all_weights_tensor[:,0] > all_weights_tensor[:,2]).float().mean().item() * 100
            
            logging.info(f"Text > Context: {text_gt_context:.2f}%")
            logging.info(f"Context > Color: {context_gt_color:.2f}%")
            logging.info(f"Text > Color: {text_gt_color:.2f}%")
            
            # Different orderings
            text_dominant = ((all_weights_tensor[:,0] > all_weights_tensor[:,1]) & 
                           (all_weights_tensor[:,0] > all_weights_tensor[:,2])).float().mean().item() * 100
            context_dominant = ((all_weights_tensor[:,1] > all_weights_tensor[:,0]) & 
                              (all_weights_tensor[:,1] > all_weights_tensor[:,2])).float().mean().item() * 100
            color_dominant = ((all_weights_tensor[:,2] > all_weights_tensor[:,0]) & 
                            (all_weights_tensor[:,2] > all_weights_tensor[:,1])).float().mean().item() * 100
            
            logging.info(f"Text dominant: {text_dominant:.2f}%")
            logging.info(f"Context dominant: {context_dominant:.2f}%")
            logging.info(f"Color dominant: {color_dominant:.2f}%")
            
            # Log interesting examples
            if len(sample_details) > 0:
                # Text-dominant examples
                text_examples = sorted(sample_details, key=lambda x: x['weights'][0], reverse=True)[:3]
                for ex in text_examples:
                    logging.info(f"High text weight ({ex['weights'][0]:.3f}): Shape {ex['shape_id']} - '{ex['description'][:50]}...'")
                
                # Context-dominant examples
                context_examples = sorted(sample_details, key=lambda x: x['weights'][1], reverse=True)[:3]
                for ex in context_examples:
                    logging.info(f"High context weight ({ex['weights'][1]:.3f}): Shape {ex['shape_id']} - '{ex['description'][:50]}...'")
                
                # Color-dominant examples
                color_examples = sorted(sample_details, key=lambda x: x['weights'][2], reverse=True)[:3]
                for ex in color_examples:
                    logging.info(f"High color weight ({ex['weights'][2]:.3f}): Shape {ex['shape_id']} - '{ex['description'][:50]}...'")
        
        scheduler.step()
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "clip_to_rgb_model_unconstrained_best.pth")
            logging.info(f"Saved best model with loss: {best_loss:.4f}")

    # Save final model
    torch.save(model.state_dict(), "clip_to_rgb_model_unconstrained.pth")
    logging.info("Final model saved as clip_to_rgb_model_unconstrained.pth")
    logging.info(f"Best model saved as clip_to_rgb_model_unconstrained_best.pth (loss: {best_loss:.4f})")

    # Save weight analysis for the final epoch
   # Create weight_analysis directory if it doesn't exist
    weight_analysis_dir = os.path.join(os.path.dirname(__file__), "weight_analysis")
    os.makedirs(weight_analysis_dir, exist_ok=True)
    
    # Save weight analysis for the final epoch
    try:
        weight_analysis_path = os.path.join(weight_analysis_dir, f"train_weight_analysis_{timestamp}.json")
        with open(weight_analysis_path, "w") as f:
            json.dump(sample_details[-100:], f)  # Save last 100 samples
            logging.info(f"Saved weight analysis to {weight_analysis_path}")
    except Exception as e:
        logging.warning(f"Failed to save weight analysis: {e}")

# ---------------- Main Function ------------------
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

    # Create dataset with contextual information
    dataset = ContextualCLIPDataset(shapes, clip_model, device)
    
    # Get embedding dimension from the dataset
    embed_dim = dataset.embed_dim
    
    # Create unconstrained weight model
    model = UnconstrainedWeightModel(input_dim=embed_dim).to(device)
    
    logging.info(f"Created unconstrained weight model with input dimension: {embed_dim}")
    logging.info("Model is free to determine optimal weights for each shape without constraints")
    
    # Train with unconstrained weights
    train_model(
        model=model,
        dataset=dataset,
        device=device,
        epochs=200,
        batch_size=16,
        learning_rate=1e-3
    )
    
    # Save model information
    with open("clip_to_rgb_model_unconstrained_info.json", "w") as f:
        model_info = {
            "embed_dim": embed_dim,
            "model_type": "unconstrained_weight"
        }
        json.dump(model_info, f)
        logging.info(f"Saved model info: {model_info}")

if __name__ == "__main__":
    main()