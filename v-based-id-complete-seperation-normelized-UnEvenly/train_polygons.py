"""
train_polygons.py (RGB Prediction Version with Complete Separation)

This script trains a model on top of CLIP's text encoder to predict RGB color values 
directly from text descriptions of shapes, with completely separate processing pathways.
"""

import os
import sys
import json
import re
import logging
import math
import random
import glob
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import clip
from PIL import Image, ImageDraw, ImageColor

# Create logs directory if it doesn't exist
logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Set up log file name with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(logs_dir, f"train_rgb_polygons_separate_{timestamp}.log")

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO, 
    format='[%(levelname)s] [%(asctime)s] [%(filename)s:%(lineno)d] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)

def compute_bounding_box(shapes):
    """
    Compute the overall bounding box (min_x, max_x, min_y, max_y) for all shapes.
    """
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    
    for shape in shapes:
        for (x, y) in shape["polygon"]:
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
    
    logging.debug("Computed bounding box: min_x=%s, max_x=%s, min_y=%s, max_y=%s", 
                 min_x, max_x, min_y, max_y)
    return min_x, max_x, min_y, max_y

def process_polygon(polygon, normalize=True):
    """
    Process polygon coordinates into a fixed-length feature vector.
    
    Args:
        polygon: List of (x, y) coordinate tuples
        normalize: Whether to normalize coordinates to [0, 1] range
        
    Returns:
        Tensor of processed polygon features
    """
    # Extract features from the polygon
    if not polygon:
        # Default empty polygon features
        return torch.zeros(10)
    
    # Get basic geometric features
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    width = max_x - min_x
    height = max_y - min_y
    
    # Calculate polygon area using the Shoelace formula
    area = 0
    for i in range(len(polygon)):
        j = (i + 1) % len(polygon)
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    area = abs(area) / 2.0
    
    # Calculate polygon centroid
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    
    # Calculate polygon perimeter
    perimeter = 0
    for i in range(len(polygon)):
        j = (i + 1) % len(polygon)
        dx = polygon[j][0] - polygon[i][0]
        dy = polygon[j][1] - polygon[i][1]
        perimeter += math.sqrt(dx*dx + dy*dy)
    
    # Aspect ratio
    aspect_ratio = width / height if height != 0 else 1.0
    
    # Compute distances from centroid to vertices
    distances = []
    for x, y in polygon:
        dx = x - cx
        dy = y - cy
        distances.append(math.sqrt(dx*dx + dy*dy))
    
    avg_distance = sum(distances) / len(distances) if distances else 0
    max_distance = max(distances) if distances else 0
    
    # Create feature vector
    features = [
        width,
        height,
        area,
        perimeter,
        cx,
        cy,
        aspect_ratio,
        avg_distance,
        max_distance,
        len(polygon)  # Number of vertices
    ]
    
    # Normalize features if requested
    if normalize:
        # Create a simple normalization - not optimal but functional
        feature_tensor = torch.tensor(features, dtype=torch.float32)
        non_zero_mask = feature_tensor != 0
        feature_tensor[non_zero_mask] = feature_tensor[non_zero_mask].log1p()  # log(1+x) for positive values
        feature_tensor = torch.tanh(feature_tensor * 0.1)  # Squash to [-1, 1]
        return feature_tensor
    
    return torch.tensor(features, dtype=torch.float32)

class RGBPredictionModel(nn.Module):
    """
    Neural network for predicting RGB values directly from text embeddings 
    and polygon features with completely separate processing pathways.
    """
    def __init__(self, text_embed_dim, poly_feature_dim=10, hidden_dims=[256, 128], 
                output_dim=3, dropout_rates=[0.4, 0.3]):
        super(RGBPredictionModel, self).__init__()
        
        # Validate parameters
        assert len(hidden_dims) == len(dropout_rates), "Must provide dropout rates for each hidden layer"
        
        # Process polygon features with its own complete pathway to RGB
        self.poly_feature_processed_dim = 32
        self.polygon_pathway = nn.Sequential(
            # Initial polygon processing
            nn.Linear(poly_feature_dim, self.poly_feature_processed_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            # First hidden layer
            nn.Linear(self.poly_feature_processed_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rates[0]),
            # Second hidden layer
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rates[1]),
            # Output layer for polygon-based RGB prediction
            nn.Linear(hidden_dims[1], output_dim),
            nn.Sigmoid()
        )
        
        # Parent description pathway to RGB
        self.parent_pathway = nn.Sequential(
            nn.Linear(text_embed_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rates[0]),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rates[1]),
            nn.Linear(hidden_dims[1], output_dim),
            nn.Sigmoid()
        )
        
        # Child description pathway to RGB
        self.child_pathway = nn.Sequential(
            nn.Linear(text_embed_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rates[0]),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rates[1]),
            nn.Linear(hidden_dims[1], output_dim),
            nn.Sigmoid()
        )
        
        # Create learnable weights for combining the three RGB predictions
        # Initialize combination weights with a strong bias toward the child description pathway.
        # Softmax([1.0, 1.0, 4.0]) gives ~87% weight to child_rgb while allowing others to contribute slightly.
        self.combination_weights = nn.Parameter(torch.tensor([1.0, 1.0, 4.0], dtype=torch.float32))

        
        logging.info(f"Model created with completely separate pathways for parent, polygon, and child")
        logging.info(f"Each pathway independently predicts RGB values that are then weighted and combined")
    
    def forward(self, parent_embed, child_polygon, child_embed):
        """
        Forward pass with three completely separate pathways, each producing its own RGB prediction.
        The final output is a weighted combination of these predictions.
        
        Args:
            parent_embed: Parent description embedding tensor
            child_polygon: Child polygon feature tensor  
            child_embed: Child description embedding tensor
            
        Returns:
            RGB values in range 0-255
        """
        # Process each input through its own complete pathway to get RGB predictions
        parent_rgb = self.parent_pathway(parent_embed)
        polygon_rgb = self.polygon_pathway(child_polygon)
        child_rgb = self.child_pathway(child_embed)
        
        # Create a normalized version of the weights that sum to 1
        normalized_weights = torch.softmax(self.combination_weights, dim=0)
        
        # Calculate weighted sum of the RGB predictions without using cat
        # Multiply each RGB prediction by its weight and sum
        weighted_rgb = (
            parent_rgb * normalized_weights[0].view(1, 1) + 
            polygon_rgb * normalized_weights[1].view(1, 1) + 
            child_rgb * normalized_weights[2].view(1, 1)
        )
        
        # Scale to 0-255 range
        rgb_values = weighted_rgb * 255.0
        return rgb_values

class RGBShapesDataset(data.Dataset):
    """
    Dataset for training RGB color prediction with parent-child relationships.
    Generates consistent RGB values for all shapes rather than extracting explicit colors.
    """
    def __init__(self, shapes, id_to_shape):
        self.samples = []
        
        # For each shape, generate deterministic RGB values
        for shape in shapes:
            shape_id = shape.get("id")
            child_desc = shape.get("description", "")
            child_polygon = shape.get("polygon", [])
            
            # Process polygon into feature vector
            polygon_features = process_polygon(child_polygon)
            
            # Get parent description if exists
            parent_id = shape.get("parent")
            parent_desc = ""
            if parent_id is not None and parent_id in id_to_shape:
                parent_desc = id_to_shape[parent_id].get("description", "")
            
            # Generate deterministic RGB based on shape properties
            # Use string hash of shape ID for reproducibility
            hash_base = hash(str(shape_id)) % 16777216  # 2^24
            r = (hash_base >> 16) & 255
            g = (hash_base >> 8) & 255
            b = hash_base & 255
            rgb_color = (r, g, b)
            
            # Store RGB values as a list of floats (normalized to 0-1)
            normalized_rgb = [c / 255.0 for c in rgb_color]
            
            # Add to samples as (parent_desc, polygon_features, child_desc, normalized_rgb)
            self.samples.append((parent_desc, polygon_features, child_desc, normalized_rgb))
            
        logging.info(f"Constructed dataset with {len(self.samples)} total samples")
        logging.info(f"All samples use algorithmically generated RGB values (no extracted colors)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        parent_desc, polygon_features, child_desc, rgb_values = self.samples[idx]
        # Return RGB values as a tensor directly
        return parent_desc, polygon_features, child_desc, torch.tensor(rgb_values, dtype=torch.float32)

def rgb_distance(pred_rgb, true_rgb):
    """
    Calculate Euclidean distance between predicted and true RGB values.
    Input tensors are expected to be in 0-255 range.
    Handles potential batch size differences.
    """
    # Ensure we're operating on the correct dimension for batched tensors
    if pred_rgb.dim() > 1 and true_rgb.dim() > 1:
        return torch.sqrt(torch.sum((pred_rgb - true_rgb) ** 2, dim=1))
    else:
        # Handle non-batched inputs (single sample case)
        return torch.sqrt(torch.sum((pred_rgb - true_rgb) ** 2))

def find_json_files(directory):
    """Find all JSON files in a directory structure"""
    json_pattern = os.path.join(directory, "**", "*.json")
    json_files = glob.glob(json_pattern, recursive=True)
    return json_files

def train_model(dataset, clip_model, rgb_model, optimizer, device, epochs=50, batch_size=16):
    """
    Train the RGB prediction model for exactly 50 epochs, logging detailed metrics.
    No early stopping to allow intentional overfitting.
    """
    # Create train/validation split
    dataset_size = len(dataset)
    val_size = int(dataset_size * 0.2)  # 20% for validation
    train_size = dataset_size - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42))  # Fixed seed for reproducibility
    
    # Create data loaders
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Set CLIP model to evaluation mode (freezing it during training)
    clip_model.eval()
    
    # Mean Squared Error loss for RGB values
    loss_fn = nn.MSELoss()
    
    # Threshold for RGB "accuracy" - consider prediction correct if within this distance
    rgb_accuracy_threshold = 30.0  # Euclidean distance in RGB space
    
    logging.info(f"Starting 50-epoch training with batch size {batch_size}, intentionally allowing overfitting")
    logging.info(f"Training on {train_size} samples, validating on {val_size} samples")
    
    # Log the initial combination weights to see how they evolve during training
    with torch.no_grad():
        weights = torch.softmax(rgb_model.combination_weights, dim=0)
        logging.info(f"Initial pathway weights: Parent={weights[0].item():.4f}, Polygon={weights[1].item():.4f}, Child={weights[2].item():.4f}")
    
    for epoch in range(epochs):
        # Training phase
        rgb_model.train()
        total_train_loss = 0.0
        train_rgb_distances = []
        train_correct = 0
        train_total = 0
        
        for (parent_texts, polygon_features, child_texts, true_rgbs) in train_loader:
            # Move tensors to device
            polygon_features = polygon_features.to(device)
            true_rgbs = true_rgbs.to(device)
            
            parent_tokens = clip.tokenize(parent_texts, truncate=True).to(device)
            child_tokens = clip.tokenize(child_texts, truncate=True).to(device)
            
            optimizer.zero_grad()
            
            # Forward pass through CLIP text encoder (no gradients)
            with torch.no_grad():
                parent_feats = clip_model.encode_text(parent_tokens)
                parent_feats /= parent_feats.norm(dim=-1, keepdim=True)
                
                child_feats = clip_model.encode_text(child_tokens)
                child_feats /= child_feats.norm(dim=-1, keepdim=True)
            
            # Forward pass through our RGB model with three separate pathways
            pred_rgbs = rgb_model(parent_feats, polygon_features, child_feats)
            loss = loss_fn(pred_rgbs, true_rgbs * 255.0)  # Scale true values to 0-255
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rgb_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Calculate RGB distance and accuracy
            distances = rgb_distance(pred_rgbs, true_rgbs * 255.0)
            train_rgb_distances.extend(distances.cpu().detach().numpy())
            
            # Count as correct if RGB distance is below threshold
            correct = (distances < rgb_accuracy_threshold).sum().item()
            train_correct += correct
            train_total += true_rgbs.size(0)
            
            # Track metrics
            total_train_loss += loss.item() * true_rgbs.size(0)
        
        # Validation phase
        rgb_model.eval()
        total_val_loss = 0.0
        val_rgb_distances = []
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for (parent_texts, polygon_features, child_texts, true_rgbs) in val_loader:
                # Move tensors to device
                polygon_features = polygon_features.to(device)
                true_rgbs = true_rgbs.to(device)
                
                parent_tokens = clip.tokenize(parent_texts, truncate=True).to(device)
                child_tokens = clip.tokenize(child_texts, truncate=True).to(device)
                
                parent_feats = clip_model.encode_text(parent_tokens)
                parent_feats /= parent_feats.norm(dim=-1, keepdim=True)
                
                child_feats = clip_model.encode_text(child_tokens)
                child_feats /= child_feats.norm(dim=-1, keepdim=True)
                
                # Forward pass through RGB model with three separate pathways
                pred_rgbs = rgb_model(parent_feats, polygon_features, child_feats)
                loss = loss_fn(pred_rgbs, true_rgbs * 255.0)
                
                # Calculate RGB distance and accuracy
                distances = rgb_distance(pred_rgbs, true_rgbs * 255.0)
                val_rgb_distances.extend(distances.cpu().numpy())
                
                # Count as correct if RGB distance is below threshold
                correct = (distances < rgb_accuracy_threshold).sum().item()
                val_correct += correct
                val_total += true_rgbs.size(0)
                
                total_val_loss += loss.item() * true_rgbs.size(0)
        
        # Calculate epoch metrics
        avg_train_loss = total_train_loss / train_total
        avg_val_loss = total_val_loss / val_total
        
        avg_train_rgb_distance = np.mean(train_rgb_distances)
        avg_val_rgb_distance = np.mean(val_rgb_distances)
        
        train_accuracy = 100 * train_correct / train_total
        val_accuracy = 100 * val_correct / val_total
        
        # Calculate overfitting metric (difference between train and val accuracy)
        overfitting = train_accuracy - val_accuracy
        
        # Log the current combination weights to see how they evolve during training
        weights = torch.softmax(rgb_model.combination_weights, dim=0)
        logging.info(f"Current pathway weights: Parent={weights[0].item():.4f}, Polygon={weights[1].item():.4f}, Child={weights[2].item():.4f}")
        
        # Log metrics in exact required format
        logging.info(f"Epoch {epoch+1}/{epochs}: Train Loss = {avg_train_loss:.4f}, "
                    f"Val Loss = {avg_val_loss:.4f}, Avg RGB Distance = {avg_val_rgb_distance:.2f}, "
                    f"Accuracy: Train = {train_accuracy:.2f}%, Val = {val_accuracy:.2f}%, "
                    f"Overfitting = {overfitting:.2f}%")
    
    # Final log of learned weights
    with torch.no_grad():
        weights = torch.softmax(rgb_model.combination_weights, dim=0)
        logging.info(f"Final learned pathway weights: Parent={weights[0].item():.4f}, Polygon={weights[1].item():.4f}, Child={weights[2].item():.4f}")
    
    return rgb_model

def main():
    logging.info("Starting RGB color prediction model training with completely separate pathways")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    shapes_dir = os.path.join(script_dir, "shapes_jsons")
    if not os.path.exists(shapes_dir):
        logging.warning(f"shapes_jsons directory not found at {shapes_dir}")
        shapes_dir = script_dir  # fallback
    
    json_files = find_json_files(shapes_dir)
    if not json_files:
        json_files = find_json_files(script_dir)
    if not json_files:
        logging.error("No JSON files found for training!")
        sys.exit(1)
    
    logging.info(f"Found {len(json_files)} JSON files for training")
    for file in json_files:
        logging.info(f"  - {os.path.basename(file)}")
    
    # Load all shapes from JSON files
    all_shapes = []
    for json_path in json_files:
        try:
            logging.info(f"Loading shapes data from: {json_path}")
            with open(json_path, "r", encoding="utf-8") as f:
                shapes = json.load(f)
                logging.info(f"Loaded {len(shapes)} shapes from {os.path.basename(json_path)}")
                all_shapes.extend(shapes)
        except Exception as e:
            logging.error(f"Error loading {json_path}: {e}")
            continue
    
    if not all_shapes:
        logging.error("No shapes loaded from JSON files!")
        sys.exit(1)
        
    logging.info(f"Total shapes loaded: {len(all_shapes)}")
    id_to_shape = {shape["id"]: shape for shape in all_shapes}
    
    # Calculate canvas dimensions
    min_x, max_x, min_y, max_y = compute_bounding_box(all_shapes)
    width, height = int(max_x - min_x + 1), int(max_y - min_y + 1)
    logging.info("Canvas size (for reference): width=%d, height=%d", width, height)
    
    # Set up device (GPU if available, otherwise CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("Using device: %s", device)
    
    # Load CLIP model
    logging.info("Loading CLIP model...")
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    logging.info("CLIP model loaded successfully.")
    
    # Construct dataset with ALL shapes (using algorithmically generated RGB values)
    dataset = RGBShapesDataset(all_shapes, id_to_shape)
    
    # Determine embedding dimension from CLIP
    with torch.no_grad():
        sample_tokens = clip.tokenize(["sample text"], truncate=True).to(device)
        sample_embedding = clip_model.encode_text(sample_tokens)
    embed_dim = sample_embedding.shape[-1]
    logging.info("CLIP text embedding dimension: %d", embed_dim)
    
    # Create RGB prediction model with completely separate pathways
    rgb_model = RGBPredictionModel(
        text_embed_dim=embed_dim,
        poly_feature_dim=10,  # Matches dimension from process_polygon function
        hidden_dims=[256, 128],
        output_dim=3,  # RGB values
        dropout_rates=[0.4, 0.3]
    ).to(device)
    logging.info(f"Created RGB prediction model with three completely separate pathways that each predict RGB values")
    
    # Configure optimizer with weight decay for regularization
    optimizer = optim.AdamW(
        rgb_model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    logging.info("Using AdamW optimizer with learning rate 1e-3 and weight decay 1e-4")
    
    # Train the model for exactly 50 epochs with no early stopping
    rgb_model = train_model(
        dataset=dataset,
        clip_model=clip_model,
        rgb_model=rgb_model,
        optimizer=optimizer,
        device=device,
        epochs=50,
        batch_size=16
    )
    logging.info("Training complete after 50 epochs (intentionally allowing overfitting)")
    
    # Save model weights
    weights_path = os.path.join(script_dir, "rgb_model_separate_weights.pth")
    torch.save(rgb_model.state_dict(), weights_path)
    logging.info("Saved separate pathways RGB model weights to: %s", weights_path)
    
    # Save model architecture information for testing
    model_info = {
        "embed_dim": embed_dim,
        "poly_feature_dim": 10,
        "hidden_dims": [256, 128],
        "output_dim": 3,
        "dropout_rates": [0.4, 0.3],
        "model_type": "separate_pathways"
    }
    info_path = os.path.join(script_dir, "rgb_model_separate_info.json")
    with open(info_path, "w") as f:
        json.dump(model_info, f)
    logging.info("Saved model architecture information to: %s", info_path)

if __name__ == "__main__":
    main()