"""
test_polygons.py (RGB Prediction Version)

This script uses a trained model to predict RGB colors for polygons from text descriptions,
handling parent-child relationships without relying on predefined color logic.
"""

import os
import sys
import json
import re
import logging
import math
import argparse
import traceback
import datetime

import torch
import torch.nn as nn
import clip
from PIL import Image, ImageDraw

# Create logs directory if it doesn't exist
logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Set up log file name with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(logs_dir, f"test_rgb_polygons_{timestamp}.log")

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] [%(asctime)s] [%(filename)s:%(lineno)d] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)

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
    and polygon features.
    """
    def __init__(self, text_embed_dim, poly_feature_dim=10, hidden_dims=[256, 128], 
                output_dim=3, dropout_rates=[0.4, 0.3]):
        super(RGBPredictionModel, self).__init__()
        
        # Validate parameters
        assert len(hidden_dims) == len(dropout_rates), "Must provide dropout rates for each hidden layer"
        
        # Process polygon features to a fixed intermediate dimension
        self.poly_feature_processed_dim = 32
        
        # Create polygon feature processing network
        self.poly_net = nn.Sequential(
            nn.Linear(poly_feature_dim, self.poly_feature_processed_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Total input dimension is: parent_embed + processed_poly_features + child_embed
        total_input_dim = (text_embed_dim * 2) + self.poly_feature_processed_dim
        
        # Create main network layers
        layers = []
        prev_dim = total_input_dim
        
        # Log the input dimension for debugging
        logging.debug(f"Model input dimension: {prev_dim} = 2*{text_embed_dim} (text) + {self.poly_feature_processed_dim} (polygon)")
        
        # Create hidden layers with dropout
        for i, (hidden_dim, dropout_rate) in enumerate(zip(hidden_dims, dropout_rates)):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Final output layer for RGB values
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())  # Output between 0 and 1, will scale to 0-255 later
        
        self.main_network = nn.Sequential(*layers)
    
    def forward(self, parent_embed, child_polygon, child_embed):
        """
        Forward pass with three inputs: parent description, child polygon, and child description
        
        Args:
            parent_embed: Parent description embedding tensor
            child_polygon: Child polygon feature tensor  
            child_embed: Child description embedding tensor
            
        Returns:
            RGB values in range 0-255
        """
        # Process polygon features through polygon network
        poly_processed = self.poly_net(child_polygon)
        
        # Concatenate all inputs
        combined = torch.cat([parent_embed, poly_processed, child_embed], dim=1)
        
        # Forward through main network
        raw_output = self.main_network(combined)
        
        # Scale to 0-255 range
        rgb_values = raw_output * 255.0
        return rgb_values

def get_color_prediction(parent_description, child_polygon, child_description, clip_model, rgb_model, device):
    """
    Predict RGB color for a polygon based on parent and child descriptions.
    
    Args:
        parent_description (str): Text description of the parent shape
        child_polygon (list): Polygon coordinates of the child shape
        child_description (str): Text description of the child shape
        clip_model: CLIP text encoder model
        rgb_model: RGB prediction model
        device: Device to perform computation on
        
    Returns:
        tuple: (R, G, B) tuple with integer values from 0-255
    """
    # Use empty string if parent_description is None
    if parent_description is None:
        parent_description = ""
        
    if child_description is None:
        child_description = ""
    
    # Process the polygon features
    polygon_features = process_polygon(child_polygon)
    polygon_features = polygon_features.unsqueeze(0).to(device)  # Add batch dimension
    
    # Generate text embeddings with CLIP
    parent_tokens = clip.tokenize([parent_description], truncate=True).to(device)
    child_tokens = clip.tokenize([child_description], truncate=True).to(device)
    
    with torch.no_grad():
        parent_feats = clip_model.encode_text(parent_tokens)
        parent_feats /= parent_feats.norm(dim=-1, keepdim=True)
        
        child_feats = clip_model.encode_text(child_tokens)
        child_feats /= child_feats.norm(dim=-1, keepdim=True)
        
        # Run through RGB model to get RGB prediction (using all three inputs)
        pred_rgb = rgb_model(parent_feats, polygon_features, child_feats)[0]
    
    # Convert to integers in 0-255 range
    r, g, b = [max(0, min(255, int(round(c.item())))) for c in pred_rgb]
    
    logging.debug(f"Predicted RGB: ({r}, {g}, {b}) for text: '{child_description}' with parent: '{parent_description}'")
    return (r, g, b)

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

def load_model(model_path, model_info_path, device):
    """
    Loads the trained model architecture and weights.
    """
    # Attempt to load model architecture information
    try:
        if os.path.exists(model_info_path):
            with open(model_info_path, 'r') as f:
                model_info = json.load(f)
                
            embed_dim = model_info.get('embed_dim', 512)
            poly_feature_dim = model_info.get('poly_feature_dim', 10)
            hidden_dims = model_info.get('hidden_dims', [256, 128])
            output_dim = model_info.get('output_dim', 3)
            dropout_rates = model_info.get('dropout_rates', [0.4, 0.3])
            
            logging.info("Loaded model architecture from info file")
        else:
            # Default architecture if info file not found
            logging.warning("Model info file not found. Using default architecture.")
            embed_dim = 512  # Standard CLIP embedding dim
            poly_feature_dim = 10
            hidden_dims = [256, 128]
            output_dim = 3
            dropout_rates = [0.4, 0.3]
    except Exception as e:
        logging.error(f"Error loading model info: {e}")
        logging.warning("Using default model architecture")
        embed_dim = 512
        poly_feature_dim = 10
        hidden_dims = [256, 128]
        output_dim = 3
        dropout_rates = [0.4, 0.3]
    
    # Create model with the appropriate architecture
    model = RGBPredictionModel(
        text_embed_dim=embed_dim,
        poly_feature_dim=poly_feature_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        dropout_rates=dropout_rates
    ).to(device)
    
    # Load weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        logging.info(f"Successfully loaded model weights from {model_path}")
    except Exception as e:
        logging.error(f"Error loading model weights: {e}")
        raise
    
    return model

def main():
    parser = argparse.ArgumentParser(description="RGB polygon color prediction testing")
    parser.add_argument("input_json", help="Input JSON file containing polygon data")
    parser.add_argument("output_image", help="Output image file path")
    parser.add_argument("--model", default="rgb_model_weights.pth", 
                       help="Path to trained model weights")
    parser.add_argument("--model-info", default="rgb_model_info.json",
                       help="Path to model architecture information")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug-level logging")
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logging.info("Starting RGB prediction test with arguments: %s", args)
    
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if not script_dir:
            script_dir = "."
        logging.debug("Script directory: %s", script_dir)
        
        # Resolve input JSON path
        json_path = args.input_json
        if not os.path.isabs(json_path):
            json_path = os.path.join(script_dir, json_path)
        logging.debug("JSON path: %s", json_path)
        
        if not os.path.isfile(json_path):
            logging.error("Error: JSON file %s not found.", json_path)
            return 1
            
        # Load JSON data
        with open(json_path, "r", encoding="utf-8") as f:
            shapes = json.load(f)
        logging.info("Loaded %d shapes from %s", len(shapes), args.input_json)
        
        # Log first couple of shapes for inspection
        for i, shape in enumerate(shapes[:2]):
            logging.debug("Shape %d: %s", i, json.dumps(shape, indent=2))
            
        id_to_shape = {s["id"]: s for s in shapes}
        
        # Compute canvas dimensions
        min_x, max_x, min_y, max_y = compute_bounding_box(shapes)
        width, height = int(max_x - min_x + 1), int(max_y - min_y + 1)
        logging.info("Canvas size: width=%d, height=%d", width, height)
        
        # Set up device (GPU if available, otherwise CPU)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info("Using device: %s", device)
        
        # Load CLIP model
        logging.info("Loading CLIP model...")
        try:
            clip_model, preprocess = clip.load("ViT-B/32", device=device)
            logging.info("CLIP model loaded successfully")
        except Exception as e:
            logging.error("Failed to load CLIP model: %s", e)
            logging.error(traceback.format_exc())
            return 1
        
        # Resolve model paths
        weights_path = os.path.join(script_dir, args.model)
        model_info_path = os.path.join(script_dir, args.model_info)
        logging.debug("Model weights path: %s", weights_path)
        logging.debug("Model info path: %s", model_info_path)
        
        if not os.path.isfile(weights_path):
            logging.error("Error: Trained model weights %s not found.", weights_path)
            return 1

        # Load the model
        try:
            rgb_model = load_model(weights_path, model_info_path, device)
            rgb_model.eval()
        except Exception as e:
            logging.error("Failed to load RGB model: %s", e)
            logging.error(traceback.format_exc())
            return 1
        
        # Create image canvas
        logging.info("Creating composite image with dimensions: %dx%d", width, height)
        try:
            composite = Image.new("RGB", (width, height), "white")
            draw = ImageDraw.Draw(composite)
        except Exception as e:
            logging.error("Failed to create image: %s", e)
            logging.error(traceback.format_exc())
            return 1

        # Compute hierarchy levels for ordered drawing
        shapes_by_level = {}
        for shape in shapes:
            level = 0
            current_id = shape.get("parent")
            while current_id is not None:
                level += 1
                parent = id_to_shape.get(current_id)
                if parent:
                    current_id = parent.get("parent")
                else:
                    break
            shapes_by_level.setdefault(level, []).append(shape)
        
        # Log hierarchy information
        for level, level_shapes in shapes_by_level.items():
            logging.debug("Level %d: %d shapes; IDs: %s", level, len(level_shapes), 
                         [s.get("id") for s in level_shapes])
        
        max_level = max(shapes_by_level.keys()) if shapes_by_level else 0
        logging.info("Maximum hierarchy level: %d", max_level)
                
        # Process and draw each shape by hierarchy level (parent shapes first)
        for level in range(max_level + 1):
            level_shapes = shapes_by_level.get(level, [])
            logging.info("Processing %d shapes at level %d", len(level_shapes), level)
            
            for shape in level_shapes:
                shape_id = shape.get("id")
                child_desc = shape.get("description", "")
                child_polygon = shape.get("polygon", [])
                logging.debug("Processing shape ID %s", shape_id)
                
                # Get parent description
                parent_id = shape.get("parent")
                parent_desc = ""
                if parent_id is not None and parent_id in id_to_shape:
                    parent_desc = id_to_shape[parent_id].get("description", "")
                
                # Always get color prediction from model, regardless of explicit colors in description
                predicted_color = get_color_prediction(
                    parent_desc, 
                    child_polygon,
                    child_desc, 
                    clip_model, 
                    rgb_model, 
                    device
                )
                
                # Draw the polygon
                polygon_points = shape["polygon"]
                offset_points = [(x - min_x, y - min_y) for (x, y) in polygon_points]
                logging.debug("Drawing polygon ID %s with color %s", shape_id, predicted_color)
                
                try:
                    draw.polygon(offset_points, fill=predicted_color)
                    logging.debug("Successfully drew polygon ID %s", shape_id)
                except Exception as e:
                    logging.error("Error drawing polygon ID %s: %s", shape_id, e)
                    logging.error("Polygon points: %s", offset_points)
                    continue

        # Save the final image
        output_path = args.output_image
        if not os.path.isabs(output_path):
            output_path = os.path.join(script_dir, output_path)
            
        # Create output directory if needed
        results_dir = os.path.dirname(output_path)
        if results_dir and not os.path.exists(results_dir):
            logging.debug("Creating results directory: %s", results_dir)
            os.makedirs(results_dir, exist_ok=True)
            
        logging.info("Saving image to: %s", output_path)
        try:
            composite.save(output_path)
            logging.info("Final composite image saved to: %s", output_path)
        except Exception as e:
            logging.error("Error saving image: %s", e)
            logging.error(traceback.format_exc())
            return 1
            
        return 0
        
    except Exception as e:
        logging.error("Unexpected error: %s", e)
        logging.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit_code = main()
    logging.info("Script completed with exit code: %s", exit_code)
    sys.exit(exit_code)