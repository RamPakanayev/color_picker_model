# =========================
# test_polygons_rgb_with_unconstrained_weights.py
# =========================

import os, sys, json, argparse, datetime, logging, traceback
from collections import defaultdict
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
import clip
import numpy as np

# ---------- Logging ----------
logs_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(logs_dir, exist_ok=True)

def setup_logging(log_file):
    """Set up logging with the specified log file"""
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode='w', encoding='utf-8')
    ]
    
    # Remove any existing handlers to avoid duplicate logging
    for handler in logging.getLogger().handlers[:]:
        logging.getLogger().removeHandler(handler)
    
    # Set up new handlers
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] [%(asctime)s] %(message)s',
        handlers=handlers
    )

def close_logging_handlers():
    for handler in logging.getLogger().handlers[:]:
        handler.flush()
        handler.close()
        logging.getLogger().removeHandler(handler)

# ---------- Unconstrained Weight Model ----------
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

# Legacy models for backward compatibility
class CLIPToRGB(nn.Module):
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

# ---------- Context Handling ----------
def expand_color_to_embedding(color_rgb, embed_dim):
    """Expand RGB color to embedding dimensions"""
    repeat_factor = embed_dim // 3
    
    # Unfold the color
    expanded = torch.cat([
        color_rgb[0].repeat(repeat_factor),
        color_rgb[1].repeat(repeat_factor),
        color_rgb[2].repeat(repeat_factor),
    ])
    
    # Pad if needed
    padding = embed_dim - expanded.shape[0]
    if padding > 0:
        expanded = torch.cat([expanded, expanded[:padding]])
    
    # Normalize the expanded color vector
    expanded = expanded / (expanded.norm() + 1e-8)
    return expanded

# ---------- Prediction ----------
def predict_rgb(text, clip_model, rgb_model, device, model_type, contexts=None, colors=None, embed_dim=None):
    """
    Predict RGB color for a text description
    
    Args:
        text: Text description of the shape
        clip_model: CLIP model for text encoding
        rgb_model: Color prediction model
        device: Device to use
        model_type: Type of model 
        contexts: Dictionary of context vectors by shape ID
        colors: Dictionary of color vectors by shape ID
        embed_dim: Embedding dimension
    
    Returns:
        Tuple of (R, G, B) values (0-255) and weights if applicable
    """
    # Get text embedding from CLIP
    tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_embed = clip_model.encode_text(tokens)[0]
        text_embed = text_embed / text_embed.norm()
        
        # For standard model or no contexts, just use the embedding directly
        if model_type == "standard" or contexts is None:
            pred_rgb = rgb_model(text_embed.unsqueeze(0)).squeeze(0)
            return pred_rgb
        
        # For unconstrained weight model, handle embedding and context separately
        elif model_type == "unconstrained_weight":
            # Get current context
            text_context = contexts.get("current", torch.zeros_like(text_embed))
            
            # Get current color context (if available)
            color_context = colors.get("color_current", torch.zeros(3, device=device))
            
            # Expand color to embedding dimensions
            expanded_color = expand_color_to_embedding(color_context, embed_dim)
            
            # Predict color and weights
            pred_rgb, pred_weights = rgb_model(
                text_embed.unsqueeze(0), 
                text_context.unsqueeze(0),
                expanded_color.unsqueeze(0)
            )
            pred_rgb = pred_rgb.squeeze(0)
            pred_weights = pred_weights.squeeze(0)
            
            logging.debug(f"Predicted weights: {pred_weights.tolist()}")
            return pred_rgb, pred_weights
        
        # For other models (backward compatibility)
        else:
            # Get current context
            context = contexts.get("current", torch.zeros_like(text_embed))
            
            # For flexible_weight or hierarchical_weight models
            if hasattr(rgb_model, 'weight_predictor') or hasattr(rgb_model, 'weight_branch'):
                # Get current color context
                color_context = colors.get("color_current", torch.zeros(3, device=device))
                
                # Expand color to embedding dimensions if needed
                if model_type == "flexible_weight":
                    expanded_color = expand_color_to_embedding(color_context, embed_dim)
                    color_input = expanded_color.unsqueeze(0)
                else:
                    color_input = color_context.unsqueeze(0)
                
                # Predict color and weights
                pred_rgb, pred_weights = rgb_model(
                    text_embed.unsqueeze(0), 
                    context.unsqueeze(0),
                    color_input
                )
                pred_rgb = pred_rgb.squeeze(0)
                pred_weights = pred_weights.squeeze(0)
                
                return pred_rgb, pred_weights
            else:
                # Simple weighted combination (fixed weights)
                combined = 0.7 * text_embed + 0.3 * context
                combined = combined / combined.norm()
                
                # Predict color
                pred_rgb = rgb_model(combined.unsqueeze(0)).squeeze(0)
                return pred_rgb

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json")
    parser.add_argument("output_image")
    parser.add_argument("--model", default="clip_to_rgb_model_unconstrained.pth")
    parser.add_argument("--model-info", default="clip_to_rgb_model_unconstrained_info.json")
    args = parser.parse_args()
    
    # Get the base name of the input JSON without extension
    json_base_name = os.path.splitext(os.path.basename(args.input_json))[0]
    
    # Set up logging with the JSON file name
    log_file = os.path.join(logs_dir, f"{json_base_name}_test.log")
    setup_logging(log_file)

    try:
        with open(args.input_json, "r", encoding="utf-8") as f:
            shapes = json.load(f)

        with open(args.model_info, "r") as f:
            model_info = json.load(f)
            embed_dim = model_info["embed_dim"]
            model_type = model_info.get("model_type", "standard")

        logging.info(f"Using model type: {model_type}")
        if model_type == "unconstrained_weight":
            logging.info("Using unconstrained weight model with fully dynamic weights")
            logging.info("Model determines optimal importance of description/context/color for each shape")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, _ = clip.load("ViT-B/32", device=device)
        
        # Select appropriate model class based on model type
        if model_type == "unconstrained_weight":
            model = UnconstrainedWeightModel(embed_dim).to(device)
        elif model_type == "flexible_weight":
            # For backward compatibility
            from test_polygons_rgb_with_flexible_weights import FlexibleWeightCLIPToRGB
            model = FlexibleWeightCLIPToRGB(embed_dim).to(device)
        elif model_type == "hierarchical_weight":
            # For backward compatibility
            from test_polygons_rgb_with_hierarchical_weights import HierarchicalWeightCLIPToRGB
            model = HierarchicalWeightCLIPToRGB(embed_dim).to(device)
        else:
            model = CLIPToRGB(embed_dim).to(device)
            
        model.load_state_dict(torch.load(args.model, map_location=device))
        model.eval()

        # Sort shapes by parent-child relationship
        sorted_shapes = sort_shapes(shapes)
        
        # Log processing order
        shape_order = [f"{s.get('id')}:{s.get('description', '')[:20]}..." for s in sorted_shapes]
        logging.info(f"Processing shapes in order: {shape_order}")

        # Calculate canvas size
        min_x = min(p[0] for s in shapes for p in s["polygon"])
        max_x = max(p[0] for s in shapes for p in s["polygon"])
        min_y = min(p[1] for s in shapes for p in s["polygon"])
        max_y = max(p[1] for s in shapes for p in s["polygon"])
        width, height = int(max_x - min_x + 1), int(max_y - min_y + 1)

        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)

        # For contextual models, maintain context and color info
        contexts = {None: torch.zeros(embed_dim, device=device)}  # Map of contexts by shape ID
        colors = {None: torch.zeros(3, device=device)}  # Map of colors by shape ID
        last_sibling = {}  # Track last sibling by parent ID

        # Track weight statistics for analysis
        all_text_weights = []
        all_context_weights = []
        all_color_weights = []
        
        # Track detailed shape information
        shape_details = []
        
        # Process each shape in sorted order
        for shape in sorted_shapes:
            shape_id = shape.get("id")
            parent_id = shape.get("parent")
            desc = shape.get("description", "")
            polygon = [(x - min_x, y - min_y) for x, y in shape["polygon"]]
            
            # Get context to use (sibling or parent)
            if model_type in ["unconstrained_weight", "flexible_weight", "hierarchical_weight"]:
                if parent_id in last_sibling:
                    # Use last sibling's context
                    context_id = last_sibling[parent_id]
                    context_to_use = contexts[context_id]
                    color_to_use = colors[context_id]
                    logging.debug(f"Shape {shape_id} using sibling context from {context_id}")
                else:
                    # Fall back to parent context
                    context_to_use = contexts.get(parent_id, contexts[None])
                    color_to_use = colors.get(parent_id, colors[None])
                    logging.debug(f"Shape {shape_id} using parent context from {parent_id}")
                
                # Set current context for prediction
                contexts["current"] = context_to_use
                colors["color_current"] = color_to_use
            
            # Predict RGB color
            if model_type in ["unconstrained_weight", "flexible_weight", "hierarchical_weight"]:
                # For weight-based models, get both color and weights
                pred_rgb, pred_weights = predict_rgb(
                    desc, clip_model, model, device, model_type, 
                    contexts, colors, embed_dim
                )
                
                # Convert to RGB tuple
                rgb = tuple((pred_rgb.clamp(0, 1) * 255).round().int().tolist())
                
                # Update context with predicted weights
                with torch.no_grad():
                    # Get text embedding
                    tokens = clip.tokenize([desc]).to(device)
                    text_embed = clip_model.encode_text(tokens)[0]
                    text_embed = text_embed / text_embed.norm()
                    
                    # Update context and color info for next shapes
                    contexts[shape_id] = text_embed
                    colors[shape_id] = pred_rgb
                    
                    # Update last sibling
                    last_sibling[parent_id] = shape_id
                
                # Track weights for analysis
                all_text_weights.append(pred_weights[0].item())
                all_context_weights.append(pred_weights[1].item())
                all_color_weights.append(pred_weights[2].item())
                
                # Track shape details
                shape_details.append({
                    "id": shape_id,
                    "description": desc,
                    "weights": pred_weights.tolist(),
                    "rgb": rgb
                })
                
                # Log predicted weights
                if model_type == "unconstrained_weight":
                    logging.info(f"Shape {shape_id} weights: text={pred_weights[0]:.3f}, "
                             f"context={pred_weights[1]:.3f}, color={pred_weights[2]:.3f}")
                else:
                    logging.info(f"Shape {shape_id} weights: embed={pred_weights[0]:.3f}, "
                             f"context={pred_weights[1]:.3f}, color={pred_weights[2]:.3f}")
                
            else:
                # For standard model, just predict color
                pred_rgb = predict_rgb(
                    desc, clip_model, model, device, model_type, 
                    contexts, colors, embed_dim
                )
                
                # Convert to RGB tuple
                rgb = tuple((pred_rgb.clamp(0, 1) * 255).round().int().tolist())
                
                # Update context for next shapes (standard simple update)
                contexts[shape_id] = contexts.get("current", contexts[None])
            
            # Draw polygon with predicted color
            logging.info(f"Shape {shape_id} - '{desc}' -> {rgb}")
            draw.polygon(polygon, fill=rgb)

        # Save output image
        os.makedirs(os.path.dirname(args.output_image), exist_ok=True)
        img.save(args.output_image)
        logging.info(f"Saved image to {args.output_image}")
        
        # Log weight statistics if available
        if model_type in ["unconstrained_weight", "flexible_weight", "hierarchical_weight"] and all_text_weights:
            # Calculate statistics
            avg_text = sum(all_text_weights) / len(all_text_weights)
            avg_context = sum(all_context_weights) / len(all_context_weights)
            avg_color = sum(all_color_weights) / len(all_color_weights)
            
            # Calculate interesting patterns
            text_dominant = sum(1 for t, c, r in zip(all_text_weights, all_context_weights, all_color_weights) 
                              if t > c and t > r)
            context_dominant = sum(1 for t, c, r in zip(all_text_weights, all_context_weights, all_color_weights) 
                                 if c > t and c > r)
            color_dominant = sum(1 for t, c, r in zip(all_text_weights, all_context_weights, all_color_weights) 
                               if r > t and r > c)
            
            # Various ordering patterns
            text_gt_context = sum(1 for t, c in zip(all_text_weights, all_context_weights) if t > c)
            context_gt_color = sum(1 for c, r in zip(all_context_weights, all_color_weights) if c > r)
            text_gt_color = sum(1 for t, r in zip(all_text_weights, all_color_weights) if t > r)
            
            total_shapes = len(all_text_weights)
            
            label_text = "text" if model_type == "unconstrained_weight" else "embed"
            
            logging.info(f"Weight statistics summary:")
            logging.info(f"Average weights: {label_text}={avg_text:.3f}, context={avg_context:.3f}, color={avg_color:.3f}")
            logging.info(f"Dominant source patterns:")
            logging.info(f"  {label_text} dominant: {text_dominant/total_shapes*100:.1f}% of shapes")
            logging.info(f"  Context dominant: {context_dominant/total_shapes*100:.1f}% of shapes")
            logging.info(f"  Color dominant: {color_dominant/total_shapes*100:.1f}% of shapes")
            logging.info(f"Pairwise comparisons:")
            logging.info(f"  {label_text} > Context: {text_gt_context/total_shapes*100:.1f}% of shapes")
            logging.info(f"  Context > Color: {context_gt_color/total_shapes*100:.1f}% of shapes") 
            logging.info(f"  {label_text} > Color: {text_gt_color/total_shapes*100:.1f}% of shapes")
            
            # Create weight_analysis directory if it doesn't exist
            weight_analysis_dir = os.path.join(os.path.dirname(__file__), "weight_analysis")
            os.makedirs(weight_analysis_dir, exist_ok=True)

            # Save weight data for analysis
            try:
                weight_analysis_path = os.path.join(weight_analysis_dir, f"{json_base_name}_weight_analysis.json")
                with open(weight_analysis_path, "w") as f:
                    json.dump(shape_details, f, indent=2)
                logging.info(f"Saved weight analysis to {weight_analysis_path}")
            except Exception as e:
                logging.warning(f"Failed to save weight analysis: {e}")

            # Find and log shapes with interesting weight patterns
            if shape_details:
                # Highest text/embed weight
                highest_text = max(shape_details, key=lambda x: x['weights'][0])
                logging.info(f"Highest {label_text} weight ({highest_text['weights'][0]:.3f}): "
                           f"Shape {highest_text['id']} - '{highest_text['description'][:50]}...'")
                
                # Highest context weight
                highest_context = max(shape_details, key=lambda x: x['weights'][1])
                logging.info(f"Highest context weight ({highest_context['weights'][1]:.3f}): "
                           f"Shape {highest_context['id']} - '{highest_context['description'][:50]}...'")
                
                # Highest color weight
                highest_color = max(shape_details, key=lambda x: x['weights'][2])
                logging.info(f"Highest color weight ({highest_color['weights'][2]:.3f}): "
                           f"Shape {highest_color['id']} - '{highest_color['description'][:50]}...'")
                
                # Any unusual orderings
                if color_dominant > 0:
                    color_dom_example = next((s for s in shape_details 
                                          if s['weights'][2] > s['weights'][0] and s['weights'][2] > s['weights'][1]), None)
                    if color_dom_example:
                        logging.info(f"Example with color dominance: Shape {color_dom_example['id']} - "
                                   f"'{color_dom_example['description'][:50]}...' - weights: {color_dom_example['weights']}")
                
                if context_dominant > 0:
                    context_dom_example = next((s for s in shape_details 
                                            if s['weights'][1] > s['weights'][0] and s['weights'][1] > s['weights'][2]), None)
                    if context_dom_example:
                        logging.info(f"Example with context dominance: Shape {context_dom_example['id']} - "
                                   f"'{context_dom_example['description'][:50]}...' - weights: {context_dom_example['weights']}")

    except Exception as e:
        logging.error(traceback.format_exc())
        sys.exit(1)
    finally:
        close_logging_handlers()

def sort_shapes(shapes):
    """Sort shapes by parent-child relationship using BFS"""
    # Create a graph representation
    children = defaultdict(list)
    for shape in shapes:
        parent = shape.get("parent")
        children[parent].append(shape)
    
    # Perform BFS traversal
    sorted_shapes = []
    queue = list(children[None])  # Start with root nodes (parent=None)
    
    # If no root nodes found, use all shapes
    if not queue and shapes:
        logging.warning("No root nodes (parent=None) found, using all shapes as roots")
        queue = shapes
        
    while queue:
        node = queue.pop(0)
        sorted_shapes.append(node)
        queue.extend(children[node.get("id")])
    
    return sorted_shapes

if __name__ == '__main__':
    main()