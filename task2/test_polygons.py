# """
# test_polygons.py (Standard Version)

# This script loads a trained classifier along with CLIP's text encoder
# to predict a color for each polygon from a given JSON file with extensive logging.
# It draws only the filled polygons on the final composite image (no debug overlays).
# """

# import os
# import sys
# import json
# import re
# import logging
# import math
# import argparse
# import traceback
# import datetime

# import torch
# import torch.nn as nn
# import clip
# from PIL import Image, ImageDraw, ImageColor

# # Create logs directory if it doesn't exist
# logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
# os.makedirs(logs_dir, exist_ok=True)

# # Set up log file name with timestamp
# timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# log_file = os.path.join(logs_dir, f"test_polygons_{timestamp}.log")

# # Configure detailed logging
# logging.basicConfig(
#     level=logging.DEBUG,
#     format='[%(levelname)s] [%(asctime)s] [%(filename)s:%(lineno)d] %(message)s',
#     handlers=[
#         logging.StreamHandler(),
#         logging.FileHandler(log_file)
#     ]
# )

# # Candidate palette (same as training)
# CANDIDATE_COLORS = [
#     ("black",  (0, 0, 0)),
#     ("white",  (255, 255, 255)),
#     ("red",    (255, 0, 0)),
#     ("green",  (0, 255, 0)),
#     ("blue",   (0, 0, 255)),
#     ("yellow", (255, 255, 0)),
#     ("orange", (255, 165, 0)),
#     ("purple", (128, 0, 128)),
#     ("pink",   (255, 192, 203)),
#     ("brown",  (165, 42, 42)),
#     ("gray",   (128, 128, 128))
# ]
# NUM_CANDIDATES = len(CANDIDATE_COLORS)

# # Color synonyms (for potential future use; not used for drawing now)
# COLOR_SYNONYMS = {
#     "black": ["dark", "ebony", "jet", "midnight"],
#     "white": ["light", "ivory", "snow", "pale", "bright", "blank"],
#     "red": ["crimson", "scarlet", "maroon", "ruby", "fire", "blood"],
#     "green": ["emerald", "lime", "forest", "olive", "grass"],
#     "blue": ["azure", "navy", "turquoise", "sky", "ocean", "teal", "water"],
#     "yellow": ["gold", "golden", "sunny", "sunshine", "lemon"],
#     "orange": ["amber", "tangerine", "apricot", "peach", "sunset"],
#     "purple": ["violet", "lavender", "indigo", "magenta", "lilac"],
#     "pink": ["rose", "salmon", "fuchsia", "blush"],
#     "brown": ["tan", "chocolate", "coffee", "wooden", "wood", "dirt", "earth"],
#     "gray": ["grey", "silver", "ash", "charcoal", "slate"]
# }


# def parse_explicit_color(description: str):
#     if not description:
#         return None
#     logging.debug("Parsing explicit numeric color from description: '%s'", description)
#     text_lower = description.lower()
#     rgb_pattern = r'rgb\s*\(?(\d+)\s*,?\s*(\d+)\s*,?\s*(\d+)\s*\)?'
#     match = re.search(rgb_pattern, text_lower)
#     if match:
#         r, g, b = int(match.group(1)), int(match.group(2)), int(match.group(3))
#         logging.debug("Found explicit rgb color: (%d, %d, %d)", r, g, b)
#         return (r, g, b)
#     hex_pattern = r'#[0-9a-fA-F]{3,6}'
#     match = re.search(hex_pattern, text_lower)
#     if match:
#         hex_str = match.group(0)
#         try:
#             rgb = ImageColor.getrgb(hex_str)
#             logging.debug("Found explicit hex color: %s -> %s", hex_str, str(rgb))
#             return rgb
#         except ValueError:
#             logging.debug("Invalid hex color found: %s", hex_str)
#     for name, rgb in CANDIDATE_COLORS:
#         patterns = [
#             fr'\b{name}\s+colou?red\b',
#             fr'\bcolou?red\s+{name}\b',
#             fr'\b{name}\s+fill\b',
#             fr'\bfill.+{name}\b'
#         ]
#         for pattern in patterns:
#             if re.search(pattern, text_lower):
#                 logging.debug("Found color description '%s' -> %s", pattern, rgb)
#                 return rgb
#     logging.debug("No explicit numeric color found.")
#     return None


# def contains_candidate_color(text: str):
#     if not text:
#         return None
#     text_lower = text.lower()
#     words = re.findall(r'\b[a-zA-Z]+\b', text_lower)
#     candidate_names = {name.lower() for (name, _) in CANDIDATE_COLORS}
#     for w in words:
#         if w in candidate_names:
#             logging.debug("Found direct candidate color word in text: '%s'", w)
#             return w
#     for color, synonyms in COLOR_SYNONYMS.items():
#         for synonym in synonyms:
#             if synonym in words:
#                 # Optional context check can go here if needed in future
#                 logging.debug("Found color synonym '%s' -> %s", synonym, color)
#                 return color
#     return None


# def get_weighted_merged_text(shape, id_to_shape):
#     desc_child = shape.get("description", "")
#     parent_id = shape.get("parent")
#     logging.debug("Processing shape ID %s with description: '%s'", shape.get("id"), desc_child)
#     logging.debug("Parent ID: %s", parent_id)
#     has_explicit_color = parse_explicit_color(desc_child) is not None
#     color_word = contains_candidate_color(desc_child)
#     logging.debug("Has explicit color: %s, Color word: %s", has_explicit_color, color_word)
#     if parent_id is not None and parent_id in id_to_shape:
#         desc_parent = id_to_shape[parent_id].get("description", "")
#         if has_explicit_color:
#             merged = (desc_child + " ") * 5 + desc_parent
#         elif color_word:
#             merged = (desc_child + " ") * 3 + desc_parent
#         else:
#             merged = (desc_child + " ") * 2 + desc_parent
#         logging.debug("Final merged text: '%s'", merged)
#         return merged
#     else:
#         return desc_child


# def compute_bounding_box(shapes):
#     min_x, max_x = float('inf'), float('-inf')
#     min_y, max_y = float('inf'), float('-inf')
#     for shape in shapes:
#         for (x, y) in shape["polygon"]:
#             min_x = min(min_x, x)
#             max_x = max(max_x, x)
#             min_y = min(min_y, y)
#             max_y = max(max_y, y)
#     logging.debug("Computed bounding box: min_x=%s, max_x=%s, min_y=%s, max_y=%s", min_x, max_x, min_y, max_y)
#     return min_x, max_x, min_y, max_y


# def get_color_confidence(logits, top_k=3):
#     probs = torch.nn.functional.softmax(logits, dim=1)[0]
#     values, indices = torch.topk(probs, min(top_k, NUM_CANDIDATES))
#     result = []
#     for i, idx in enumerate(indices):
#         color_name = CANDIDATE_COLORS[idx.item()][0]
#         confidence = values[i].item() * 100
#         result.append((color_name, confidence))
#     return result


# def main():
#     parser = argparse.ArgumentParser(description="Test polygon color prediction")
#     parser.add_argument("input_json", help="Input JSON file containing polygon data")
#     parser.add_argument("output_image", help="Output image file path")
#     parser.add_argument("--model", default="classifier_weights.pth", 
#                         help="Path to trained model weights")
#     args = parser.parse_args()

#     logging.info("Starting test_polygons.py with arguments: %s", args)
    
#     try:
#         script_dir = os.path.dirname(os.path.abspath(__file__))
#         if not script_dir:
#             script_dir = "."
#         logging.debug("Script directory: %s", script_dir)
        
#         json_path = args.input_json
#         if not os.path.isabs(json_path):
#             json_path = os.path.join(script_dir, json_path)
#         logging.debug("JSON path: %s", json_path)
        
#         if not os.path.isfile(json_path):
#             logging.error("Error: JSON file %s not found.", json_path)
#             return 1
            
#         with open(json_path, "r", encoding="utf-8") as f:
#             shapes = json.load(f)
#         logging.info("Loaded %d shapes from %s", len(shapes), args.input_json)
        
#         # Log first couple of shapes for inspection
#         for i, shape in enumerate(shapes[:2]):
#             logging.debug("Shape %d: %s", i, json.dumps(shape, indent=2))
            
#         id_to_shape = {s["id"]: s for s in shapes}
#         min_x, max_x, min_y, max_y = compute_bounding_box(shapes)
#         width, height = int(max_x - min_x + 1), int(max_y - min_y + 1)
#         logging.info("Canvas size: width=%d, height=%d", width, height)
        
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         logging.info("Using device: %s", device)
        
#         logging.info("Loading CLIP model...")
#         try:
#             clip_model, preprocess = clip.load("ViT-B/32", device=device)
#             logging.info("CLIP model loaded successfully")
#         except Exception as e:
#             logging.error("Failed to load CLIP model: %s", e)
#             logging.error(traceback.format_exc())
#             return 1
        
#         weights_path = os.path.join(script_dir, args.model)
#         logging.debug("Model weights path: %s", weights_path)
#         if not os.path.isfile(weights_path):
#             logging.error("Error: Trained classifier weights %s not found.", weights_path)
#             return 1

#         with torch.no_grad():
#             sample_tokens = clip.tokenize(["sample text"], truncate=True).to(device)
#             sample_embedding = clip_model.encode_text(sample_tokens)
#         embed_dim = sample_embedding.shape[-1]
#         logging.info("CLIP text embedding dimension: %d", embed_dim)
        
#         # Define classifier architecture (as per training)
#         classifier = nn.Sequential(
#             nn.Linear(embed_dim, 128),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(128, NUM_CANDIDATES)
#         ).to(device)
#         try:
#             classifier.load_state_dict(torch.load(weights_path, map_location=device))
#             logging.info("Loaded classifier model successfully.")
#         except Exception as e:
#             logging.error("Failed to load classifier model: %s", e)
#             logging.error(traceback.format_exc())
#             return 1
#         classifier.eval()
        
#         logging.info("Creating composite image with dimensions: %dx%d", width, height)
#         try:
#             composite = Image.new("RGB", (width, height), "white")
#             draw = ImageDraw.Draw(composite)
#         except Exception as e:
#             logging.error("Failed to create image: %s", e)
#             logging.error(traceback.format_exc())
#             return 1

#         # Compute hierarchy levels (for potential ordering; no debug overlays now)
#         shapes_by_level = {}
#         for shape in shapes:
#             level = 0
#             current_id = shape.get("parent")
#             while current_id is not None:
#                 level += 1
#                 parent = id_to_shape.get(current_id)
#                 if parent:
#                     current_id = parent.get("parent")
#                 else:
#                     break
#             shapes_by_level.setdefault(level, []).append(shape)
#         for level, level_shapes in shapes_by_level.items():
#             logging.debug("Level %d: %d shapes; IDs: %s", level, len(level_shapes), [s.get("id") for s in level_shapes])
#         max_level = max(shapes_by_level.keys()) if shapes_by_level else 0
#         logging.info("Maximum hierarchy level: %d", max_level)
        
#         # Process and draw each shape (only polygon fill is applied)
#         for level in range(max_level + 1):
#             level_shapes = shapes_by_level.get(level, [])
#             logging.info("Processing %d shapes at level %d", len(level_shapes), level)
#             for shape in level_shapes:
#                 shape_id = shape.get("id")
#                 logging.debug("Processing shape ID %s", shape_id)
#                 merged_text = get_weighted_merged_text(shape, id_to_shape)
#                 tokens = clip.tokenize([merged_text], truncate=True).to(device)
#                 with torch.no_grad():
#                     text_feats = clip_model.encode_text(tokens)
#                     text_feats /= text_feats.norm(dim=-1, keepdim=True)
#                     logits = classifier(text_feats)
#                     predicted_label = torch.argmax(logits, dim=-1).item()
#                 predicted_color = CANDIDATE_COLORS[predicted_label][1]
#                 color_confidence = get_color_confidence(logits, top_k=3)
#                 logging.debug("Shape %s: Predicted %s with confidence %.1f%%", shape_id,
#                               CANDIDATE_COLORS[predicted_label][0], color_confidence[0][1])
#                 logging.debug("Top predictions: %s", color_confidence)
#                 polygon_points = shape["polygon"]
#                 offset_points = [(x - min_x, y - min_y) for (x, y) in polygon_points]
#                 logging.debug("Drawing polygon ID %s with color %s", shape_id, predicted_color)
#                 try:
#                     draw.polygon(offset_points, fill=predicted_color)
#                     logging.debug("Successfully drew polygon ID %s", shape_id)
#                 except Exception as e:
#                     logging.error("Error drawing polygon ID %s: %s", shape_id, e)
#                     logging.error("Polygon points: %s", offset_points)
#                     continue

#         output_path = args.output_image
#         if not os.path.isabs(output_path):
#             output_path = os.path.join(script_dir, output_path)
#         results_dir = os.path.dirname(output_path)
#         if results_dir and not os.path.exists(results_dir):
#             logging.debug("Creating results directory: %s", results_dir)
#             os.makedirs(results_dir, exist_ok=True)
#         logging.info("Saving image to: %s", output_path)
#         try:
#             composite.save(output_path)
#             logging.info("Final composite image saved to: %s", output_path)
#         except Exception as e:
#             logging.error("Error saving image: %s", e)
#             logging.error(traceback.format_exc())
#             return 1
#         return 0
#     except Exception as e:
#         logging.error("Unexpected error: %s", e)
#         logging.error(traceback.format_exc())
#         return 1

# if __name__ == "__main__":
#     exit_code = main()
#     logging.info("Script completed with exit code: %s", exit_code)
#     sys.exit(exit_code)
"""
test_polygons.py (Enhanced Version)

This script loads a trained classifier along with CLIP's text encoder to predict
colors for polygons from a given JSON file. Key improvements include:
- Advanced semantic color understanding
- Confidence-based color selection
- Enhanced parent-child relationship handling
- Smart fallback for low-confidence predictions
- Hierarchical drawing with transparency control
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
import random

import torch
import torch.nn as nn
import clip
from PIL import Image, ImageDraw, ImageColor

# Create logs directory if it doesn't exist
logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Set up log file name with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(logs_dir, f"test_polygons_{timestamp}.log")

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] [%(asctime)s] [%(filename)s:%(lineno)d] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)

# Candidate palette (same as training)
CANDIDATE_COLORS = [
    ("black",  (0, 0, 0)),
    ("white",  (255, 255, 255)),
    ("red",    (255, 0, 0)),
    ("green",  (0, 255, 0)),
    ("blue",   (0, 0, 255)),
    ("yellow", (255, 255, 0)),
    ("orange", (255, 165, 0)),
    ("purple", (128, 0, 128)),
    ("pink",   (255, 192, 203)),
    ("brown",  (165, 42, 42)),
    ("gray",   (128, 128, 128))
]
NUM_CANDIDATES = len(CANDIDATE_COLORS)

# Expanded color synonyms and associations for enhanced semantic understanding
COLOR_SYNONYMS = {
    "black": ["dark", "ebony", "jet", "midnight", "obsidian", "shadow", "cosmic", "void", "space", "night", "darkness"],
    "white": ["light", "ivory", "snow", "pale", "bright", "blank", "alabaster", "pearl", "glistening", "star", "moon", "cloud"],
    "red": ["crimson", "scarlet", "maroon", "ruby", "fire", "blood", "ember", "flame", "burning", "cherry", "rose", "lava"],
    "green": ["emerald", "lime", "forest", "olive", "grass", "moss", "jade", "mint", "alien", "nature", "leaf", "plant", "jungle"],
    "blue": ["azure", "navy", "turquoise", "sky", "ocean", "teal", "water", "cosmic", "sapphire", "cobalt", "cerulean", "aqua", "sea"],
    "yellow": ["gold", "golden", "sunny", "sunshine", "lemon", "amber", "solar", "radiant", "sunflower", "bright", "daylight"],
    "orange": ["amber", "tangerine", "apricot", "peach", "sunset", "rust", "copper", "burning", "saffron", "autumn", "harvest"],
    "purple": ["violet", "lavender", "indigo", "magenta", "lilac", "amethyst", "royal", "cosmic", "twilight", "nebula", "mauve", "plum"],
    "pink": ["rose", "salmon", "fuchsia", "blush", "coral", "mallow", "flamingo", "pastel", "cherry-blossom", "bubblegum"],
    "brown": ["tan", "chocolate", "coffee", "wooden", "wood", "dirt", "earth", "russet", "copper", "chestnut", "mahogany", "mocha", "umber"],
    "gray": ["grey", "silver", "ash", "charcoal", "slate", "smoke", "pewter", "stone", "graphite", "foggy", "cloud", "overcast"]
}

# Semantic associations for deeper understanding
SEMANTIC_ASSOCIATIONS = {
    "space": "black",
    "sky": "blue",
    "star": "white",
    "nebula": "purple",
    "galaxy": "blue",
    "cosmos": "black",
    "soil": "brown",
    "grass": "green",
    "forest": "green",
    "sea": "blue",
    "sun": "yellow",
    "moon": "white",
    "fire": "red",
    "blood": "red",
    "energy": "yellow",
    "cloud": "white",
    "metal": "gray",
    "stone": "gray",
    "rose": "pink",
    "sunset": "orange",
    "cosmic": "blue",
    "earth": "brown",
    "flame": "orange",
    "water": "blue",
    "rock": "gray",
    "tree": "brown",
    "plant": "green",
    "planet": "blue",
    "sand": "yellow",
    "alien": "green",
    "royal": "purple",
    "lava": "red",
    "beam": "yellow",
    "glow": "white",
    "shadow": "black",
    "night": "black",
    "day": "blue",
    "ice": "white",
    "snow": "white",
    "digital": "blue",
    "tech": "blue",
    "futuristic": "blue",
    "cyber": "blue",
    "ancient": "brown",
    "portal": "purple",
    "light": "white",
    "dark": "black",
    "mystical": "purple",
    "natural": "green",
    "artificial": "gray",
    "horizon": "blue"
}

class EnhancedColorClassifier(nn.Module):
    """
    Enhanced neural network classifier with additional layers and dropout.
    """
    def __init__(self, input_dim, hidden_dims=[256, 128], num_classes=NUM_CANDIDATES, dropout_rates=[0.4, 0.3]):
        super(EnhancedColorClassifier, self).__init__()
        
        # Validate parameters
        assert len(hidden_dims) == len(dropout_rates), "Must provide dropout rates for each hidden layer"
        
        layers = []
        prev_dim = input_dim
        
        # Create hidden layers with dropout
        for i, (hidden_dim, dropout_rate) in enumerate(zip(hidden_dims, dropout_rates)):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

def parse_explicit_color(description: str):
    """
    Enhanced explicit color detection: returns an (R, G, B) tuple if found.
    Handles various notation formats including RGB and hex.
    """
    if not description:
        return None
    
    text_lower = description.lower()
    
    # Match flexible RGB format
    rgb_pattern = r'rgb\s*\(?(\d+)\s*,?\s*(\d+)\s*,?\s*(\d+)\s*\)?'
    match = re.search(rgb_pattern, text_lower)
    if match:
        r, g, b = int(match.group(1)), int(match.group(2)), int(match.group(3))
        logging.debug("Found explicit rgb color: (%d, %d, %d)", r, g, b)
        return (r, g, b)
    
    # Match hex format
    hex_pattern = r'#[0-9a-fA-F]{3,6}'
    match = re.search(hex_pattern, text_lower)
    if match:
        hex_str = match.group(0)
        try:
            rgb = ImageColor.getrgb(hex_str)
            logging.debug("Found explicit hex color: %s -> %s", hex_str, str(rgb))
            return rgb
        except ValueError:
            logging.debug("Invalid hex color found: %s", hex_str)
    
    # Match color descriptions
    for name, rgb in CANDIDATE_COLORS:
        patterns = [
            fr'\b{name}\s+colou?red\b',
            fr'\bcolou?red\s+{name}\b',
            fr'\b{name}\s+fill\b',
            fr'\bfill.+{name}\b'
        ]
        for pattern in patterns:
            if re.search(pattern, text_lower):
                logging.debug("Found color description '%s' -> %s", pattern, rgb)
                return rgb
                
    return None

def analyze_semantic_cues(text: str):
    """
    Analyzes text for semantic color cues based on associated concepts and themes.
    Returns a dictionary mapping color names to their semantic strength scores.
    """
    if not text:
        return {}
    
    text_lower = text.lower()
    words = re.findall(r'\b[a-zA-Z]+\b', text_lower)
    
    color_scores = {color: 0.0 for color, _ in CANDIDATE_COLORS}
    
    # Check for semantic associations
    for word in words:
        if word in SEMANTIC_ASSOCIATIONS:
            color = SEMANTIC_ASSOCIATIONS[word]
            color_scores[color] += 1.0
    
    # Look for descriptive phrases that suggest colors
    cosmic_phrases = [
        ("deep space", "black", 1.5),
        ("night sky", "black", 1.5),
        ("cosmic horizon", "blue", 1.2),
        ("blazing comet", "orange", 1.5),
        ("fiery trail", "red", 1.3),
        ("mysterious portal", "purple", 1.4),
        ("alien structure", "green", 1.3),
        ("energy beam", "yellow", 1.2),
        ("glowing light", "white", 1.2),
        ("reflective surface", "white", 1.0),
        ("distant star", "white", 1.2),
        ("swirling nebula", "purple", 1.5),
        ("cosmic dust", "gray", 1.1),
        ("digital grid", "blue", 1.2),
        ("ancient artifact", "brown", 1.3),
        ("crystal formation", "blue", 1.1),
        ("molten core", "red", 1.4)
    ]
    
    for phrase, color, weight in cosmic_phrases:
        if phrase in text_lower:
            color_scores[color] += weight
    
    # Normalize scores
    total_score = sum(color_scores.values()) 
    if total_score > 0:
        for color in color_scores:
            color_scores[color] /= total_score
    
    return color_scores

def contains_candidate_color(text: str):
    """
    Detects if a text contains a candidate color word or strong synonym.
    Returns the candidate color word (lowercase) if found, otherwise None.
    """
    if not text:
        return None
    
    text_lower = text.lower()
    words = re.findall(r'\b[a-zA-Z]+\b', text_lower)
    
    # First check for direct candidate color names
    for w in words:
        if w in {name.lower() for name, _ in CANDIDATE_COLORS}:
            logging.debug("Found direct candidate color word in text: '%s'", w)
            return w
    
    # Check for color synonyms
    for color, synonyms in COLOR_SYNONYMS.items():
        for synonym in synonyms:
            if synonym in words:
                # Look for context words that strengthen the color association
                context_words = ["color", "colour", "fill", "filled", "background", "foreground", 
                               "hue", "tint", "shade", "tone", "painted", "colored"]
                
                word_indices = [i for i, word in enumerate(words) if word == synonym]
                for idx in word_indices:
                    context_range = range(max(0, idx-5), min(len(words), idx+5))
                    for context_idx in context_range:
                        if words[context_idx] in context_words:
                            logging.debug("Found color synonym '%s' near context word '%s' -> %s", 
                                        synonym, words[context_idx], color)
                            return color
                
                # Check for visual objects that imply colors
                visual_objects = {
                    "sky": "blue",
                    "grass": "green",
                    "sun": "yellow",
                    "blood": "red",
                    "snow": "white",
                    "night": "black",
                    "wood": "brown",
                    "rose": "red",
                    "fire": "red",
                    "water": "blue",
                    "nebula": "purple",
                    "star": "white",
                    "cosmic": "blue",
                    "flame": "orange"
                }
                
                for obj, color_name in visual_objects.items():
                    if obj in words and color_name == color:
                        logging.debug("Found visual object '%s' implying color '%s'", obj, color_name)
                        return color_name
                
                # Strong explicit color synonyms that always indicate a color
                strong_synonyms = [
                    "scarlet", "crimson", "azure", "indigo", "violet", "emerald", 
                    "ebony", "ivory", "golden", "amber", "ruby", "sapphire"
                ]
                if synonym in strong_synonyms:
                    logging.debug("Found strong color synonym '%s' -> %s", synonym, color)
                    return color
    
    return None

def get_weighted_merged_text(shape, id_to_shape):
    """
    Returns a sophisticated merged text combining child and parent descriptions,
    with weighted emphasis based on color information and semantic content.
    """
    desc_child = shape.get("description", "")
    parent_id = shape.get("parent")
    shape_id = shape.get("id")
    
    logging.debug("Processing shape ID %s with description: '%s'", shape_id, desc_child)
    logging.debug("Parent ID: %s", parent_id)
    
    # Analyze for explicit and semantic color indicators
    has_explicit_color = parse_explicit_color(desc_child) is not None
    color_word = contains_candidate_color(desc_child)
    semantic_scores = analyze_semantic_cues(desc_child)
    has_strong_semantics = any(score > 0.4 for score in semantic_scores.values())
    
    logging.debug("Has explicit color: %s, Color word: %s, Strong semantics: %s", 
                 has_explicit_color, color_word, has_strong_semantics)
    
    if parent_id is not None and parent_id in id_to_shape:
        desc_parent = id_to_shape[parent_id].get("description", "")
        
        # Weight child description based on color information
        if has_explicit_color:
            # Explicit RGB/HEX colors are given highest priority
            merged = (desc_child + " ") * 5 + desc_parent
            logging.debug("Child has EXPLICIT color - heavily weighted merge for shape %s", shape_id)
        elif color_word:
            # Direct color words get strong weight
            merged = (desc_child + " ") * 3 + desc_parent
            logging.debug("Child has color word - heavily weighted merge for shape %s", shape_id)
        elif has_strong_semantics:
            # Strong semantic color cues get moderate weight
            merged = (desc_child + " ") * 3 + desc_parent
            logging.debug("Child has strong semantic cues - moderately weighted merge for shape %s", shape_id)
        else:
            # Default balanced merge
            merged = (desc_child + " ") * 2 + desc_parent
            logging.debug("Balanced merge for shape %s", shape_id)
            
        logging.debug("Final merged text: '%s'", merged)
        return merged
    else:
        logging.debug("Using only child description for shape %s", shape_id)
        return desc_child

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

def get_color_confidence(logits, top_k=3):
    """
    Returns the top color predictions and their confidence scores.
    """
    probs = torch.nn.functional.softmax(logits, dim=1)[0]
    values, indices = torch.topk(probs, min(top_k, NUM_CANDIDATES))
    
    result = []
    for i, idx in enumerate(indices):
        color_name = CANDIDATE_COLORS[idx.item()][0]
        confidence = values[i].item() * 100
        result.append((color_name, confidence))
    
    return result

def select_best_color(logits, shape_id, parent_color=None, confidence_threshold=0.3):
    """
    Selects the best color based on prediction confidence and context.
    Uses parent color as a fallback for low confidence predictions.
    """
    probs = torch.nn.functional.softmax(logits, dim=1)[0]
    max_prob, predicted_idx = torch.max(probs, dim=0)
    predicted_color = CANDIDATE_COLORS[predicted_idx.item()][1]
    
    # Get top predictions for logging
    top_predictions = get_color_confidence(logits)
    
    # Log prediction information
    prediction_msg = f"Shape {shape_id}: Predicted {CANDIDATE_COLORS[predicted_idx.item()][0]} with confidence {max_prob.item()*100:.1f}%"
    logging.debug(prediction_msg)
    logging.debug("Top predictions: %s", top_predictions)
    
    # Apply confidence-based selection
    if max_prob.item() >= confidence_threshold:
        return predicted_color, prediction_msg
    elif parent_color:
        fallback_msg = f"{prediction_msg} (low confidence; using parent color instead)"
        logging.debug("Using parent color as fallback due to low confidence")
        return parent_color, fallback_msg
    else:
        # Default neutral color for very uncertain predictions with no parent color
        neutral_colors = ["gray", "white", "blue", "black"]
        fallback_color_name = random.choice(neutral_colors)
        fallback_color = next(rgb for name, rgb in CANDIDATE_COLORS if name == fallback_color_name)
        
        fallback_msg = f"{prediction_msg} (low confidence; using {fallback_color_name} as fallback)"
        logging.debug("Using fallback color due to low confidence and no parent color")
        return fallback_color, fallback_msg

def load_model(model_path, model_info_path, device):
    """
    Loads the trained model architecture and weights.
    """
    # Attempt to load model architecture information
    try:
        if os.path.exists(model_info_path):
            with open(model_info_path, 'r') as f:
                model_info = json.load(f)
                
            input_dim = model_info.get('embed_dim', 512)
            hidden_dims = model_info.get('hidden_dims', [256, 128])
            num_classes = model_info.get('num_classes', NUM_CANDIDATES)
            dropout_rates = model_info.get('dropout_rates', [0.4, 0.3])
            
            logging.info("Loaded model architecture from info file")
        else:
            # Default architecture if info file not found
            logging.warning("Model info file not found. Using default architecture.")
            input_dim = 512  # Standard CLIP embedding dim
            hidden_dims = [256, 128]
            num_classes = NUM_CANDIDATES
            dropout_rates = [0.4, 0.3]
    except Exception as e:
        logging.error(f"Error loading model info: {e}")
        logging.warning("Using default model architecture")
        input_dim = 512
        hidden_dims = [256, 128]
        num_classes = NUM_CANDIDATES
        dropout_rates = [0.4, 0.3]
    
    # Create model with the appropriate architecture
    model = EnhancedColorClassifier(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
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
    parser = argparse.ArgumentParser(description="Enhanced polygon color prediction testing")
    parser.add_argument("input_json", help="Input JSON file containing polygon data")
    parser.add_argument("output_image", help="Output image file path")
    parser.add_argument("--model", default="classifier_weights.pth", 
                       help="Path to trained model weights")
    parser.add_argument("--model-info", default="model_info.json",
                       help="Path to model architecture information")
    parser.add_argument("--confidence", type=float, default=0.3,
                       help="Confidence threshold for color selection (0.0-1.0)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug-level logging")
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logging.info("Starting enhanced test_polygons.py with arguments: %s", args)
    
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
            logging.error("Error: Trained classifier weights %s not found.", weights_path)
            return 1

        # Load the model
        try:
            classifier = load_model(weights_path, model_info_path, device)
            classifier.eval()
        except Exception as e:
            logging.error("Failed to load classifier model: %s", e)
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
        
        # Store predicted colors by shape ID for parent color lookup
        shape_colors = {}
        
        # Process and draw each shape by hierarchy level (parent shapes first)
        for level in range(max_level + 1):
            level_shapes = shapes_by_level.get(level, [])
            logging.info("Processing %d shapes at level %d", len(level_shapes), level)
            
            for shape in level_shapes:
                shape_id = shape.get("id")
                logging.debug("Processing shape ID %s", shape_id)
                
                # Get parent color for potential fallback
                parent_id = shape.get("parent")
                parent_color = shape_colors.get(parent_id) if parent_id else None
                
                # Get merged description text
                merged_text = get_weighted_merged_text(shape, id_to_shape)
                
                # Check for explicit color in description
                explicit_color = parse_explicit_color(merged_text)
                if explicit_color:
                    # Use explicit color directly if present in description
                    logging.debug("Using explicit color for shape %s: %s", shape_id, explicit_color)
                    shape_colors[shape_id] = explicit_color
                    polygon_points = shape["polygon"]
                    offset_points = [(x - min_x, y - min_y) for (x, y) in polygon_points]
                    draw.polygon(offset_points, fill=explicit_color)
                    continue
                
                # Process with neural model if no explicit color
                tokens = clip.tokenize([merged_text], truncate=True).to(device)
                with torch.no_grad():
                    text_feats = clip_model.encode_text(tokens)
                    text_feats /= text_feats.norm(dim=-1, keepdim=True)
                    logits = classifier(text_feats)
                
                # Select best color with confidence-based fallback
                predicted_color, prediction_msg = select_best_color(
                    logits, shape_id, parent_color, args.confidence
                )
                
                # Store color for this shape (for children)
                shape_colors[shape_id] = predicted_color
                
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