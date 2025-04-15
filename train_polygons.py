"""
train_polygons.py (Enhanced Version)

This script trains a sophisticated classifier on top of CLIP's text encoder to predict
colors for polygons based on their textual descriptions with advanced features:
- Rich semantic color understanding with expanded synonym support
- Adaptive data augmentation with class balancing
- Deep neural network architecture with additional layers
- Dynamic class weighting for underrepresented colors
- Enhanced parent-child relationship handling
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
from torch.utils.data.sampler import WeightedRandomSampler

# Create logs directory if it doesn't exist
logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Set up log file name with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(logs_dir, f"train_polygons_{timestamp}.log")

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO, 
    format='[%(levelname)s] [%(asctime)s] [%(filename)s:%(lineno)d] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)

# Expanded candidate palette with consistent ordering
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

# Expanded color name sets for detection
CANDIDATE_COLOR_NAMES = {name.lower() for (name, _) in CANDIDATE_COLORS}

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

# Extended color words (combined set from synonyms and associations)
EXTENDED_COLOR_WORDS = CANDIDATE_COLOR_NAMES.copy()
for color, synonyms in COLOR_SYNONYMS.items():
    for synonym in synonyms:
        EXTENDED_COLOR_WORDS.add(synonym)

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
        if w in CANDIDATE_COLOR_NAMES:
            logging.debug("Found direct candidate color word in text: '%s'", w)
            return w
    
    # Check for color synonyms
    for color, synonyms in COLOR_SYNONYMS.items():
        for synonym in synonyms:
            if synonym in words:
                # Look for context words that strengthen the color association
                context_words = ["color", "colour", "fill", "filled", "background", "foreground", 
                                "hue", "tint", "shade", "tone", "painted", "colored"]
                
                for context_word in context_words:
                    if context_word in words:
                        logging.debug("Found color synonym '%s' near context word '%s' -> %s", 
                                     synonym, context_word, color)
                        return color
                
                # Even without context words, return the color if it's a strong synonym
                strong_synonyms = ["scarlet", "crimson", "emerald", "azure", "ebony", 
                                  "golden", "silver", "ruby", "ivory", "jade", "amber"]
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

def get_target_label(text: str):
    """
    Returns the index of the most likely candidate color based on the text,
    using explicit colors, direct color words, and semantic analysis.
    """
    # First check for explicit color
    numeric_color = parse_explicit_color(text)
    if numeric_color is not None:
        # Check for exact match with candidate colors
        for i, (name, rgb) in enumerate(CANDIDATE_COLORS):
            if rgb == numeric_color:
                logging.debug("Exact match for explicit numeric color found: index=%d", i)
                return i
        
        # Find nearest candidate color by Euclidean distance
        min_dist = float('inf')
        best_idx = None
        for i, (name, rgb) in enumerate(CANDIDATE_COLORS):
            dist = math.sqrt((rgb[0] - numeric_color[0]) ** 2 +
                           (rgb[1] - numeric_color[1]) ** 2 +
                           (rgb[2] - numeric_color[2]) ** 2)
            if dist < min_dist:
                min_dist = dist
                best_idx = i
        
        logging.debug("Nearest candidate for numeric color found: index=%s, dist=%f", str(best_idx), min_dist)
        return best_idx

    # Look for direct color words
    text_lower = text.lower()
    words = re.findall(r'\b[a-zA-Z]+\b', text_lower)
    
    for w in words:
        if w in CANDIDATE_COLOR_NAMES:
            for i, (name, rgb) in enumerate(CANDIDATE_COLORS):
                if w == name.lower():
                    logging.debug("Candidate color word found: '%s' -> index=%d", w, i)
                    return i
    
    # Check for color synonyms
    for i, (color_name, _) in enumerate(CANDIDATE_COLORS):
        if color_name in COLOR_SYNONYMS:
            for synonym in COLOR_SYNONYMS[color_name]:
                if synonym in words:
                    logging.debug("Found color synonym '%s' -> %s, index=%d", synonym, color_name, i)
                    return i
    
    # Run semantic analysis
    semantic_scores = analyze_semantic_cues(text)
    if semantic_scores:
        # Find color with highest semantic score
        max_score = max(semantic_scores.values())
        if max_score > 0.3:  # Threshold for confidence
            for i, (color_name, _) in enumerate(CANDIDATE_COLORS):
                if semantic_scores[color_name] == max_score:
                    logging.debug("Selected color based on semantic analysis: %s, index=%d, score=%f", 
                                 color_name, i, max_score)
                    return i
    
    logging.debug("No target candidate color determined from text.")
    return None

def augment_dataset(sample, augment_factor=0.5, class_counts=None):
    """
    Augment a training sample by adding variant phrasings.
    Uses class_counts to apply stronger augmentation to underrepresented classes.
    """
    text, target = sample
    augmented = [(text, target)]
    
    if target is None:
        return augmented
    
    color_name = CANDIDATE_COLORS[target][0]
    
    # Calculate adaptive augmentation factor based on class rarity
    adaptive_factor = augment_factor
    if class_counts and class_counts.get(target, 0) > 0:
        max_count = max(class_counts.values())
        # The fewer samples of a class, the more augmentation it gets
        rarity_factor = 1.0 - (class_counts[target] / max_count)
        # Scale the augmentation factor
        adaptive_factor = min(0.9, augment_factor + rarity_factor * 0.4)
        logging.debug("Using adaptive augment factor %.2f for %s (count: %d)", 
                     adaptive_factor, color_name, class_counts[target])
    
    # Basic variations with adaptive selection probability
    variants = [
        f"This is a {color_name} shape.",
        f"The {color_name} polygon.",
        f"A shape that is {color_name} in color.",
        f"An object with {color_name} fill.",
        f"{color_name} filled shape."
    ]
    
    # Add variations for rare colors (< 10 samples)
    if class_counts and class_counts.get(target, 0) < 10:
        # Special handling for underrepresented colors
        if color_name == "purple":
            variants.extend([
                f"A mysterious {color_name} shape with cosmic energy.",
                f"An enigmatic {color_name} form suggesting royalty.",
                f"A shape with the color of twilight and magic."
            ])
        elif color_name == "brown":
            variants.extend([
                f"A warm {color_name} shape reminiscent of wood and earth.",
                f"An earthy {color_name} element with natural tones.",
                f"A shape with the rich color of soil and bark."
            ])
        elif color_name == "orange":
            variants.extend([
                f"A vibrant {color_name} shape like a setting sun.",
                f"A warm {color_name} form with fiery energy.",
                f"A shape with the color of autumn leaves."
            ])
    
    # Add all basic variations with adaptive probability
    for variant in variants:
        if random.random() < adaptive_factor:
            augmented.append((variant, target))
    
    # Add synonym variations with reduced probability
    if color_name in COLOR_SYNONYMS:
        for synonym in COLOR_SYNONYMS[color_name]:
            variant = f"A {synonym} colored shape."
            if random.random() < (adaptive_factor * 0.5):
                augmented.append((variant, target))
    
    # Add semantic variants based on common associations for rare classes
    if class_counts and class_counts.get(target, 0) < 5:
        semantic_variants = []
        
        if color_name == "blue":
            semantic_variants = [
                "A shape reminiscent of the deep ocean.",
                "An element with the color of the sky on a clear day.",
                "A cosmic form suggesting distant galaxies."
            ]
        elif color_name == "green":
            semantic_variants = [
                "A shape with the freshness of spring leaves.",
                "An element colored like a lush forest.",
                "A natural form with the vibrance of new growth."
            ]
        elif color_name == "purple":
            semantic_variants = [
                "A shape with the mystical quality of twilight.",
                "An element reminiscent of cosmic nebulae.",
                "A form with the rich hue of ancient royalty."
            ]
        
        for variant in semantic_variants:
            if random.random() < (adaptive_factor * 0.7):
                augmented.append((variant, target))
    
    return augmented

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

class ShapesDataset(data.Dataset):
    """
    Enhanced dataset for training color prediction with advanced augmentation
    and class balancing strategies.
    """
    def __init__(self, shapes, id_to_shape, augment=True):
        self.samples = []
        # Compute basic samples and count per class
        basic_samples = []
        color_counts = {i: 0 for i in range(NUM_CANDIDATES)}
        
        for shape in shapes:
            merged_text = get_weighted_merged_text(shape, id_to_shape)
            target = get_target_label(merged_text)
            if target is not None:
                basic_samples.append((merged_text, target))
                color_counts[target] += 1
            else:
                logging.debug("Skipping shape id %s due to no target label.", shape.get("id"))
        
        # Save counts for weighting and reporting
        self.basic_color_counts = color_counts
        self.total_basic_samples = len(basic_samples)
        
        logging.info("Color distribution in original samples:")
        for i, (name, _) in enumerate(CANDIDATE_COLORS):
            logging.info(f"  {name}: {color_counts[i]} samples")
        
        if augment:
            # Apply enhanced augmentation with adaptive factors
            for sample in basic_samples:
                augmented = augment_dataset(sample, augment_factor=0.6, class_counts=color_counts)
                self.samples.extend(augmented)
                
            # Report final dataset statistics
            augmented_counts = {i: 0 for i in range(NUM_CANDIDATES)}
            for _, target in self.samples:
                augmented_counts[target] += 1
                
            logging.info("Color distribution after augmentation:")
            for i, (name, _) in enumerate(CANDIDATE_COLORS):
                logging.info(f"  {name}: {augmented_counts[i]} samples")
        else:
            self.samples = basic_samples
            
        logging.info("Constructed dataset with %d samples (after augmentation).", len(self.samples))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

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

def calculate_dynamic_weights(dataset):
    """
    Calculate class weights dynamically based on class distribution,
    with special handling for severely underrepresented classes.
    """
    color_counts = dataset.basic_color_counts
    total_samples = dataset.total_basic_samples
    weights = []
    
    for i in range(NUM_CANDIDATES):
        count = color_counts[i]
        if count > 0:
            # Base inverse frequency weighting
            weight = total_samples / (NUM_CANDIDATES * count)
            
            # Apply additional boost for very rare classes
            if count < 5:
                weight *= 1.5  # 50% boost for very rare classes
            elif count < 10:
                weight *= 1.2  # 20% boost for somewhat rare classes
            
            weights.append(weight)
        else:
            weights.append(0.0)
    
    # Normalize weights to avoid extreme values
    max_weight = max(weights)
    if max_weight > 5.0:
        scale_factor = 5.0 / max_weight
        weights = [w * scale_factor for w in weights]
    
    return torch.tensor(weights, dtype=torch.float)

def train_model(dataset, clip_model, classifier, optimizer, device, 
               class_weights, epochs=15, patience=3, batch_size=16):
    """
    Train the classifier with advanced techniques including:
    - Early stopping
    - Learning rate scheduling
    - Dynamic class weighting
    - Gradient clipping
    """
    # Create train/validation split
    dataset_size = len(dataset)
    val_size = int(dataset_size * 0.2)  # 20% for validation
    train_size = dataset_size - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42))  # Fixed seed for reproducibility
    
    # Specialized sampler for training to handle class imbalance
    train_targets = [dataset.samples[i][1] for i in train_dataset.indices]
    sample_weights = [class_weights[t].item() for t in train_targets]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset), replacement=True)
    
    # Create data loaders
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Set CLIP model to evaluation mode (freezing it during training)
    clip_model.eval()
    
    # Use weighted loss for class imbalance
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Learning rate scheduler for adaptive learning
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # Early stopping setup
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        classifier.train()
        total_train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for texts, targets in train_loader:
            tokens = clip.tokenize(texts, truncate=True).to(device)
            optimizer.zero_grad()
            
            # Forward pass through CLIP text encoder (no gradients)
            with torch.no_grad():
                text_feats = clip_model.encode_text(tokens)
                text_feats /= text_feats.norm(dim=-1, keepdim=True)
            
            # Forward pass through our classifier
            logits = classifier(text_feats)
            loss = loss_fn(logits, targets.to(device))
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Track metrics
            total_train_loss += loss.item() * targets.size(0)
            _, predicted = torch.max(logits.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets.to(device)).sum().item()
        
        # Validation phase
        classifier.eval()
        total_val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Track per-class accuracy
        class_correct = {name: 0 for name, _ in CANDIDATE_COLORS}
        class_total = {name: 0 for name, _ in CANDIDATE_COLORS}
        
        with torch.no_grad():
            for texts, targets in val_loader:
                tokens = clip.tokenize(texts, truncate=True).to(device)
                text_feats = clip_model.encode_text(tokens)
                text_feats /= text_feats.norm(dim=-1, keepdim=True)
                logits = classifier(text_feats)
                loss = loss_fn(logits, targets.to(device))
                
                total_val_loss += loss.item() * targets.size(0)
                _, predicted = torch.max(logits.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets.to(device)).sum().item()
                
                # Update per-class metrics
                for i, target in enumerate(targets):
                    class_name = CANDIDATE_COLORS[target.item()][0]
                    class_total[class_name] += 1
                    if predicted[i] == target.to(device):
                        class_correct[class_name] += 1
        
        # Calculate epoch metrics
        avg_train_loss = total_train_loss / train_total
        avg_val_loss = total_val_loss / val_total
        train_accuracy = 100 * train_correct / train_total
        val_accuracy = 100 * val_correct / val_total
        
        # Log overall metrics
        logging.info(f"Epoch {epoch+1}/{epochs}: Train Loss = {avg_train_loss:.4f}, "
                    f"Val Loss = {avg_val_loss:.4f}, Train Acc = {train_accuracy:.2f}%, "
                    f"Val Acc = {val_accuracy:.2f}%")
        
        # Log per-class validation accuracies
        class_acc_log = "Per-class validation accuracies: "
        for name in class_correct:
            if class_total[name] > 0:
                acc = 100 * class_correct[name] / class_total[name]
                class_acc_log += f"{name}: {acc:.1f}%, "
        logging.info(class_acc_log)
        
        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = classifier.state_dict().copy()
            patience_counter = 0
            logging.info(f"New best model saved at epoch {epoch+1} with validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            logging.info(f"No improvement in validation loss. Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                logging.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Restore best model
    if best_state:
        classifier.load_state_dict(best_state)
        logging.info(f"Restored best model with validation loss: {best_val_loss:.4f}")
    
    return classifier

def find_json_files(directory):
    """Find all JSON files in a directory structure"""
    json_pattern = os.path.join(directory, "**", "*.json")
    json_files = glob.glob(json_pattern, recursive=True)
    return json_files

def main():
    logging.info("Starting enhanced polygon color prediction training")
    
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
    
    # Construct dataset with augmentation
    dataset = ShapesDataset(all_shapes, id_to_shape, augment=True)
    if len(dataset) == 0:
        logging.error("No training samples available. Check your JSON files!")
        sys.exit(1)
    
    # Calculate dynamic class weights
    class_weights = calculate_dynamic_weights(dataset)
    logging.info("Computed class weights: %s", class_weights)
    
    # Determine embedding dimension from CLIP
    with torch.no_grad():
        sample_tokens = clip.tokenize(["sample text"], truncate=True).to(device)
        sample_embedding = clip_model.encode_text(sample_tokens)
    embed_dim = sample_embedding.shape[-1]
    logging.info("CLIP text embedding dimension: %d", embed_dim)
    
    # Create enhanced classifier
    classifier = EnhancedColorClassifier(
        input_dim=embed_dim,
        hidden_dims=[256, 128],
        num_classes=NUM_CANDIDATES,
        dropout_rates=[0.4, 0.3]
    ).to(device)
    
    # Configure optimizer with weight decay for regularization
    optimizer = optim.AdamW(
        classifier.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # Train the model
    logging.info("Starting training with early stopping...")
    classifier = train_model(
        dataset=dataset,
        clip_model=clip_model,
        classifier=classifier,
        optimizer=optimizer,
        device=device,
        class_weights=class_weights,
        epochs=15,
        patience=3,
        batch_size=16
    )
    logging.info("Training complete.")
    
    # Save model weights
    weights_path = os.path.join(script_dir, "classifier_weights.pth")
    torch.save(classifier.state_dict(), weights_path)
    logging.info("Saved classifier weights to: %s", weights_path)
    
    # Save model architecture information for testing
    model_info = {
        "embed_dim": embed_dim,
        "hidden_dims": [256, 128],
        "num_classes": NUM_CANDIDATES,
        "dropout_rates": [0.4, 0.3]
    }
    info_path = os.path.join(script_dir, "model_info.json")
    with open(info_path, "w") as f:
        json.dump(model_info, f)
    logging.info("Saved model architecture information to: %s", info_path)

if __name__ == "__main__":
    main()