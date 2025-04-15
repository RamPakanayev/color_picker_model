# color_picker_model

This project contains a sophisticated polygon color prediction system that uses a trained neural network classifier built on top of OpenAI's CLIP text encoder. It analyzes textual descriptions of shapes and predicts appropriate colors for polygons defined in JSON files, producing composite images with the predicted colors.

## Features

- Advanced semantic color understanding with expanded synonym support
- Confidence-based color selection with smart fallback mechanisms
- Enhanced parent-child relationship handling for contextual color inheritance
- Adaptive data augmentation with class balancing for better training
- Hierarchical drawing with transparency control

## Directory Structure

```
color_picker_model/
├── __pycache__/
│   └── color_extraction.cpython-312.pyc
├── logs/                      # Log files directory
├── original_images/           # Sample reference images
│   ├── s1.png
│   └── s2.png
├── results/                   # Generated output images
├── shapes_jsons/              # Input polygon definitions
│   ├── abstract_art.json
│   ├── car.json
│   ├── color_conflict.json
│   ├── complex64.json
│   ├── cosmos.json
│   ├── cyberpunk.json
│   ├── face.json
│   ├── flags.json
│   ├── house.json
│   ├── landscape.json
│   ├── logo.json
│   ├── nested_shapes.json
│   ├── rgb_explicit_test.json
│   ├── semantic_colors.json
│   ├── shapes.json
│   └── shapes2.json
├── classifier_weights.pth     # Trained model weights
├── inst.bat                   # Dependency installation script
├── model_info.json            # Model architecture information
├── README.md                  # This file
├── script.py                  # Batch processing script
├── test_polygons.py           # Inference script
└── train_polygons.py          # Training script
```

## Installation

Run the included batch script to install all required dependencies:

```bash
.\inst.bat
```

This will install:
- PyTorch (CPU version by default)
- numpy (for numerical computations)
- pillow (for image processing - PIL fork)
- CLIP from OpenAI's GitHub repository (along with its dependencies)

If you're using a different environment or operating system, the core dependencies can be installed with:

```bash
pip install torch numpy pillow git+https://github.com/openai/CLIP.git
```

## Usage

### Training (Optional)

The repository already includes pre-trained weights (`classifier_weights.pth`), but if you want to retrain the model with your own data:

```bash
python train_polygons.py
```

This will:
1. Load and process all JSON files from the `shapes_jsons` directory
2. Train an enhanced classifier with dynamic class weighting and regularization
3. Save new weights to `classifier_weights.pth` and architecture information to `model_info.json`

### Inference

To run the model on test JSON files and generate colored images:

```bash
python script.py
```

The script will:
1. Process the JSON files specified in the script
2. Generate colored output images in the `results/` directory
3. Log detailed information to the `logs/` directory

### Manual Testing

To run the model on a specific JSON file:

```bash
python test_polygons.py [input_json_path] [output_image_path] [--confidence 0.3] [--debug]
```

Example:
```bash
python test_polygons.py shapes_jsons/cosmos.json results/output_cosmos.png
```

Parameters:
- `input_json_path`: Path to the input JSON file containing polygon data
- `output_image_path`: Path where the output image will be saved
- `--confidence`: (Optional) Confidence threshold for color selection (0.0-1.0)
- `--debug`: (Optional) Enable debug-level logging

## Input JSON Format

The model expects JSON files with the following structure:

```json
[
  {
    "id": 0,
    "description": "A black cosmic background of deep space",
    "polygon": [[0, 0], [800, 0], [800, 600], [0, 600]],
    "parent": null
  },
  {
    "id": 1,
    "description": "A glowing blue nebula",
    "polygon": [[200, 150], [600, 150], [500, 400], [300, 400]],
    "parent": 0
  }
]
```

Each shape must have:
- `id`: Unique identifier
- `description`: Textual description used to determine color
- `polygon`: Array of [x, y] coordinate pairs defining the shape
- `parent`: ID of the parent shape (or null)

The descriptions can include:
- Direct color words ("red", "blue", etc.)
- Color synonyms ("crimson", "azure", etc.)
- Semantic associations ("fire", "sky", "forest", etc.)
- Explicit RGB or hex values ("rgb(255, 0, 0)", "#FF0000")

## Logging

The system creates detailed logs in the `logs/` directory with timestamps, including:
- Shape processing details
- Color prediction confidences
- Parent-child relationship handling
- Any errors encountered

## Output

The output images show all the polygons filled with their predicted colors, drawn in hierarchical order (parents first, then children).

## Technical Details

The model architecture consists of:
- OpenAI's CLIP text encoder (frozen pre-trained weights)
- A custom neural network classifier with:
  - Multiple hidden layers with dropout
  - Enhanced regularization
  - Class weighting to handle imbalanced data

The color prediction process includes:
1. Parsing explicit colors from descriptions
2. Analyzing semantic color cues
3. Intelligent text merging of parent and child descriptions
4. Neural network classification with confidence-based selection
5. Parent color inheritance for low-confidence predictions

## License

[Include your license information here]

## Acknowledgments

- This project uses OpenAI's CLIP for text encoding
- PyTorch framework for neural network implementation