# Polygon Color Prediction with CLIP

A deep learning system that predicts RGB colors for polygons based on text descriptions using OpenAI's CLIP model.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Testing](#testing)
- [JSON Format](#json-format)
- [Technical Details](#technical-details)

## Overview

This project implements a neural network model that can predict appropriate RGB colors for polygons based on their text descriptions. It utilizes OpenAI's CLIP (Contrastive Language-Image Pretraining) model to convert textual descriptions into embeddings, which are then used to predict RGB color values through a simple neural network.

The system processes JSON files containing polygon data (coordinates and descriptions) and generates images with appropriately colored shapes. It can be used for automatic coloring of vector graphics based on natural language descriptions.

## Project Structure

```
polygon-color-prediction/
├── __pycache__/           # Python cache directory
├── logs/                  # Directory for log files
├── results/               # Output images directory
├── shapes_jsons/          # Directory containing input JSON files
├── clip_to_rgb_model_info.json  # Model metadata
├── clip_to_rgb_model.pth  # Trained model weights
├── inst.bat               # Installation script for Windows
├── README.md              # This documentation file
├── script.py              # Script to run tests on all JSON files
├── test_polygons.py       # Script to test the model on a single JSON file
└── train_polygons.py      # Script to train the model
```

## Prerequisites

- Python 3.6 or higher
- Windows (for the batch installation script) or adapt the commands for your OS

## Installation

1. Clone this repository to your local machine.

2. Run the installation script to install all required dependencies:
   ```
   inst.bat
   ```

   This script installs:
   - PyTorch (CPU version)
   - NumPy
   - Pillow (PIL fork)
   - OpenAI's CLIP model

   If you're not using Windows, you can manually install the dependencies using pip:
   ```
   pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
   pip install numpy pillow
   pip install git+https://github.com/openai/CLIP.git
   ```

## Usage

### Training

The model has been pre-trained and the weights are stored in `clip_to_rgb_model.pth`. If you want to retrain the model with your own data or improve the existing model, you can run:

```
python train_polygons.py
```

This script:
1. Loads all JSON files from the `shapes_jsons` directory
2. Extracts polygon descriptions and their corresponding RGB values
3. Uses CLIP to convert text descriptions to embeddings
4. Trains a neural network to map these embeddings to RGB values
5. Saves the trained model to `clip_to_rgb_model.pth`

Training progress is logged to the console and to a log file in the `logs` directory.

### Testing

To generate images for all JSON files in the `shapes_jsons` directory:

```
python script.py
```

This script:
1. Finds all JSON files in the `shapes_jsons` directory
2. For each JSON file, calls `test_polygons.py` to generate an image
3. Saves the output images to the `results` directory
4. Reports success/failure for each file

To test a specific JSON file:

```
python test_polygons.py path/to/input.json path/to/output.png
```

## JSON Format

The input JSON files should contain an array of polygon objects with the following structure:

```json
[
  {
    "id": 0,
    "description": "A white background canvas for a simple face drawing.",
    "parent": null,
    "polygon": [
      [0, 0],
      [400, 0],
      [400, 400],
      [0, 400]
    ],
    "rgb": [255, 255, 255]
  },
  {
    "id": 1,
    "description": "A skin-colored circular face shape.",
    "parent": 0,
    "polygon": [
      [100, 75],
      [150, 50],
      /* more coordinates */
    ],
    "rgb": [255, 222, 173]
  }
  /* more polygons */
]
```

Each polygon object should include:
- `id`: Unique identifier for the polygon
- `description`: Text description of the polygon (used to predict color)
- `parent`: ID of the parent polygon (or `null` if none)
- `polygon`: Array of [x, y] coordinate pairs defining the polygon's shape
- `rgb`: RGB color value [r, g, b] (optional during testing, used during training)

## Technical Details

### Model Architecture

The system uses two main components:

1. **CLIP Model (ViT-B/32)**: Converts text descriptions into 512-dimensional embeddings
2. **Color Prediction Network**: A simple feed-forward neural network with:
   - Input layer: 512 neurons (CLIP embedding dimension)
   - Hidden layer: 256 neurons with ReLU activation
   - Output layer: 3 neurons (R, G, B) with Sigmoid activation to ensure values between 0-1

### Training Process

The model is trained using:
- AdamW optimizer with a learning rate of 0.001
- Mean Squared Error (MSE) loss function
- Cosine Annealing Warm Restarts learning rate scheduler
- Batch size of 16
- 200 epochs

### Prediction Process

During prediction:
1. The text description is encoded using CLIP to get an embedding
2. The embedding is normalized and passed through the color prediction network
3. The output is converted from [0,1] range to [0,255] range for RGB values
4. These colors are used to fill the polygons in the output image