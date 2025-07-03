# Perceptron Image Classification Project

A NumPy-based perceptron implementation for binary image classification using PIL for image processing and matplotlib for visualization.

## Overview

This project implements a perceptron neural network model that can classify images into two categories (X and E). The model uses:
- **Sparse connection matrices** with excitatory and inhibitory connections
- **Zero-sum value matrices** for output classification
- **Error-based learning** with interleaved training
- **Image preprocessing** including grayscale conversion, resizing, and normalization

## Features

- Binary image classification using perceptron architecture
- Configurable network parameters (A-units, excitatory/inhibitory ratios, threshold)
- Training data size vs accuracy plotting
- PIL-based image preprocessing
- Memory-efficient float32 operations

## Requirements

Make sure you have Python 3.7+ installed. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Dependencies
- `numpy>=1.20.0` - Numerical computations and matrix operations
- `Pillow>=8.0.0` - Image loading, processing, and resizing
- `matplotlib>=3.3.0` - Plotting and visualization

## Dataset Setup

### If you have ZIP files:

1. **Extract your image datasets:**
   ```bash
   # For ZIP files
   unzip X_images.zip -d ./
   unzip E_images.zip -d ./
   
   # For tar.gz files
   tar -xzf X_images.tar.gz
   tar -xzf E_images.tar.gz
   
   # For 7z files (install p7zip-full first)
   7z x X_images.7z
   7z x E_images.7z
   ```

2. **Ensure your directory structure looks like:**
   ```
   /
   ├── perceptron.py
   ├── requirements.txt
   ├── README.md
   ├── .gitignore
   ├── X/                 # Directory containing X-class images
   │   ├── image1.jpg
   │   ├── image2.png
   │   └── ...
   └── E/                 # Directory containing E-class images
       ├── image1.jpg
       ├── image2.png
       └── ...
   ```

### Manual Setup:
If you don't have ZIP files, create the directories and add your images:
```bash
mkdir X E
# Copy your X-class images to X/ directory
# Copy your E-class images to E/ directory
```

## Usage

### Basic Usage

Run the complete training and evaluation:
```bash
python perceptron.py
```

This will:
1. Load and preprocess images from `X/` and `E/` directories
2. Train perceptron models with different training data sizes
3. Display a plot showing training size vs accuracy
4. Print a summary table of results

### Custom Training

You can modify the training parameters in the code:

```python
# Create a perceptron with custom parameters
model = perceptron(
    S_units=100,        # Input units (10x10 image = 100)
    A_units=1000,       # Association units
    R_units=2,          # Response units (binary classification)
    excitory=700,       # Excitatory connections
    inhibitory=300,     # Inhibitory connections
    threshold=0.2       # Activation threshold
)

# Train with your data
model.train([image_data], class_label)

# Make predictions
prediction = model.predict(test_image)
```

### Plotting Training Curves

Customize the training size analysis:

```python
# Test specific training sizes
sizes, accuracies = plot_training_size_vs_accuracy(
    X_path, E_path, X_imgs, E_imgs,
    sizes_to_test=[10, 25, 50, 100, 200],
    test_size=100,
    random_seed=42
)
```

## Model Architecture

- **S-units (Sensory)**: 100 units (10×10 flattened images)
- **A-units (Association)**: 1000 units with sparse connections
  - 700 excitatory connections (+1)
  - 300 inhibitory connections (-1)
- **R-units (Response)**: 2 units for binary classification
- **Activation**: Threshold-based (default: 0.2)
- **Learning**: Error-based weight updates

## Image Preprocessing

All images are automatically:
1. Converted to grayscale
2. Resized to 10×10 pixels using LANCZOS resampling
3. Normalized to [0, 1] range
4. Flattened to 100-dimensional vectors

## Troubleshooting

### Common Issues

1. **"No such file or directory" error:**
   - Ensure `X/` and `E/` directories exist
   - Check that directories contain image files

2. **Memory issues with large datasets:**
   - Reduce the number of images in directories
   - Use smaller training sizes in `plot_training_size_vs_accuracy()`

3. **Poor accuracy:**
   - Increase training data size
   - Adjust model parameters (A_units, threshold)
   - Ensure image quality is sufficient

4. **Matplotlib display issues:**
   ```bash
   # On Linux without display
   export MPLBACKEND=Agg
   
   # Or install GUI backend
   sudo apt-get install python3-tk
   ```

### Supported Image Formats
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)
- GIF (.gif)

## File Structure

```
paper/
├── perceptron.py           # Main implementation
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── .gitignore             # Git ignore patterns
├── X/                     # X-class images (excluded from git)
├── E/                     # E-class images (excluded from git)
└── plots/                 # Generated plots (optional)
```

## Development

### Adding New Features

The code is modular and easy to extend:

- **New preprocessing**: Modify `create_dataset()` function
- **Different architectures**: Adjust perceptron class parameters
- **Additional metrics**: Extend the evaluation functions
- **New visualizations**: Add plotting functions

### Performance Optimization

For larger datasets:
- Increase `A_units` for better representation capacity
- Adjust `excitory`/`inhibitory` ratio
- Modify learning rate in the training loop
- Use batch processing for training

## License

This project is for educational and research purposes.

## Contributing

Feel free to submit issues and enhancement requests!
