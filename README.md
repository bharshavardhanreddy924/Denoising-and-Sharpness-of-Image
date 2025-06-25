# Advanced Image Signal Processing (ISP) Pipeline

## Project Overview
This project implements an advanced Image Signal Processing (ISP) pipeline focusing on sophisticated image enhancement techniques, specifically targeting denoising, sharpening, and comprehensive quality assessment for RAW images.

## Deep Learning Model Training

### Dataset
- **Name**: Smartphone Image Denoising Dataset (SIDD)
- **Training Configuration**:
  - Epochs: 3
  - Batch Size: 16
  - Learning Rate: 0.001

### Custom Denoising CNN Architecture
- 3-channel input
- 17 convolutional layers
- Base feature count: 64
- Utilizes BatchNorm and ReLU activations
- Noise estimation and subtraction approach

## Key Components

### Image Processing Pipeline
- **Demosaicing**: Edge-aware GRBG Bayer pattern conversion
- **White Balance**: Gray world algorithm 
- **Gamma Correction**: 12-bit to 8-bit transformation
- **Denoising Methods**:
  - Gaussian Filtering
  - Median Filtering
  - Bilateral Filtering
  - Deep Learning DnCNN

### Sharpening Techniques
- Unsharp Mask
- Laplacian Sharpening

## Quality Metrics
- Signal-to-Noise Ratio (SNR)
- Edge Strength
- Region of Interest (ROI) Analysis

1. Clone this repository:
   ```bash
   git clone "https://github.com/dattaaaaa/PixelFusion_Emmetra"
   open the Assignment 2 folder in terminal

2. Install the required Python dependencies:
    ```bash
   pip install -r requirements.txt
  
3. Run the Streamlit application:
   ```bash
    streamlit run app.py


## Prerequisites
- Python 3.8+
- Dependencies:
  ```
  torch
  torchvision
  opencv-python
  numpy
  streamlit
  rich
  reportlab
  ```

## Usage

### Training Denoising Model
```bash
python 1.py
```

### Streamlit Web Application
```bash
streamlit run app.py
```

## Project Structure
- `1.py`: Deep learning model training script
- `app.py`: Streamlit web interface
- `pipe.py`: Core ISP pipeline implementation

## Model Training Details
- **Dataset**: Smartphone Image Denoising Dataset (SIDD)
- **Training Configuration**:
  - Total Epochs: 3
  - Batch Size: 16
  - Learning Rate: 0.001
  - Optimizer: Adam
  - Loss Function: Mean Squared Error (MSE)

## Computational Requirements
- Recommended: CUDA-enabled GPU
- Minimum 16GB RAM
- High-performance CPU for CPU-based training

## Limitations
- Requires pre-trained DnCNN weights
- Performance varies with input image quality
- Computationally intensive for deep learning methods

## Future Enhancements
- Expand deep learning denoising models
- Implement adaptive processing algorithms
- Integrate more advanced sharpening techniques
