# Face Processor for ComfyUI

A custom node collection for ComfyUI that provides advanced face detection, alignment, and transformation capabilities using MediaPipe Face Mesh.

## Features

- **Face Detection & Landmark Extraction**: Uses MediaPipe Face Mesh to detect and extract 468 facial landmarks
- **Face Alignment**: Automatic face alignment based on eye positions
- **Face Transformation**: 
  - Warping between source and target face landmarks
  - Scale and translation controls for face shape adjustment
  - CPU and CUDA-accelerated processing options
- **Debug Visualization**: 
  - Visual landmark overlay with customizable parameters
  - Support for both detected and target landmark visualization
  - Optional landmark labels

## Installation

1. Clone this repository into your ComfyUI's `custom_nodes` directory:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/SykkoAtHome/ComfyUI_FaceProcessor.git face_processor
```

2. Install required dependencies:
```bash
pip install mediapipe opencv-python numpy pandas pillow torch
```

For CUDA acceleration:
1. Install [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
2. Install [CuPy](https://docs.cupy.dev/en/stable/install.html):
```bash
pip install cupy-cuda12x  # Replace with your CUDA version
```

## Nodes

### FaceWrapper
Main node for face detection and transformation operations.

#### Inputs:
- `image`: Input image (ComfyUI IMAGE type)
- `mode`: Operating mode
  - `Debug`: Visualization of detected landmarks
  - `Un-Wrap`: Transform face to normalized position
  - `Wrap`: Transform normalized face back to original position
- `device`: Processing device (`CPU` or `CUDA`)
- `show_detection`: Toggle detected landmarks visualization
- `show_target`: Toggle target landmarks visualization
- `landmark_size`: Size of landmark points in visualization
- `show_labels`: Toggle landmark index labels
- `x_scale`: Horizontal scaling factor (0.5 to 1.0)
- `y_transform`: Vertical translation (-0.5 to 0.5)
- `processor_settings`: Optional settings dictionary

#### Outputs:
- `image`: Processed image
- `processor_settings`: Updated settings dictionary

### FaceFitAndRestore
Node for face cropping and restoration operations.

#### Inputs:
- `mode`: Operating mode
  - `Fit`: Crop and align face
  - `Restore`: Place processed face back in original image
- `image`: Input image
- `padding_percent`: Additional padding around face (0.0 to 1.0)
- `bbox_size`: Output size for cropped face (512, 1024, or 2048)
- `processor_settings`: Required for Restore mode

#### Outputs:
- `image`: Processed image
- `processor_settings`: Updated settings dictionary
- `mask`: Mask indicating face region

## Technical Details

### Core Components

#### Face Detection
- Uses MediaPipe Face Mesh for robust face detection and landmark extraction
- Provides 468 facial landmarks with 3D coordinates
- Supports various input formats (PIL Image, numpy array, torch tensor)

#### Image Processing
- Automatic face rotation based on eye positions
- Aspect ratio-preserving resizing
- Support for square cropping with configurable padding
- Boundary triangulation for complete face warping

#### Face Warping
- Triangle-based warping using predefined mesh topology from MediaPipe Face Mesh
- CPU implementation using pure Python/NumPy
- CUDA-accelerated GPU implementation using CuPy
- Handles both forward and inverse warping

### Performance Considerations

- GPU acceleration requires CUDA toolkit and CuPy
- CPU fallback available for all operations
- Progressive feedback during long operations
- Memory-efficient processing for large images

## Example Usage

Basic face detection and visualization:
```python
face_wrapper = FaceWrapper()
result_image, settings = face_wrapper.detect_face(
    image=input_image,
    mode="Debug",
    device="CPU",
    show_detection=True,
    show_target=False,
    landmark_size=4,
    show_labels=True,
    x_scale=1.0,
    y_transform=0.0
)
```

Face normalization workflow:
1. Detect and normalize face:
```python
# Unwrap face to normalized position
normalized_face, settings = face_wrapper.detect_face(
    image=input_image,
    mode="Un-Wrap",
    device="CUDA",
    x_scale=1.0,
    y_transform=0.0
)
```

2. Process normalized face with your preferred method

3. Restore face to original position:

```python
# Wrap processed face back
final_image, _ = face_wrapper.detect_face(
    image=processed_face,
    mode="Wrap",
    device="CUDA",
    fp_pipe=settings
)
```

## License

MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Acknowledgments

- MediaPipe Face Mesh for facial landmark detection
- ComfyUI project for the node system framework