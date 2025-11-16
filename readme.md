# Face Processor for ComfyUI

A custom node collection for ComfyUI that provides advanced face detection, alignment, and transformation capabilities using MediaPipe Face Mesh.

## Features

- **Face Detection & Landmark Extraction**: Uses MediaPipe Face Mesh to detect and extract 468 facial landmarks
- **FacePipe Workflow**: `FaceFitAndRestore` prepares canonical crops while `FaceWrapper` unwraps and rewraps them using shared pipe metadata.
- **Torch-Based Deformation**: Triangle meshes are warped with the differentiable `TorchDeformer`, supporting CPU and CUDA automatically through PyTorch.
- **Restoration Helpers**: Optional Dlib refinement, canonical landmark models, and consistent frame tracking via `FrameData` and `FacePipe` utilities.
- **Utility Nodes**: Keep using helper nodes such as `ImageFeeder` or `HighPassFilter` to integrate with broader ComfyUI graphs.

## Installation

1. Clone this repository into your ComfyUI's `custom_nodes` directory:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/SykkoAtHome/ComfyUI_FaceProcessor.git face_processor
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
pip install opencv-python numpy pillow
```

PyTorch automatically selects CPU or CUDA execution depending on your environment.

## Nodes

### FaceFitAndRestore
Creates canonical crops (`Fit`) and restores processed faces back into their source frames (`Restore`). It produces a shared `fp_pipe` dictionary used across the workflow.

#### Inputs:
- `mode`: `Fit` or `Restore`
- `bbox_size`: Canonical crop resolution (512/1024/2048)
- `padding_percent`: Optional padding for crops
- `image`: Batch of frames (ComfyUI IMAGE)
- `image_paths`: Alternative string paths batch
- `fp_pipe`: (optional) pipeline dictionary when restoring

#### Outputs:
- `image`: Batched canonical crops or restored frames
- `mask`: Binary masks matching the output
- `fp_pipe`: Updated pipeline dictionary

### FaceWrapper
Transforms canonical crops between the aligned space and the original frame layout using the shared Torch deformer.

#### Inputs:
- `mode`: `UNWRAP` (canonical → unwrap size) or `WRAP` (unwrap size → original)
- `image`: Canonical crops or processed faces
- `unwrap_size`: Resolution for unwrap/warp operations (512/768/1024)
- `mask`: Optional masks accompanying the batch
- `fp_pipe`: Pipeline dictionary emitted by `FaceFitAndRestore`

#### Outputs:
- `image`: Warped batch in the requested space
- `mask`: Updated masks
- `fp_pipe`: Updated pipeline dictionary with stored metadata

## Recommended Workflow

The canonical FaceProcessor graph in ComfyUI follows this sequence:

1. **ImageFeeder** (optional convenience node)
2. **FaceFitAndRestore** with `mode=Fit` to produce canonical crops and an `fp_pipe` dictionary
3. **FaceWrapper** with `mode=UNWRAP` to enter the editable unwrap space
4. **Any custom face editing nodes** operating on the unwrapped images
5. **FaceWrapper** with `mode=WRAP` to project edited faces back to the source frame
6. **FaceFitAndRestore** with `mode=Restore` to composite the result into the original frame

The `workflow/FaceProcessor_basic.json` sample graph wires these nodes together with sensible defaults that you can import directly into ComfyUI.

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
- Torch-powered implementation that runs on CPU or CUDA depending on the active PyTorch device
- Handles both forward and inverse warping through the shared `TorchDeformer`

### Performance Considerations

- PyTorch automatically selects CPU or CUDA execution based on availability
- Progressive feedback during long operations
- Memory-efficient processing for large images

## Example Usage

Face normalization workflow:
```python
from faceprocessor.nodes_fit_restore import FaceFitAndRestore
from faceprocessor.nodes_wrapper import FaceWrapper

fit_node = FaceFitAndRestore()
wrapper = FaceWrapper()

faces, masks, fp_pipe = fit_node.process(
    mode="Fit",
    bbox_size="1024",
    padding_percent=0.05,
    image=input_image,
)

unwrap_faces, unwrap_masks, fp_pipe = wrapper.process(
    mode="UNWRAP",
    image=faces,
    unwrap_size="1024",
    mask=masks,
    fp_pipe=fp_pipe,
)

# ... run your edits on unwrap_faces ...
edited_faces = unwrap_faces

rewrapped, rewrap_masks, fp_pipe = wrapper.process(
    mode="WRAP",
    image=edited_faces,
    unwrap_size="1024",
    mask=unwrap_masks,
    fp_pipe=fp_pipe,
)

restored, restored_masks, _ = fit_node.process(
    mode="Restore",
    bbox_size="1024",
    padding_percent=0.0,
    image=rewrapped,
    fp_pipe=fp_pipe,
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