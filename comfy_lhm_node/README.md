# LHM ComfyUI Node

This is a ComfyUI custom node for the LHM (Large Human Model) project, which enables human reconstruction and animation in ComfyUI workflows.

## Features

- Human reconstruction from single images
- Support for both LHM-0.5B and LHM-1B models
- Background removal and image preprocessing
- Motion sequence integration
- 3D mesh export
- ComfyUI workflow integration

## Installation

1. Clone the LHM repository:
```bash
git clone https://github.com/aigc3d/LHM.git
cd LHM
```

2. Install the ComfyUI node dependencies:
```bash
cd comfy_lhm_node
pip install -r requirements.txt
```

3. Copy the `comfy_lhm_node` directory to your ComfyUI's `custom_nodes` directory:
```bash
cp -r comfy_lhm_node /path/to/ComfyUI/custom_nodes/
```

4. Download the model weights:
```bash
# From the LHM root directory
bash download_weights.sh
```

## Usage

1. Launch ComfyUI
2. Look for the "LHM" category in the node menu
3. Add the "LHM Human Reconstruction" node to your workflow
4. Connect an image input to the node
5. Configure the node parameters:
   - Model Version: Choose between LHM-0.5B and LHM-1B
   - Motion Path: Path to SMPL-X motion parameters
   - Export Mesh: Enable/disable 3D mesh export
   - Remove Background: Enable/disable background removal
   - Recenter: Enable/disable image recentering

## Node Inputs

- `input_image`: Input image for human reconstruction
- `model_version`: LHM model version to use
- `motion_path`: Path to motion sequence parameters
- `export_mesh`: Whether to export 3D mesh
- `remove_background`: Whether to remove image background
- `recenter`: Whether to recenter the image

## Node Outputs

- `processed_image`: Preprocessed input image
- `animation`: Generated animation sequence
- `3d_mesh`: 3D mesh model (if export_mesh is enabled)

## Requirements

- Python 3.10+
- CUDA-compatible GPU
- ComfyUI
- See requirements.txt for full dependency list

## Troubleshooting

1. Ensure CUDA is properly installed and configured
2. Verify model weights are downloaded correctly
3. Check ComfyUI logs for any error messages
4. Make sure all dependencies are installed correctly

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details. 