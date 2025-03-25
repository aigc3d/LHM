# LHM ComfyUI Node

This is a ComfyUI custom node for the LHM (Large Animatable Human Model) project, which enables human reconstruction and animation in ComfyUI workflows.

## Features

- Human reconstruction from single images
- Support for both LHM-0.5B and LHM-1B models
- Background removal and image preprocessing
- Motion sequence integration
- 3D mesh export
- ComfyUI workflow integration
- Progress feedback with real-time status updates

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

## Example Workflow

We've included an example workflow in the `example_workflow.json` file that demonstrates how to use the LHM node. To use it:

1. Open ComfyUI
2. Click on the "Load" button in the top menubar
3. Navigate to `/path/to/ComfyUI/custom_nodes/comfy_lhm_node/example_workflow.json`
4. Load the workflow

The example workflow:
- Loads an input image
- Processes it with the LHM node
- Displays the processed image
- Combines the animation frames into a video
- Displays the animation

**Note:** The example workflow requires the VHS video extensions for ComfyUI to be installed. If you don't have them, you can still use the node by connecting the `processed_image` output to a `PreviewImage` node.

## Client-Side Extensions

The node includes client-side extensions that provide:

- Real-time progress updates during processing
- Custom styling for the node in the UI
- Improved labels for boolean switches

## Requirements

- Python 3.10+
- CUDA-compatible GPU
- ComfyUI
- See requirements.txt for full dependency list

## Troubleshooting

1. **Issue**: "Module not found" error
   - Make sure all dependencies are installed
   - Check that the LHM project is properly added to Python path

2. **Issue**: "Model weights not found" error
   - Run `download_weights.sh` from the LHM root directory
   - Check the model path is correct

3. **Issue**: Node not showing up in ComfyUI
   - Restart ComfyUI after installation
   - Check that the node is properly copied to the custom_nodes directory

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details. 