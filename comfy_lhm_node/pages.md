---
layout: default
title: LHM ComfyUI Node
---

# Large Animatable Human Model (LHM) ComfyUI Node

![LHM Node Preview](./img/lhm_node_preview.png)

A specialized ComfyUI node that provides 3D human reconstruction and animation capabilities using the Large Animatable Human Model (LHM) framework.

## Features

- **Single-Image Reconstruction**: Generate 3D human models from a single image
- **Animation Support**: Apply motion sequences to the reconstructed human
- **Background Removal**: Automatically remove background from input images
- **Recentering**: Center the subject in the frame for better reconstruction
- **3D Mesh Export**: Generate and export 3D meshes for use in other applications
- **Progress Feedback**: Real-time progress tracking with visual indicators
- **Memory Management**: Smart resource handling for optimal performance

## Installation

```bash
# Clone the repository
git clone https://github.com/aigraphix/aigraphix.github.io.git
cd aigraphix.github.io

# Install ComfyUI node requirements
pip install -r comfy_lhm_node/requirements.txt

# Download model weights
./download_weights.sh
```

## Usage

1. **Load the node in ComfyUI**: The LHM node will appear in the "LHM" category
2. **Connect an image input**: Provide a single image of a person
3. **Configure options**:
   - Select model version (LHM-0.5B or LHM-1B)
   - Choose whether to remove background and recenter
   - Enable mesh export if needed
4. **Connect to outputs**: Use the processed image, animation sequence, or 3D mesh

## Example Workflow

We provide an [example workflow](./example_workflow.json) that demonstrates the node's capabilities:

![Example Workflow](./img/workflow_example.png)

To use it:
1. Open ComfyUI
2. Click "Load" in the menu
3. Select the example_workflow.json file
4. Replace the input image with your own

## Settings

The LHM node comes with customizable settings accessible through the ComfyUI settings panel:

- **Progress Bar Color**: Customize the appearance of progress indicators
- **Animation Preview FPS**: Set the frame rate for animation previews
- **Memory Optimization**: Balance between performance and memory usage
- **Auto-unload**: Automatically free resources when nodes are removed
- **Debug Mode**: Enable detailed logging for troubleshooting

## Troubleshooting

If you encounter issues:

1. **Model weights not found**: Ensure you've run the download_weights.sh script
2. **Out of memory errors**: Try using the LHM-0.5B model instead of LHM-1B
3. **Background removal issues**: Experiment with different preprocessing options
4. **Motion sequence errors**: Verify the motion_path points to valid motion data

## License

This project is licensed under the [Apache License 2.0](../LICENSE). 