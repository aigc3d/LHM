# LHM Node for ComfyUI

A custom node for ComfyUI that integrates the Large Human Model (LHM) for 3D human reconstruction from a single image.

## Features

- Reconstruct 3D human avatars from a single image
- Generate animated sequences with the reconstructed avatar
- Background removal option
- Mesh export option for use in other 3D applications
- Preview scaling for faster testing
- Error handling with fallback to simplified implementation

## Installation

### Prerequisites

- ComfyUI installed and running
- Python 3.10+ with pip

### Installation Steps

1. Clone this repository into your ComfyUI custom_nodes directory:
   ```bash
   cd /path/to/ComfyUI/custom_nodes
   git clone https://github.com/aigraphix/comfy_lhm_node.git
   ```

2. Run the installation script:
   ```bash
   cd comfy_lhm_node
   chmod +x install_dependencies.sh
   ./install_dependencies.sh
   ```
   
   Alternatively, you can use the Python installation script:
   ```bash
   cd comfy_lhm_node
   chmod +x install_dependencies.py
   ./install_dependencies.py
   ```

3. Restart ComfyUI

### Optional: Using the Test Workflow

We've included a sample workflow to help you test the LHM node functionality:

1. Run the test workflow creation script:
   ```bash
   cd comfy_lhm_node
   chmod +x create_test_workflow.py
   ./create_test_workflow.py
   ```

2. Place a test image named `test_human.png` in your ComfyUI input directory
   
3. In ComfyUI, load the workflow by clicking on the Load button and selecting `lhm_test_workflow.json`

4. Click "Queue Prompt" to run the workflow

The test workflow includes:
- A LoadImage node that loads `test_human.png`
- The LHM Reconstruction node configured with recommended settings
- A TensorReshape node to format the animation output correctly
- Preview Image nodes to display both the processed image and animation frames

## Model Weights

The model weights are automatically downloaded the first time you run the node. If you encounter any issues with the automatic download, you can manually download the weights from:

- https://github.com/YuliangXiu/large-human-model

Place the weights in the `models` directory inside this node's folder.

## Troubleshooting

If you encounter any issues with the installation or running the node, check the [TROUBLESHOOTING.md](TROUBLESHOOTING.md) file for solutions to common problems.

## License

This project is licensed under the terms of the MIT license. See [LICENSE](LICENSE) for more details. 