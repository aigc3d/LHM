---
layout: default
title: Large Animatable Human Model (LHM)
---

# Large Animatable Human Model (LHM)

A framework for reconstructing 3D animatable humans from single images.

## Overview

LHM is an efficient method for reconstructing 3D human models from single images that can be animated with arbitrary motion sequences. It offers high-quality reconstruction with state-of-the-art performance.

![LHM Demo](./assets/teaser.gif)

## Features

- **Single Image Input**: Reconstruct 3D human models from just one image
- **Animation Support**: Apply various motion sequences to the reconstructed 3D model
- **ComfyUI Integration**: Use LHM directly in ComfyUI with our custom node
- **High-Quality Results**: State-of-the-art reconstruction quality

## ComfyUI Node

We provide a ComfyUI node for easy integration with the ComfyUI workflow. The node allows you to:

1. Input a single image of a person
2. Automatically remove the background and recenter
3. Generate a 3D reconstruction
4. Apply animation sequences
5. Export 3D meshes for further use

[Learn more about the ComfyUI node](./comfy_lhm_node/README.md)

## Installation

```bash
# Clone the repository
git clone https://github.com/aigraphix/aigraphix.github.io.git
cd aigraphix.github.io

# Install dependencies
pip install -r requirements.txt

# Download model weights
./download_weights.sh
```

## Example Usage

```python
from LHM.models.lhm import LHM
from PIL import Image
import torch

# Load model
model = LHM(img_size=512)
model.load_state_dict(torch.load("checkpoints/lhm-0.5b.pth"))
model.eval()

# Process image
image = Image.open("input_image.jpg")
results = model(image)

# Access results
reconstructed_image = results['processed_image']
animation = results['animation']
mesh = results['mesh']
```

## Example Workflow

Try our example workflow to see LHM in action:

1. Load an image of a person
2. Use the LHM node to reconstruct the 3D model
3. Apply different motion sequences
4. Export results

[Download Example Workflow](./comfy_lhm_node/example_workflow.json)

## Papers

If you find LHM useful, please cite our paper:

```bibtex
@article{lhm2023,
  title={Large Animatable Human Model},
  author={LHM Team},
  journal={arXiv preprint},
  year={2023}
}
```

## License

This project is licensed under the [Apache License 2.0](LICENSE). 