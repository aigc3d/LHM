#!/usr/bin/env python3
"""
Create Test Workflow for LHM Node in ComfyUI

This script generates a JSON workflow file that demonstrates the LHM node functionality
in ComfyUI. The workflow includes loading an image, processing it through the LHM node,
and properly displaying the results.

Usage:
    python create_test_workflow.py [output_path]

The script will create a test workflow and save it to the specified output_path
or to "lhm_test_workflow.json" in the current directory if no path is provided.
"""

import os
import json
import argparse
import uuid
from pathlib import Path

def generate_unique_id():
    """Generate a unique node ID for ComfyUI."""
    return str(uuid.uuid4())

def create_test_workflow(output_path="lhm_test_workflow.json"):
    """
    Create a test workflow for the LHM node in ComfyUI.
    
    Args:
        output_path: Path where the workflow JSON file will be saved
    """
    # Create unique IDs for each node
    load_image_id = generate_unique_id()
    lhm_node_id = generate_unique_id()
    preview_processed_id = generate_unique_id()
    reshape_node_id = generate_unique_id()
    preview_animation_id = generate_unique_id()
    
    # Create the workflow dictionary
    workflow = {
        "last_node_id": 5,
        "last_link_id": 5,
        "nodes": [
            {
                "id": load_image_id,
                "type": "LoadImage",
                "pos": [200, 200],
                "size": {"0": 315, "1": 102},
                "flags": {},
                "order": 0,
                "mode": 0,
                "outputs": [
                    {"name": "IMAGE", "type": "IMAGE", "links": [{"node": lhm_node_id, "slot": 0}]},
                    {"name": "MASK", "type": "MASK", "links": []},
                ],
                "properties": {"filename": "test_human.png"},
                "widgets_values": ["test_human.png"]
            },
            {
                "id": lhm_node_id,
                "type": "LHMReconstructionNode",
                "pos": [600, 200],
                "size": {"0": 315, "1": 178},
                "flags": {},
                "order": 1,
                "mode": 0,
                "inputs": [
                    {"name": "input_image", "type": "IMAGE", "link": 0}
                ],
                "outputs": [
                    {"name": "processed_image", "type": "IMAGE", "links": [{"node": preview_processed_id, "slot": 0}]},
                    {"name": "animation_frames", "type": "IMAGE", "links": [{"node": reshape_node_id, "slot": 0}]}
                ],
                "properties": {},
                "widgets_values": ["LHM-0.5B", False, True, True, 1.0]
            },
            {
                "id": preview_processed_id,
                "type": "PreviewImage",
                "pos": [1000, 100],
                "size": {"0": 210, "1": 246},
                "flags": {},
                "order": 2,
                "mode": 0,
                "inputs": [
                    {"name": "images", "type": "IMAGE", "link": 1}
                ],
                "properties": {},
                "widgets_values": []
            },
            {
                "id": reshape_node_id,
                "type": "TensorReshape",
                "pos": [1000, 350],
                "size": {"0": 315, "1": 82},
                "flags": {},
                "order": 3,
                "mode": 0,
                "inputs": [
                    {"name": "tensor", "type": "IMAGE", "link": 2}
                ],
                "outputs": [
                    {"name": "tensor", "type": "IMAGE", "links": [{"node": preview_animation_id, "slot": 0}]}
                ],
                "properties": {},
                "widgets_values": ["-1", "-1", "3"]
            },
            {
                "id": preview_animation_id,
                "type": "PreviewImage",
                "pos": [1300, 350],
                "size": {"0": 210, "1": 246},
                "flags": {},
                "order": 4,
                "mode": 0,
                "inputs": [
                    {"name": "images", "type": "IMAGE", "link": 3}
                ],
                "properties": {},
                "widgets_values": []
            }
        ],
        "links": [
            {"id": 0, "from_node": load_image_id, "from_output": 0, "to_node": lhm_node_id, "to_input": 0},
            {"id": 1, "from_node": lhm_node_id, "from_output": 0, "to_node": preview_processed_id, "to_input": 0},
            {"id": 2, "from_node": lhm_node_id, "from_output": 1, "to_node": reshape_node_id, "to_input": 0},
            {"id": 3, "from_node": reshape_node_id, "from_output": 0, "to_node": preview_animation_id, "to_input": 0}
        ],
        "groups": [],
        "config": {},
        "extra": {},
        "version": 0.4
    }
    
    # Save the workflow to a JSON file
    with open(output_path, 'w') as f:
        json.dump(workflow, f, indent=2)
    
    print(f"Test workflow created and saved to: {output_path}")
    print("Note: You may need to place a test image named 'test_human.png' in your ComfyUI input directory")

def main():
    parser = argparse.ArgumentParser(description="Create a test workflow for the LHM node in ComfyUI")
    parser.add_argument("output_path", nargs="?", default="lhm_test_workflow.json", 
                        help="Path where the workflow JSON file will be saved")
    args = parser.parse_args()
    
    create_test_workflow(args.output_path)

if __name__ == "__main__":
    main() 