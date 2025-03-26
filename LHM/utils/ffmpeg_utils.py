import os
import pdb
import subprocess
import tempfile

import cv2
import imageio.v3 as iio
import numpy as np
import torch

VIDEO_TYPE_LIST = {'.avi','.mp4','.gif','.AVI','.MP4','.GIF'}

def encodeffmpeg(inputs, frame_rate, output, format="png"):
    """output: need video_name"""
    assert (
        os.path.splitext(output)[-1] in VIDEO_TYPE_LIST
    ), "output is the format of video, e.g., mp4"
    assert os.path.isdir(inputs), "input dir is NOT file format"

    inputs = inputs[:-1] if inputs[-1] == "/" else inputs

    output = os.path.abspath(output)

    cmd = (
        f"ffmpeg -r {frame_rate} -pattern_type glob -i '{inputs}/*.{format}' "
        + f'-vcodec libx264 -crf 10 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" '
        + f"-pix_fmt yuv420p {output} > /dev/null 2>&1"
    )

    print(cmd)

    output_dir = os.path.dirname(output)
    if os.path.exists(output):
        os.remove(output)
    os.makedirs(output_dir, exist_ok=True)

    print("encoding imgs to video.....")
    os.system(cmd)
    print("video done!")

def images_to_video(images, output_path, fps, gradio_codec: bool, verbose=False, bitrate="10M"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    frames = []
    for i in range(images.shape[0]):
        if isinstance(images, torch.Tensor):
            frame = (images[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            assert frame.shape[0] == images.shape[2] and frame.shape[1] == images.shape[3], \
                f"Frame shape mismatch: {frame.shape} vs {images.shape}"
            assert frame.min() >= 0 and frame.max() <= 255, \
                f"Frame value out of range: {frame.min()} ~ {frame.max()}"
        else:
            frame = images[i]
        frames.append(frame)
    
    frames = np.stack(frames)
    iio.imwrite(output_path,frames,fps=fps,codec="libx264",pixelformat="yuv420p",bitrate=bitrate,macro_block_size=16)


# def images_to_video(images, output_path, fps, gradio_codec: bool, verbose=False, bitrate="10M", batch_size=500):
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     temp_files = []
    
#     try:
#         for batch_idx in range(0, images.shape[0], batch_size):
#             batch = images[batch_idx:batch_idx + batch_size]
            
#             with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
#                 temp_path = temp_file.name
#                 temp_files.append(temp_path)

#             frames = []
#             for i in range(batch.shape[0]):
#                 if isinstance(batch, torch.Tensor):
#                     frame = (batch[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
#                     assert frame.shape[0] == batch.shape[2] and frame.shape[1] == batch.shape[3], \
#                         f"Frame shape mismatch: {frame.shape} vs {batch.shape}"
#                     assert 0 <= frame.min() and frame.max() <= 255, \
#                         f"Frame value out of range: {frame.min()} ~ {frame.max()}"
#                 else:
#                     frame = batch[i]
#                 frames.append(frame)
            
#             frames = np.stack(frames)
#             iio.imwrite(
#                 temp_path,
#                 frames,
#                 fps=fps,
#                 codec="libx264",
#                 pixelformat="yuv420p",
#                 bitrate=bitrate,
#                 macro_block_size=16
#             )
            
#             del batch, frames
#             if isinstance(images, torch.Tensor):
#                 torch.cuda.empty_cache()

#         _concat_videos(temp_files, output_path)

#     finally:
#         for f in temp_files:
#             try:
#                 os.remove(f)
#             except:
#                 pass


# def _concat_videos(input_files, output_path):
#     list_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
#     try:
#         content = "\n".join([f"file '{f}'" for f in input_files])
#         list_file.write(content)
#         list_file.close()

#         cmd = [
#             'ffmpeg', '-y', '-f', 'concat',
#             '-safe', '0', '-i', list_file.name,
#             '-c', 'copy', output_path
#         ]
#         subprocess.run(cmd, check=True)
#     finally:
#         os.remove(list_file.name)