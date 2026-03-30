#!/bin/bash
# Given pose sequence or video, generating animation video.
#
# Usage:
#   With pre-extracted motion params:
#     bash inference.sh LHM-1B ./train_data/example_imgs/ ./train_data/motion_video/mimo6/smplx_params/
#
#   With custom video input (video2motion runs automatically):
#     bash inference.sh LHM-1B ./train_data/example_imgs/ --video ./path/to/dance.mp4

MODEL_NAME=LHM-1B
IMAGE_INPUT="./train_data/example_imgs/"
MOTION_SEQS_DIR="./train_data/motion_video/mimo6/smplx_params/"

MODEL_NAME=${1:-$MODEL_NAME}
IMAGE_INPUT=${2:-$IMAGE_INPUT}

# Check if 3rd argument is --video flag for custom video input
if [ "$3" == "--video" ] && [ -n "$4" ]; then
    MOTION_VIDEO_PATH=$4
    echo "IMAGE_INPUT: $IMAGE_INPUT"
    echo "MODEL_NAME: $MODEL_NAME"
    echo "MOTION_VIDEO_PATH: $MOTION_VIDEO_PATH"
    echo "INFERENCE VIDEO (with video2motion)"

    MOTION_IMG_DIR=None
    VIS_MOTION=true
    MOTION_IMG_NEED_MASK=true
    RENDER_FPS=30
    MOTION_VIDEO_READ_FPS=30
    EXPORT_VIDEO=True

    python -m LHM.launch infer.human_lrm model_name=$MODEL_NAME \
            image_input=$IMAGE_INPUT \
            export_video=$EXPORT_VIDEO \
            motion_video_path=$MOTION_VIDEO_PATH \
            motion_seqs_dir=None motion_img_dir=$MOTION_IMG_DIR \
            vis_motion=$VIS_MOTION motion_img_need_mask=$MOTION_IMG_NEED_MASK \
            render_fps=$RENDER_FPS motion_video_read_fps=$MOTION_VIDEO_READ_FPS
else
    MOTION_SEQS_DIR=${3:-$MOTION_SEQS_DIR}
    echo "IMAGE_INPUT: $IMAGE_INPUT"
    echo "MODEL_NAME: $MODEL_NAME"
    echo "MOTION_SEQS_DIR: $MOTION_SEQS_DIR"
    echo "INFERENCE VIDEO"

    MOTION_IMG_DIR=None
    VIS_MOTION=true
    MOTION_IMG_NEED_MASK=true
    RENDER_FPS=30
    MOTION_VIDEO_READ_FPS=30
    EXPORT_VIDEO=True

    python -m LHM.launch infer.human_lrm model_name=$MODEL_NAME \
            image_input=$IMAGE_INPUT \
            export_video=$EXPORT_VIDEO \
            motion_seqs_dir=$MOTION_SEQS_DIR motion_img_dir=$MOTION_IMG_DIR  \
            vis_motion=$VIS_MOTION motion_img_need_mask=$MOTION_IMG_NEED_MASK \
            render_fps=$RENDER_FPS motion_video_read_fps=$MOTION_VIDEO_READ_FPS
fi
