# Model Card for LHM v0.1

## Overview

- This model card is for the [LHM](https://github.com/aigc3d/LHM) project, which is an official implementation of the paper [LHM](https://arxiv.org/pdf/2503.10625).
- Information contained in this model card corresponds to [Version 0.1](https://github.com/aigc3d/LHM).

## Model Details

- Training data

    | Model | Training Data | Training Strategy |
    | :---: | :---: | :---: |
    | [LHM-MINI](https://modelscope.cn/models/Damo_XR_Lab/LHM-500M) | 2K2K & RP & THuman + 300K Videos | Random Crop Body Size |
    | [LHM-500M](https://modelscope.cn/models/Damo_XR_Lab/LHM-500M) | 2K2K & RP & THuman + 300K Videos | Full Body |
    | [LHM-500M-HF](https://modelscope.cn/models/Damo_XR_Lab/LHM-500M-HF) | 2K2K & RP & THuman + 300K Videos | Random Crop Body Size |
    | [LHM-1B](https://modelscope.cn/models/Damo_XR_Lab/LHM-1B) | 2K2K & RP & THuman + 300K Videos | Full Body |
    | [LHM-1B-HF](https://modelscope.cn/models/Damo_XR_Lab/LHM-1B-HF) | 2K2K & RP & THuman + 300K Videos | Random Crop Body Size |


- Model architecture (version==0.3)

    | Type  | Layers | Feat. Dim | Attn. Heads | The number of GS Points. | Input Res. | Image Encoder     | Encoder Dim. | Service Requirement |
    | :---: | :----: | :-------: | :---------: | :-----------: | :--------: | :---------------: | :----------: | :---: |
    | LHM-MINI |  2 |    1024    |    16 |   20K |    512     | dinov2_vits14_reg & Sapiens-1B | 1024 | 16G GPU, 24G VRAM |
    | LHM-500M |  5 |    1024    |    16 |      40K |    512     | dinov2_vits14_reg & Sapiens-1B | 1024 | 18G GPU, 24G VRAM |
    | LHM-500M-HF  | 5 |    1024    |    16      |      40K       |    512 | dinov2_vitb14_reg & Sapiens-1B |      1024 | 18G GPU, 24G VRAM |
    | LHM-1B |   15 |   1024    |     16      |      40K |    1024 | dinov2_vitb14_reg & Sapiens-1B | 1024 | 22G GPU, 24G VRAM |
    | LHM-1B-HF |   15|   1024    |     16      |   40K |    1024 | dinov2_vitb14_reg & Sapiens-1B | 1024 | 22G GPU, 24G VRAM |

- Model architecture (with motion & save_memory; maximum supported length for 720P video is 20s)

    | Type  | Layers | Feat. Dim | Attn. Heads | The number of GS Points. | Input Res. | Image Encoder     | Encoder Dim. | Service Requirement |
    | :---: | :----: | :-------: | :---------: | :-----------: | :--------: | :---------------: | :----------: | :---: |
    | LHM-MINI |  2 |    1024    |    16 |   20K |    512     | dinov2_vits14_reg & Sapiens-1B | 1024 | 14G GPU, 24G VRAM |
    | LHM-500M |  5 |    1024    |    16 |      40K |    512     | dinov2_vits14_reg & Sapiens-1B | 1024 | 16G GPU, 24G VRAM |
    | LHM-500M-HF  | 5 |    1024    |    16      |      40K       |    512 | dinov2_vitb14_reg & Sapiens-1B |      1024 | 16 GPU, 24G VRAM |
    | LHM-1B |   15 |   1024    |     16      |      40K |    1024 | dinov2_vitb14_reg & Sapiens-1B | 1024 | 18G GPU, 24G VRAM |
    | LHM-1B-HF |   15|   1024    |     16      |   40K |    1024 | dinov2_vitb14_reg & Sapiens-1B | 1024 | 18G GPU, 24G VRAM |


- Model architecture (with motion)

    | Type  | Layers | Feat. Dim | Attn. Heads | The number of GS Points. | Input Res. | Image Encoder     | Encoder Dim. | Service Requirement |
    | :---: | :----: | :-------: | :---------: | :-----------: | :--------: | :---------------: | :----------: | :---: |
    | LHM-MINI |  2 |    1024    |    16 |   20K |    512     | dinov2_vits14_reg & Sapiens-1B | 1024 | 20G GPU, 24G VRAM |
    | LHM-500M |  5 |    1024    |    16 |      40K |    512     | dinov2_vits14_reg & Sapiens-1B | 1024 | 22G GPU, 24G VRAM |
    | LHM-500M-HF  | 5 |    1024    |    16      |      40K       |    512 | dinov2_vitb14_reg & Sapiens-1B |      1024 | 22 GPU, 24G VRAM |
    | LHM-1B |   15 |   1024    |     16      |      40K |    1024 | dinov2_vitb14_reg & Sapiens-1B | 1024 | 28G GPU, 24G VRAM |
    | LHM-1B-HF |   15|   1024    |     16      |   40K |    1024 | dinov2_vitb14_reg & Sapiens-1B | 1024 | 28G GPU, 24G VRAM |


## License

- The model weights are released under the [Creative Commons Attribution-NonCommercial 4.0 International License](LICENSE_WEIGHT).
- They are provided for research purposes only, and CANNOT be used commercially.