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
    | LHM-1B-HF |   15|   1024    |     16      |   40K |    1024 | dinov2_vitb14_reg & Sapiens-1B | 1024 | 222G GPU, 24G VRAM |

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

The project maintainer clarified in [issue #175](https://github.com/aigc3d/LHM/issues/175#issuecomment-4747846880) that the released LHM model weights are provided under Apache-2.0 and may be used commercially. The repository's source code is also licensed under Apache-2.0; see [`LICENSE`](LICENSE).

LHM is built upon SMPL-X and FLAME. Their model parameters are subject to separate, non-commercial terms unless separate commercial authorization is obtained from the corresponding rights holders. Commercial users are responsible for obtaining any necessary SMPL-X and/or FLAME licenses when their use case depends on those assets.
