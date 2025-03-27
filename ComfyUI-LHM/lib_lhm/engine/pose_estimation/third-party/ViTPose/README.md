<h1 align="left">ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation<a href="https://arxiv.org/abs/2204.12484"><img src="https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg" ></a> </h1> 

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vitpose-simple-vision-transformer-baselines/pose-estimation-on-coco-test-dev)](https://paperswithcode.com/sota/pose-estimation-on-coco-test-dev?p=vitpose-simple-vision-transformer-baselines)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vitpose-simple-vision-transformer-baselines/pose-estimation-on-aic)](https://paperswithcode.com/sota/pose-estimation-on-aic?p=vitpose-simple-vision-transformer-baselines)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vitpose-simple-vision-transformer-baselines/pose-estimation-on-crowdpose)](https://paperswithcode.com/sota/pose-estimation-on-crowdpose?p=vitpose-simple-vision-transformer-baselines)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vitpose-simple-vision-transformer-baselines/pose-estimation-on-ochuman)](https://paperswithcode.com/sota/pose-estimation-on-ochuman?p=vitpose-simple-vision-transformer-baselines)

<p align="center">
  <a href="#Results">Results</a> |
  <a href="#Updates">Updates</a> |
  <a href="#Usage">Usage</a> |
  <a href='#Todo'>Todo</a> |
  <a href="#Acknowledge">Acknowledge</a>
</p>

<p align="center">
<a href="https://giphy.com/gifs/UfPQB1qKir7Vqem6sL/fullscreen"><img src="https://media.giphy.com/media/ZewXwZuixYKS2lZmNL/giphy.gif"></a>   <a href="https://giphy.com/gifs/DCvf1DrWZgbwPa8bWZ/fullscreen"><img src="https://media.giphy.com/media/2AEeuicbIjwqp2mbug/giphy.gif"></a>
</p>
<p align="center">
<a href="https://giphy.com/gifs/r3GaZz7H1H6zpuIvPI/fullscreen"><img src="https://media.giphy.com/media/13oe6zo6b2B7CdsOac/giphy.gif"></a>    <a href="https://giphy.com/gifs/FjzrGJxsOzZAXaW7Vi/fullscreen"><img src="https://media.giphy.com/media/4JLERHxOEgH0tt5DZO/giphy.gif"></a>
</p>

This branch contains the pytorch implementation of <a href="https://arxiv.org/abs/2204.12484">ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation</a> and <a href="https://arxiv.org/abs/2212.04246">ViTPose+: Vision Transformer Foundation Model for Generic Body Pose Estimation</a>. It obtains 81.1 AP on MS COCO Keypoint test-dev set.

<img src="figures/Throughput.png" class="left" width='80%'>

## Web Demo

- Integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). Try out the Web Demo for video: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/hysts/ViTPose_video) and images [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Gradio-Blocks/ViTPose)

## MAE Pre-trained model

- The small size MAE pre-trained model can be found in [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgccZeiFjh4DJ7gjYyg?e=iTMdMq). 
- The base, large, and huge pre-trained models using MAE can be found in the [MAE official repo](https://github.com/facebookresearch/mae).

## Results from this repo on MS COCO val set (single-task training)

Using detection results from a detector that obtains 56 mAP on person. The configs here are for both training and test.

> With classic decoder

| Model | Pretrain | Resolution | AP | AR | config | log | weight |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| ViTPose-S | MAE | 256x192 | 73.8 | 79.2 | [config](configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_small_coco_256x192.py) | [log](https://1drv.ms/u/s!AimBgYV7JjTlgcchdNXBAh7ClS14pA?e=dKXmJ6) | [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgccifT1XlGRatxg3vw?e=9wz7BY) |
| ViTPose-B | MAE | 256x192 | 75.8 | 81.1 | [config](configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py) | [log](logs/vitpose-b.log.json) | [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgSMjp1_NrV3VRSmK?e=Q1uZKs) |
| ViTPose-L | MAE | 256x192 | 78.3 | 83.5 | [config](configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_large_coco_256x192.py) | [log](logs/vitpose-l.log.json) | [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgSd9k_kuktPtiP4F?e=K7DGYT) |
| ViTPose-H | MAE | 256x192 | 79.1 | 84.1 | [config](configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py) | [log](logs/vitpose-h.log.json) | [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgShLMI-kkmvNfF_h?e=dEhGHe) |

> With simple decoder

| Model | Pretrain | Resolution | AP | AR | config | log | weight |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| ViTPose-S | MAE | 256x192 | 73.5 | 78.9 | [config](configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_small_simple_coco_256x192.py) | [log](https://1drv.ms/u/s!AimBgYV7JjTlgccfkqELJqE67kpRtw?e=InSjJP) | [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgccgb_50jIgiYkHvdw?e=D7RbH2) |
| ViTPose-B | MAE | 256x192 | 75.5 | 80.9 | [config](configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_simple_coco_256x192.py) | [log](logs/vitpose-b-simple.log.json) | [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgSRPKrD5PmDRiv0R?e=jifvOe) |
| ViTPose-L | MAE | 256x192 | 78.2 | 83.4 | [config](configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_large_simple_coco_256x192.py) | [log](logs/vitpose-l-simple.log.json) | [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgSVS6DP2LmKwZ3sm?e=MmCvDT) |
| ViTPose-H | MAE | 256x192 | 78.9 | 84.0 | [config](configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_simple_coco_256x192.py) | [log](logs/vitpose-h-simple.log.json) | [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgSbHyN2mjh2n2LyG?e=y0FgMK) |


## Results with multi-task training

**Note** \* There may exist duplicate images in the crowdpose training set and the validation images in other datasets, as discussed in [issue #24](https://github.com/ViTAE-Transformer/ViTPose/issues/24). Please be careful when using these models for evaluation. We provide the results without the crowpose dataset for reference.

### Human datasets (MS COCO, AIC, MPII, CrowdPose)
> Results on MS COCO val set

Using detection results from a detector that obtains 56 mAP on person. Note the configs here are only for evaluation.

| Model | Dataset | Resolution | AP | AR | config | weight |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| ViTPose-B | COCO+AIC+MPII | 256x192 | 77.1 | 82.2 | [config](configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py)  | [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgcccwaTZ8xCFFM3Sjg?e=chmiK5) |
| ViTPose-L | COCO+AIC+MPII | 256x192 | 78.7 | 83.8 | [config](configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_large_coco_256x192.py) | [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgccdOLQqSo6E87GfMw?e=TEurgW) |
| ViTPose-H | COCO+AIC+MPII | 256x192 | 79.5 | 84.5 | [config](configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py) | [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgccmHofkmfJDQDukVw?e=gRK224) |
| ViTPose-G | COCO+AIC+MPII | 576x432 | 81.0 | 85.6 | | |
| ViTPose-B* | COCO+AIC+MPII+CrowdPose | 256x192 | 77.5 | 82.6 | [config](configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py)  |[Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgSrlMB093JzJtqq-?e=Jr5S3R) |
| ViTPose-L* | COCO+AIC+MPII+CrowdPose | 256x192 | 79.1 | 84.1 | [config](configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_large_coco_256x192.py) | [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgTBm3dCVmBUbHYT6?e=fHUrTq) |
| ViTPose-H* | COCO+AIC+MPII+CrowdPose | 256x192 | 79.8 | 84.8 | [config](configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py) | [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgS5rLeRAJiWobCdh?e=41GsDd) |
| **ViTPose+-S** | COCO+AIC+MPII+AP10K+APT36K+WholeBody | 256x192 | 75.8 | 82.6 | [config](configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vitPose+_small_coco+aic+mpii+ap10k+apt36k+wholebody_256x192_udp.py)  | [log](https://1drv.ms/u/s!AimBgYV7JjTlgccqO1JBHtBjNaeCbQ?e=ZN5NSz) \| [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgccrwORr61gT9E4n8g?e=kz9sz5) |
| **ViTPose+-B** | COCO+AIC+MPII+AP10K+APT36K+WholeBody | 256x192 | 77.0 | 82.6 | [config](configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vitPose+_base_coco+aic+mpii+ap10k+apt36k+wholebody_256x192_udp.py)  | [log](https://1drv.ms/u/s!AimBgYV7JjTlgccjj9lgPTlkGT1HTw?e=OlS5zv) \| [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgcckRZk1bIAuRa_E1w?e=ylDB2G) |
| **ViTPose+-L** | COCO+AIC+MPII+AP10K+APT36K+WholeBody | 256x192 | 78.6 | 84.1 | [config](configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vitPose+_large_coco+aic+mpii+ap10k+apt36k+wholebody_256x192_udp.py) | [log](https://1drv.ms/u/s!AimBgYV7JjTlgccp7HJf4QMeQQpeyA?e=JagPNt) \| [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgccs1SNFUGSTsmRJ8w?e=a9zKwZ) |
| **ViTPose+-H** | COCO+AIC+MPII+AP10K+APT36K+WholeBody | 256x192 | 79.4 | 84.8 | [config](configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vitPose+_huge_coco+aic+mpii+ap10k+apt36k+wholebody_256x192_udp.py) | [log](https://1drv.ms/u/s!AimBgYV7JjTlgcclxZOlwRJdqpIIjA?e=nFQgVC) \| [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgccoXv8rCUgVe7oD9Q?e=ZBw6gR) |


> Results on OCHuman test set

Using groundtruth bounding boxes. Note the configs here are only for evaluation.

| Model | Dataset | Resolution | AP | AR | config | weight |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| ViTPose-B | COCO+AIC+MPII | 256x192 | 88.0 | 89.6 | [config](configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/ochuman/ViTPose_base_ochuman_256x192.py)  | [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgcccwaTZ8xCFFM3Sjg?e=chmiK5) |
| ViTPose-L | COCO+AIC+MPII | 256x192 | 90.9 | 92.2 | [config](configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/ochuman/ViTPose_large_ochuman_256x192.py) | [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgccdOLQqSo6E87GfMw?e=TEurgW) |
| ViTPose-H | COCO+AIC+MPII | 256x192 | 90.9 | 92.3 | [config](configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/ochuman/ViTPose_huge_ochuman_256x192.py) | [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgccmHofkmfJDQDukVw?e=gRK224) |
| ViTPose-G | COCO+AIC+MPII | 576x432 | 93.3 | 94.3 | | |
| ViTPose-B* | COCO+AIC+MPII+CrowdPose | 256x192 | 88.2 | 90.0 | [config](configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/ochuman/ViTPose_base_ochuman_256x192.py)  |[Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgSrlMB093JzJtqq-?e=Jr5S3R) |
| ViTPose-L* | COCO+AIC+MPII+CrowdPose | 256x192 | 91.5 | 92.8 | [config](configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/ochuman/ViTPose_large_ochuman_256x192.py) | [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgTBm3dCVmBUbHYT6?e=fHUrTq) |
| ViTPose-H* | COCO+AIC+MPII+CrowdPose | 256x192 | 91.6 | 92.8 | [config](configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/ochuman/ViTPose_huge_ochuman_256x192.py) | [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgS5rLeRAJiWobCdh?e=41GsDd) |
| **ViTPose+-S** | COCO+AIC+MPII+AP10K+APT36K+WholeBody | 256x192 | 78.4 | 80.6 | [config](configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/ochuman/ViTPose_small_ochuman_256x192.py)  | [log](https://1drv.ms/u/s!AimBgYV7JjTlgccqO1JBHtBjNaeCbQ?e=ZN5NSz) \| [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgccrwORr61gT9E4n8g?e=kz9sz5) |
| **ViTPose+-B** | COCO+AIC+MPII+AP10K+APT36K+WholeBody | 256x192 | 82.6 | 84.8 | [config](configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/ochuman/ViTPose_base_ochuman_256x192.py)  | [log](https://1drv.ms/u/s!AimBgYV7JjTlgccjj9lgPTlkGT1HTw?e=OlS5zv) \| [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgcckRZk1bIAuRa_E1w?e=ylDB2G) |
| **ViTPose+-L** | COCO+AIC+MPII+AP10K+APT36K+WholeBody | 256x192 | 85.7 | 87.5 | [config](configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/ochuman/ViTPose_large_ochuman_256x192.py) | [log](https://1drv.ms/u/s!AimBgYV7JjTlgccp7HJf4QMeQQpeyA?e=JagPNt) \| [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgccs1SNFUGSTsmRJ8w?e=a9zKwZ) |
| **ViTPose+-H** | COCO+AIC+MPII+AP10K+APT36K+WholeBody | 256x192 | 85.7 | 87.4 | [config](configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/ochuman/ViTPose_huge_ochuman_256x192.py) | [log](https://1drv.ms/u/s!AimBgYV7JjTlgcclxZOlwRJdqpIIjA?e=nFQgVC) \| [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgccoXv8rCUgVe7oD9Q?e=ZBw6gR) |

> Results on MPII val set

Using groundtruth bounding boxes. Note the configs here are only for evaluation. The metric is PCKh.

| Model | Dataset | Resolution | Mean | config | weight |
| :----: | :----: | :----: | :----: | :----: | :----: |
| ViTPose-B | COCO+AIC+MPII | 256x192 | 93.3 | [config](configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/ViTPose_base_mpii_256x192.py)  | [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgcccwaTZ8xCFFM3Sjg?e=chmiK5) |
| ViTPose-L | COCO+AIC+MPII | 256x192 | 94.0 | [config](configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/ViTPose_large_mpii_256x192.py) | [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgccdOLQqSo6E87GfMw?e=TEurgW) |
| ViTPose-H | COCO+AIC+MPII | 256x192 | 94.1 | [config](configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/ViTPose_huge_mpii_256x192.py) | [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgccmHofkmfJDQDukVw?e=gRK224) |
| ViTPose-G | COCO+AIC+MPII | 576x432 | 94.3 | | |
| ViTPose-B* | COCO+AIC+MPII+CrowdPose | 256x192 | 93.4 | [config](configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/ViTPose_base_mpii_256x192.py)  |[Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgSy_OSEm906wd2LB?e=GOSg14) |
| ViTPose-L* | COCO+AIC+MPII+CrowdPose | 256x192 | 93.9 | [config](configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/ViTPose_large_mpii_256x192.py) | [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgTM32I6Kpjr-esl6?e=qvh0Yl) |
| ViTPose-H* | COCO+AIC+MPII+CrowdPose | 256x192 | 94.1 | [config](configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/ViTPose_huge_mpii_256x192.py) | [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgTT90XEQBKy-scIH?e=D2WhTS) |
| **ViTPose+-S** | COCO+AIC+MPII+AP10K+APT36K+WholeBody | 256x192 | 92.7 | [config](configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/ViTPose_small_mpii_256x192.py)  | [log](https://1drv.ms/u/s!AimBgYV7JjTlgccqO1JBHtBjNaeCbQ?e=ZN5NSz) \| [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgccrwORr61gT9E4n8g?e=kz9sz5) |
| **ViTPose+-B** | COCO+AIC+MPII+AP10K+APT36K+WholeBody | 256x192 | 92.8 | [config](configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/ViTPose_base_mpii_256x192.py)  | [log](https://1drv.ms/u/s!AimBgYV7JjTlgccjj9lgPTlkGT1HTw?e=OlS5zv) \| [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgcckRZk1bIAuRa_E1w?e=ylDB2G) |
| **ViTPose+-L** | COCO+AIC+MPII+AP10K+APT36K+WholeBody | 256x192 | 94.0 | [config](configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/ViTPose_large_mpii_256x192.py) | [log](https://1drv.ms/u/s!AimBgYV7JjTlgccp7HJf4QMeQQpeyA?e=JagPNt) \| [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgccs1SNFUGSTsmRJ8w?e=a9zKwZ) |
| **ViTPose+-H** | COCO+AIC+MPII+AP10K+APT36K+WholeBody | 256x192 | 94.2 | [config](configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/ViTPose_huge_mpii_256x192.py) | [log](https://1drv.ms/u/s!AimBgYV7JjTlgcclxZOlwRJdqpIIjA?e=nFQgVC) \| [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgccoXv8rCUgVe7oD9Q?e=ZBw6gR) |


> Results on AI Challenger test set

Using groundtruth bounding boxes. Note the configs here are only for evaluation.

| Model | Dataset | Resolution | AP | AR | config | weight |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| ViTPose-B | COCO+AIC+MPII | 256x192 | 32.0 | 36.3 | [config](configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/aic/ViTPose_base_aic_256x192.py)  | [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgcccwaTZ8xCFFM3Sjg?e=chmiK5) |
| ViTPose-L | COCO+AIC+MPII | 256x192 | 34.5 | 39.0 | [config](configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/aic/ViTPose_large_aic_256x192.py) | [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgccdOLQqSo6E87GfMw?e=TEurgW) |
| ViTPose-H | COCO+AIC+MPII | 256x192 | 35.4 | 39.9 | [config](configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/aic/ViTPose_huge_aic_256x192.py) | [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgccmHofkmfJDQDukVw?e=gRK224) |
| ViTPose-G | COCO+AIC+MPII | 576x432 | 43.2 | 47.1 | | |
| ViTPose-B* | COCO+AIC+MPII+CrowdPose | 256x192 | 31.9 | 36.3 | [config](configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/aic/ViTPose_base_aic_256x192.py)  |[Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgSlvdVaXTC92SHYH?e=j7iqcp) |
| ViTPose-L* | COCO+AIC+MPII+CrowdPose | 256x192 | 34.6 | 39.0 | [config](configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/aic/ViTPose_large_aic_256x192.py) | [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgTF06FX3FSAm0MOH?e=rYts9F) |
| ViTPose-H* | COCO+AIC+MPII+CrowdPose | 256x192 | 35.3 | 39.8 | [config](configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/aic/ViTPose_huge_aic_256x192.py) | [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgS1MRmb2mcow_K04?e=q9jPab) |
| **ViTPose+-S** | COCO+AIC+MPII+AP10K+APT36K+WholeBody | 256x192 | 29.7 | 34.3 | [config](configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/ochuman/ViTPose_small_ochuman_256x192.py)  | [log](https://1drv.ms/u/s!AimBgYV7JjTlgccqO1JBHtBjNaeCbQ?e=ZN5NSz) \| [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgccrwORr61gT9E4n8g?e=kz9sz5) |
| **ViTPose+-B** | COCO+AIC+MPII+AP10K+APT36K+WholeBody | 256x192 | 31.8 | 36.3 | [config](configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/ochuman/ViTPose_base_ochuman_256x192.py)  | [log](https://1drv.ms/u/s!AimBgYV7JjTlgccjj9lgPTlkGT1HTw?e=OlS5zv) \| [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgcckRZk1bIAuRa_E1w?e=ylDB2G) |
| **ViTPose+-L** | COCO+AIC+MPII+AP10K+APT36K+WholeBody | 256x192 | 34.3 | 38.9 | [config](configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/ochuman/ViTPose_large_ochuman_256x192.py) | [log](https://1drv.ms/u/s!AimBgYV7JjTlgccp7HJf4QMeQQpeyA?e=JagPNt) \| [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgccs1SNFUGSTsmRJ8w?e=a9zKwZ) |
| **ViTPose+-H** | COCO+AIC+MPII+AP10K+APT36K+WholeBody | 256x192 | 34.8 | 39.1 | [config](configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/ochuman/ViTPose_huge_ochuman_256x192.py) | [log](https://1drv.ms/u/s!AimBgYV7JjTlgcclxZOlwRJdqpIIjA?e=nFQgVC) \| [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgccoXv8rCUgVe7oD9Q?e=ZBw6gR) |

> Results on CrowdPose test set

Using YOLOv3 human detector. Note the configs here are only for evaluation.

| Model | Dataset | Resolution | AP | AP(H) | config | weight |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| ViTPose-B* | COCO+AIC+MPII+CrowdPose | 256x192 | 74.7 | 63.3 | [config](configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/crowdpose/ViTPose_base_crowdpose_256x192.py)  |[Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgStrrCb91cPlaxJx?e=6Xobo6) |
| ViTPose-L* | COCO+AIC+MPII+CrowdPose | 256x192 | 76.6 | 65.9 | [config](configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/crowdpose/ViTPose_large_crowdpose_256x192.py) | [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgTK3dug-r7c6GFyu?e=1ZBpEG) |
| ViTPose-H* | COCO+AIC+MPII+CrowdPose | 256x192 | 76.3 | 65.6 | [config](configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/crowdpose/ViTPose_huge_crowdpose_256x192.py) | [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgS-oAvEV4MTD--Xr?e=EeW2Fu) |

### Animal datasets (AP10K, APT36K)

> Results on AP-10K test set

| Model | Dataset | Resolution | AP | config | weight |
| :----: | :----: | :----: | :----: | :----: | :----: |
| **ViTPose+-S** | COCO+AIC+MPII+AP10K+APT36K+WholeBody | 256x192 | 71.4 | [config](configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/ViTPose_small_ap10k_256x192.py)  | [log](https://1drv.ms/u/s!AimBgYV7JjTlgccqO1JBHtBjNaeCbQ?e=ZN5NSz) \| [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgccrwORr61gT9E4n8g?e=kz9sz5) |
| **ViTPose+-B** | COCO+AIC+MPII+AP10K+APT36K+WholeBody | 256x192 | 74.5 | [config](configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/ViTPose_base_ap10k_256x192.py)  | [log](https://1drv.ms/u/s!AimBgYV7JjTlgccjj9lgPTlkGT1HTw?e=OlS5zv) \| [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgcckRZk1bIAuRa_E1w?e=ylDB2G) |
| **ViTPose+-L** | COCO+AIC+MPII+AP10K+APT36K+WholeBody | 256x192 | 80.4 | [config](configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/ViTPose_large_ap10k_256x192.py) | [log](https://1drv.ms/u/s!AimBgYV7JjTlgccp7HJf4QMeQQpeyA?e=JagPNt) \| [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgccs1SNFUGSTsmRJ8w?e=a9zKwZ) |
| **ViTPose+-H** | COCO+AIC+MPII+AP10K+APT36K+WholeBody | 256x192 | 82.4 | [config](configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/ViTPose_huge_ap10k_256x192.py) | [log](https://1drv.ms/u/s!AimBgYV7JjTlgcclxZOlwRJdqpIIjA?e=nFQgVC) \| [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgccoXv8rCUgVe7oD9Q?e=ZBw6gR) |

> Results on APT-36K val set

| Model | Dataset | Resolution | AP | config | weight |
| :----: | :----: | :----: | :----: | :----: | :----: |
| **ViTPose+-S** | COCO+AIC+MPII+AP10K+APT36K+WholeBody | 256x192 | 74.2 | [config](configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/apt36k/ViTPose_small_apt36k_256x192.py)  | [log](https://1drv.ms/u/s!AimBgYV7JjTlgccqO1JBHtBjNaeCbQ?e=ZN5NSz) \| [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgccrwORr61gT9E4n8g?e=kz9sz5) |
| **ViTPose+-B** | COCO+AIC+MPII+AP10K+APT36K+WholeBody | 256x192 | 75.9 | [config](configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/apt36k/ViTPose_base_apt36k_256x192.py)  | [log](https://1drv.ms/u/s!AimBgYV7JjTlgccjj9lgPTlkGT1HTw?e=OlS5zv) \| [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgcckRZk1bIAuRa_E1w?e=ylDB2G) |
| **ViTPose+-L** | COCO+AIC+MPII+AP10K+APT36K+WholeBody | 256x192 | 80.8 | [config](configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/apt36k/ViTPose_large_apt36k_256x192.py) | [log](https://1drv.ms/u/s!AimBgYV7JjTlgccp7HJf4QMeQQpeyA?e=JagPNt) \| [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgccs1SNFUGSTsmRJ8w?e=a9zKwZ) |
| **ViTPose+-H** | COCO+AIC+MPII+AP10K+APT36K+WholeBody | 256x192 | 82.3 | [config](configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/apt36k/ViTPose_huge_apt36k_256x192.py) | [log](https://1drv.ms/u/s!AimBgYV7JjTlgcclxZOlwRJdqpIIjA?e=nFQgVC) \| [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgccoXv8rCUgVe7oD9Q?e=ZBw6gR) |

### WholeBody dataset

| Model | Dataset | Resolution | AP | config | weight |
| :----: | :----: | :----: | :----: | :----: | :----: |
| **ViTPose+-S** | COCO+AIC+MPII+AP10K+APT36K+WholeBody | 256x192 | 54.4 | [config](configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/ViTPose_small_wholebody_256x192.py)  | [log](https://1drv.ms/u/s!AimBgYV7JjTlgccqO1JBHtBjNaeCbQ?e=ZN5NSz) \| [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgccrwORr61gT9E4n8g?e=kz9sz5) |
| **ViTPose+-B** | COCO+AIC+MPII+AP10K+APT36K+WholeBody | 256x192 | 57.4 | [config](cconfigs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/ViTPose_base_wholebody_256x192.py)  | [log](https://1drv.ms/u/s!AimBgYV7JjTlgccjj9lgPTlkGT1HTw?e=OlS5zv) \| [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgcckRZk1bIAuRa_E1w?e=ylDB2G) |
| **ViTPose+-L** | COCO+AIC+MPII+AP10K+APT36K+WholeBody | 256x192 | 60.6 | [config](configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/ViTPose_large_wholebody_256x192.py) | [log](https://1drv.ms/u/s!AimBgYV7JjTlgccp7HJf4QMeQQpeyA?e=JagPNt) \| [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgccs1SNFUGSTsmRJ8w?e=a9zKwZ) |
| **ViTPose+-H** | COCO+AIC+MPII+AP10K+APT36K+WholeBody | 256x192 | 61.2 | [config](configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/ViTPose_huge_wholebody_256x192.py) | [log](https://1drv.ms/u/s!AimBgYV7JjTlgcclxZOlwRJdqpIIjA?e=nFQgVC) \| [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgccoXv8rCUgVe7oD9Q?e=ZBw6gR) |

### Transfer results on the hand dataset (InterHand2.6M)

| Model | Dataset | Resolution | AUC | config | weight |
| :----: | :----: | :----: | :----: | :----: | :----: |
| **ViTPose+-S** | COCO+AIC+MPII+WholeBody | 256x192 | 86.5 | [config](configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/interhand2d/ViTPose_small_interhand2d_all_256x192.py)  | Coming Soon |
| **ViTPose+-B** | COCO+AIC+MPII+WholeBody | 256x192 | 87.0 | [config](configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/interhand2d/ViTPose_base_interhand2d_all_256x192.py)  | Coming Soon |
| **ViTPose+-L** | COCO+AIC+MPII+WholeBody | 256x192 | 87.5 | [config](configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/interhand2d/ViTPose_large_interhand2d_all_256x192.py) | Coming Soon |
| **ViTPose+-H** | COCO+AIC+MPII+WholeBody | 256x192 | 87.6 | [config](configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/interhand2d/ViTPose_huge_interhand2d_all_256x192.py) | Coming Soon |

## Updates

> [2023-01-10] Update ViTPose+! It uses MoE strategies to jointly deal with human, animal, and wholebody pose estimation tasks.

> [2022-05-24] Upload the single-task training code, single-task pre-trained models, and multi-task pretrained models.

> [2022-05-06] Upload the logs for the base, large, and huge models!

> [2022-04-27] Our ViTPose with ViTAE-G obtains 81.1 AP on COCO test-dev set! 

> Applications of ViTAE Transformer include: [image classification](https://github.com/ViTAE-Transformer/ViTAE-Transformer/tree/main/Image-Classification) | [object detection](https://github.com/ViTAE-Transformer/ViTAE-Transformer/tree/main/Object-Detection) | [semantic segmentation](https://github.com/ViTAE-Transformer/ViTAE-Transformer/tree/main/Semantic-Segmentation) | [animal pose segmentation](https://github.com/ViTAE-Transformer/ViTAE-Transformer/tree/main/Animal-Pose-Estimation) | [remote sensing](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Remote-Sensing) | [matting](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Matting) | [VSA](https://github.com/ViTAE-Transformer/ViTAE-VSA) | [ViTDet](https://github.com/ViTAE-Transformer/ViTDet)

## Usage

We use PyTorch 1.9.0 or NGC docker 21.06, and mmcv 1.3.9 for the experiments.
```bash
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
git checkout v1.3.9
MMCV_WITH_OPS=1 pip install -e .
cd ..
git clone https://github.com/ViTAE-Transformer/ViTPose.git
cd ViTPose
pip install -v -e .
```

After install the two repos, install timm and einops, i.e.,
```bash
pip install timm==0.4.9 einops
```

After downloading the pretrained models, please conduct the experiments by running

```bash
# for single machine
bash tools/dist_train.sh <Config PATH> <NUM GPUs> --cfg-options model.pretrained=<Pretrained PATH> --seed 0

# for multiple machines
python -m torch.distributed.launch --nnodes <Num Machines> --node_rank <Rank of Machine> --nproc_per_node <GPUs Per Machine> --master_addr <Master Addr> --master_port <Master Port> tools/train.py <Config PATH> --cfg-options model.pretrained=<Pretrained PATH> --launcher pytorch --seed 0
```

To test the pretrained models performance, please run 

```bash
bash tools/dist_test.sh <Config PATH> <Checkpoint PATH> <NUM GPUs>
```

For ViTPose+ pre-trained models, please first re-organize the pre-trained weights using

```bash
python tools/model_split.py --source <Pretrained PATH>
```

## Todo

This repo current contains modifications including:

- [x] Upload configs and pretrained models

- [x] More models with SOTA results

- [x] Upload multi-task training config

## Acknowledge
We acknowledge the excellent implementation from [mmpose](https://github.com/open-mmlab/mmdetection) and [MAE](https://github.com/facebookresearch/mae).

## Citing ViTPose

For ViTPose

```
@inproceedings{
  xu2022vitpose,
  title={Vi{TP}ose: Simple Vision Transformer Baselines for Human Pose Estimation},
  author={Yufei Xu and Jing Zhang and Qiming Zhang and Dacheng Tao},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022},
}
```

For ViTPose+

```
@article{xu2022vitpose+,
  title={ViTPose+: Vision Transformer Foundation Model for Generic Body Pose Estimation},
  author={Xu, Yufei and Zhang, Jing and Zhang, Qiming and Tao, Dacheng},
  journal={arXiv preprint arXiv:2212.04246},
  year={2022}
}
```

For ViTAE and ViTAEv2, please refer to:
```
@article{xu2021vitae,
  title={Vitae: Vision transformer advanced by exploring intrinsic inductive bias},
  author={Xu, Yufei and Zhang, Qiming and Zhang, Jing and Tao, Dacheng},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}

@article{zhang2022vitaev2,
  title={ViTAEv2: Vision Transformer Advanced by Exploring Inductive Bias for Image Recognition and Beyond},
  author={Zhang, Qiming and Xu, Yufei and Zhang, Jing and Tao, Dacheng},
  journal={arXiv preprint arXiv:2202.10108},
  year={2022}
}
```
