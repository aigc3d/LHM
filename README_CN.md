# <span><img src="./assets/LHM_logo_parsing.png" height="35" style="vertical-align: top;"> - 官方 PyTorch 实现</span>

# <div align="center">LHM: Large Animatable Human Reconstruction Model for Single Image to 3D in Seconds<div> 
###  <p align="center"> [Lingteng Qiu<sup>*</sup>](https://lingtengqiu.github.io/), [Xiaodong Gu<sup>*</sup>](https://scholar.google.com.hk/citations?user=aJPO514AAAAJ&hl=zh-CN&oi=ao), [Peihao Li<sup>*</sup>](https://liphao99.github.io/), [Qi Zuo<sup>*</sup>](https://scholar.google.com/citations?user=UDnHe2IAAAAJ&hl=zh-CN)<br>[Weichao Shen](https://scholar.google.com/citations?user=7gTmYHkAAAAJ&hl=zh-CN), [Junfei Zhang](https://scholar.google.com/citations?user=oJjasIEAAAAJ&hl=en), [Kejie Qiu](https://sites.google.com/site/kejieqiujack/home), [Weihao Yuan](https://weihao-yuan.com/), <br>[Guanying Chen<sup>+</sup>](https://guanyingc.github.io/), [Zilong Dong<sup>+</sup>](https://baike.baidu.com/item/%E8%91%A3%E5%AD%90%E9%BE%99/62931048), [Liefeng Bo](https://scholar.google.com/citations?user=FJwtMf0AAAAJ&hl=zh-CN)</p>
##  <p align="center"> 阿里巴巴通义实验室</p>

[![项目主页](https://img.shields.io/badge/🌐-项目主页-blueviolet)](https://lingtengqiu.github.io/LHM/)
[![arXiv论文](https://img.shields.io/badge/📜-arXiv:2503-10625)](https://arxiv.org/pdf/2503.10625)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace_Space-blue)](https://huggingface.co/spaces/DyrusQZ/LHM)
[![Apache协议](https://img.shields.io/badge/📃-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)

<p align="center">
  <img src="./assets/LHM_teaser.png" heihgt="100%">
</p>

## 📢 最新动态
**[2025年3月24日]** SAM2难装 😭😭😭? 👉 那就用rembg吧!<br>
**[2025年3月20日]** 发布视频动作处理脚本<br>
**[2025年3月19日]** 本地部署 Gradio<br>
**[2025年3月19日]** HuggingFace Demo：更快更稳定 <br>
**[2025年3月15日]** 推理时间优化：提速30% <br>
**[2025年3月13日]** 首次版本发布包含：  
✅ 推理代码库  
✅ 预训练 LHM-0.5B 模型  
✅ 预训练 LHM-1B 模型  
✅ 实时渲染管线  
✅ Huggingface 在线演示  

### 待办清单
- [x] 核心推理管线 (v0.1) 🔥🔥🔥
- [x] HuggingFace 演示集成 🤗🤗🤗
- [ ] ModelScope 部署
- [x] 动作处理脚本 
- [ ] 训练代码发布

## 🚀 快速开始

### 环境配置
克隆仓库
```bash
git clone git@github.com:aigc3d/LHM.git
cd LHM
```

通过脚本安装依赖
```
# cuda 11.8
sh ./install_cu118.sh
pip install rembg

# cuda 12.1
sh ./install_cu121.sh
pip install rembg
```
环境已在 python3.10、CUDA 11.8 和 CUDA 12.1 下测试通过。

也可按步骤手动安装依赖，详见[INSTALL.md](INSTALL.md)

### 模型参数 

<span style="color:red">如果你没下载模型，模型将会自动下载</span>

模型	训练数据	BH-T层数	下载链接	推理时间
LHM-0.5B	5K合成数据	5	OSS	2.01 s
LHM-0.5B	300K视频+5K合成数据	5	OSS	2.01 s
LHM-0.7B	300K视频+5K合成数据	10	OSS	4.13 s
LHM-1.0B	300K视频+5K合成数据	15	OSS	6.57 s

| 模型 | 训练数据 | Transformer 层数 | 下载链接 | 推理时间 |
| :--- | :--- | :--- | :--- | :--- |
| LHM-0.5B | 5K合成数据| 5 | OSS | 2.01 s |
| LHM-0.5B | 300K视频+5K合成数据 | 5 | [OSS](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/for_lingteng/LHM/LHM-0.5B.tar) | 2.01 s |
| LHM-0.7B | 300K视频+5K合成数据 | 10 | OSS | 4.13 s  |
| LHM-1.0B | 300K视频+5K合成数据 | 15 | [OSS](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/for_lingteng/LHM/LHM-1B.tar) | 6.57 s |

```bash
# 下载预训练模型权重
wget https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/for_lingteng/LHM/LHM-0.5B.tar
tar -xvf LHM-0.5B.tar 
wget https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/for_lingteng/LHM/LHM-1B.tar
tar -xvf LHM-1B.tar
```

### 下载先验模型权重
```bash
# 下载先验模型权重
wget https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/for_lingteng/LHM/LHM_prior_model.tar
tar -xvf LHM_prior_model.tar 
```

### 动作数据准备
我们提供了测试动作示例：

```bash
# 下载先验模型权重
wget https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/for_lingteng/LHM/motion_video.tar
tar -xvf ./motion_video.tar 
```

下载完成后项目目录结构如下：
```bash
├── configs
│   ├── inference
│   ├── accelerate-train-1gpu.yaml
│   ├── accelerate-train-deepspeed.yaml
│   ├── accelerate-train.yaml
│   └── infer-gradio.yaml
├── engine
│   ├── BiRefNet
│   ├── pose_estimation
│   ├── SegmentAPI
├── example_data
│   └── test_data
├── exps
│   ├── releases
├── LHM
│   ├── datasets
│   ├── losses
│   ├── models
│   ├── outputs
│   ├── runners
│   ├── utils
│   ├── launch.py
├── pretrained_models
│   ├── dense_sample_points
│   ├── gagatracker
│   ├── human_model_files
│   ├── sam2
│   ├── sapiens
│   ├── voxel_grid
│   ├── arcface_resnet18.pth
│   ├── BiRefNet-general-epoch_244.pth
├── scripts
│   ├── exp
│   ├── convert_hf.py
│   └── upload_hub.py
├── tools
│   ├── metrics
├── train_data
│   ├── example_imgs
│   ├── motion_video
├── inference.sh
├── README.md
├── requirements.txt
```



### 💻 本地部署 
```bash
python ./app.py
```

### 🏃 推理流程
```bash
# MODEL_NAME={LHM-500M, LHM-1B}
# bash ./inference.sh ./configs/inference/human-lrm-500M.yaml LHM-500M ./train_data/example_imgs/ ./train_data/motion_video/mimo1/smplx_params
# bash ./inference.sh ./configs/inference/human-lrm-1B.yaml LHM-1B ./train_data/example_imgs/ ./train_data/motion_video/mimo1/smplx_params

# export animation video
bash inference.sh ${CONFIG} ${MODEL_NAME} ${IMAGE_PATH_OR_FOLDER}  ${MOTION_SEQ}
# export mesh 
bash ./inference_mesh.sh ${CONFIG} ${MODEL_NAME} 
```
### 处理视频动作数据

- 下载动作提取相关的预训练模型权重
  ```bash
  wget -P ./pretrained_models/human_model_files/pose_estimate https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/for_lingteng/LHM/yolov8x.pt

  wget -P ./pretrained_models/human_model_files/pose_estimate https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/for_lingteng/LHM/vitpose-h-wholebody.pth
  ```

- 安装额外的依赖
  ```bash
  cd ./engine/pose_estimation
  pip install -v -e third-party/ViTPose
  pip install ultralytics
  ```

- 运行以下命令，从视频中提取动作数据
   ```bash
   # python ./engine/pose_estimation/video2motion.py --video_path ./train_data/demo.mp4 --output_path ./train_data/custom_motion

   python ./engine/pose_estimation/video2motion.py --video_path ${VIDEO_PATH} --output_path ${OUTPUT_PATH}

   ```

- 使用提取的动作数据驱动数字人
  ```bash
  # bash ./inference.sh ./configs/inference/human-lrm-500M.yaml LHM-500M ./train_data/example_imgs/ ./train_data/custom_motion/demo/smplx_params

  bash inference.sh ${CONFIG} ${MODEL_NAME} ${IMAGE_PATH_OR_FOLDER}  ${OUTPUT_PATH}/${VIDEO_NAME}/smplx_params
  ```

## 计算指标
我们提供了简单的指标计算脚本：
```bash
# download pretrain model into ./pretrained_models/
wget https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/for_lingteng/arcface_resnet18.pth
# Face Similarity
python ./tools/metrics/compute_facesimilarity.py -f1 ${gt_folder} -f2 ${results_folder}
# PSNR 
python ./tools/metrics/compute_psnr.py -f1 ${gt_folder} -f2 ${results_folder}
# SSIM LPIPS 
python ./tools/metrics/compute_ssim_lpips.py -f1 ${gt_folder} -f2 ${results_folder} 
```

## 致谢

本工作基于以下优秀研究成果和开源项目构建：

- [OpenLRM](https://github.com/3DTopia/OpenLRM)
- [ExAvatar](https://github.com/mks0601/ExAvatar_RELEASE)
- [DreamGaussian](https://github.com/dreamgaussian/dreamgaussian)

感谢这些杰出工作对3D生成和数字人领域的重要贡献。

## 引用 
```
@inproceedings{qiu2025LHM,
  title={LHM: Large Animatable Human Reconstruction Model from a Single Image in Seconds},
  author={Lingteng Qiu and Xiaodong Gu and Peihao Li  and Qi Zuo
     and Weichao Shen and Junfei Zhang and Kejie Qiu and Weihao Yuan
     and Guanying Chen and Zilong Dong and Liefeng Bo 
    },
  booktitle={arXiv preprint arXiv:2503.10625},
  year={2025}
}
```
