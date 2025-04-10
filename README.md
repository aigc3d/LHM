#### Overview
![](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LHM/ComfyUI/UI.png)

The following install guide is by default for linux. For windows users, things become a little complicate. I prepare a standalone install guide for it [Windows Install](https://github.com/aigc3d/LHM/blob/feat/comfyui/Windows11_install.md)

#### How to Install on linux
This repo contains the required nodes and UI settings to run LHM on custom videos and custom input human images. Let's see how it going~

##### Prepare a python env
```bash
conda create -n lhm_comfy python=3.10
conda activate lhm_comfy
```

##### Clone ComfyUI
```bash
# If you do not install comfyui before
git clone https://github.com/comfyanonymous/ComfyUI
cd ComfyUI
```

##### Clone && Install LHM
```bash 
git clone --branch feat/comfyui https://github.com/aigc3d/LHM

# link the nodes under the custom_nodes(using absolute path!)
# If you install ComfyUI in /home/workspace, then /path/to equals /home/workspace.
ln -s /path/to/ComfyUI/LHM/ComfyUI-LHM /path/to/ComfyUI/custom_nodes/ComfyUI-LHM
ln -s /path/to/ComfyUI/LHM/ComfyUI-VideoHelperSuite /path/to/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite

# install packages
cd LHM && bash install_cu118.sh && cd ..
```

##### Prepare checkpoints
```bash
# in the source dir of ComfyUI
cd models/checkpoints
wget https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LHM/ComfyUI/LHM_ComfyUI.zip
unzip LHM_ComfyUI.zip
rm LHM_ComfyUI.zip
cd ../..
```

#####
```bash
# start the ComfyUI
python main.py 
```

#### How to use

we provide full pipeline or you can infer from offline motion. We also provide a offline motion extraction pipeline.

##### Standalone Motion Extract Workflow
```bash
# add showtext node
ln -s /path/to/ComfyUI/LHM/ComfyUI-Show-Text /path/to/ComfyUI/custom_nodes/ComfyUI-Show-Text
```
Open LHM_Motion_Extraction.json, the extracted motion path will be shown after executation. This workflow is designed to avoid extracting motion params each time when you want to make videos based on the same motion. It will save much time in such case.

##### Animation from offline extracted motion
 Open LHM_Animation_Infer.json, paste specific offline motion path, upload your human image and click run. 

##### Full Pipeline
Click 'Workflow' on the ComfyUI-FrondEnd and open LHM_WorkFlow.json, Upload your input image and driven video and click 'Run'!