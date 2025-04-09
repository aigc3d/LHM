### How to install on Windows11

#### Base software
- Python3.10: Download from [official website](https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe)  rather than Windows Store(which has a path length limit by default which will cause error)
- Visual Studio 2019: 2022 will cause some compilation error on cuda operators. Download it from [techspot](https://www.techspot.com/downloads/7241-visual-studio-2019.html)
- Nvidia Cuda 11.8: It is on my case. You can install other version accroding to your convenience. [CUDA ToolKits](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Windows)
- Git: Download and install from [Git official](https://git-scm.com/)

#### Install Steps
Note we use "x64 Native Tools" as the compilation and do not use powershell or cmd!!! Search it after install Visual Studio 2019. It offer MSVC environment for python package compilation.

After Open "x64 Native Tools", we can install the dependency step by step:
- Cloning
    - Clone ComfyUI
    ```bash
    git clone https://github.com/comfyanonymous/ComfyUI
    ```
    - Clone LHM
    ```bash
    git clone --branch feat/comfyui https://github.com/aigc3d/LHM
    ```
    - Copy the nodes into ComfyUI, this need you to manually "Ctrl+C" and "Ctrl+V" to copy "ComfyUI-LHM" && "ComfyUI-VideoHelperSuite" into "ComfyUI/custom_nodes/"

- Downloading
    - Open this [link](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LHM/ComfyUI/LHM_ComfyUI.zip)
    - After download is finished, move it into "ComfyUI\models\checkpoints\"
    - Unzip it as folder "LHM" as follows
    ```bash
    - ComfyUI
        - models
            - checkpoints
                - LHM
                - LHM_ComfyUI.zip
                - *.txt
    ```

- Installing dependency(x64 Native Tools)
    - Install pytorch and xformers
    ```bash
    pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
    pip install -U xformers==0.0.26.post1 --index-url https://download.pytorch.org/whl/cu118
    ```
    - Install packages which do not need compilation
    ```bash
    # ensure your terminal path is in xx\xx\LHM\
    pip install -r requirements.txt
    ```
    - resolve some conflicts
    ```bash
    pip uninstall basicsr
    pip install git+https://github.com/XPixelGroup/BasicSR
    pip install torchsde onnxruntime==1.14.0 numpy==1.25.0
    pip install git+https://github.com/mattloper/chumpy
    ```

    - Install packages which need compilation
    ```bash 
    # The biggest problem is Pytorch3d, which I offer some docs I refer to:
    # https://blog.csdn.net/m0_70229101/article/details/127196699
    # https://blog.csdn.net/qq_61247019/article/details/139927752
    set DISTUTILS_USE_SDK=1
    set PYTORCH3D_NO_NINJA=1
    git clone https://github.com/facebookresearch/pytorch3d.git
    cd pytorch3d
    # modify setup.py
    # add "-DWIN32_LEAN_AND_MEAN" in nvcc_args
    python setup.py install

    # install diff-gaussian and simple-knn
    pip install git+https://github.com/ashawkey/diff-gaussian-rasterization/
    pip install git+https://github.com/camenduru/simple-knn/

    # install VitPose
    # ensure your terminal path is in xx\xx\ComfyUI\
    pip install -v -e .\custom_nodes\ComfyUI-LHM\lib_lhm\engine\pose_estimation\third-party\ViTPose
    pip install ultralytics mmcv-full
    ```

#### Running
```bash
# Go to ComfyUI root using your x64 terminal
python main.py

# After opening, Load LHM_WorkFlow.json and Try it~
```


#### FQA

- '*.obj" can not be compiled: Usually open "setup.py" and change:
```bash
BuildExtension -> BuildExtension.with_options(use_ninja=False)
```
will fix this issue.

- The ternimal log indicates that xx/xx/xx.mp4 do not exists. This happens when your computer can not correctly recognize it, change it to xx\xx\xx.mp4 and try again.