# install torch 2.3.0
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install -U xformers==0.0.26.post1 --index-url https://download.pytorch.org/whl/cu121

# install dependencies
pip install -r requirements.txt

# install from source code to avoid the conflict with torchvision
pip uninstall basicsr
pip install git+https://github.com/XPixelGroup/BasicSR

# install pytorch3d
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# install diff-gaussian-rasterization
pip install git+https://github.com/ashawkey/diff-gaussian-rasterization/

# install simple-knn
pip install git+https://github.com/camenduru/simple-knn/

pip install torchsde onnxruntime numpy==1.25.0

pip install git+https://github.com/mattloper/chumpy

pip install -v -e ./custom_nodes/ComfyUI-LHM/lib_lhm/engine/pose_estimation/third-party/ViTPose
pip install ultralytics mmcv-full