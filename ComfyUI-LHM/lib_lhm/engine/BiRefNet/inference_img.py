# Imports
import pdb
import time

import torch
import tqlt.utils as tu
from models.birefnet import BiRefNet
from PIL import Image
from torchvision import transforms

# # Option 1: loading BiRefNet with weights:
from transformers import AutoModelForImageSegmentation

# # Option-3: Loading model and weights from local disk:
from utils import check_state_dict

# birefnet = AutoModelForImageSegmentation.from_pretrained(
#     "zhengpeng7/BiRefNet", trust_remote_code=True, local
# )

# # Option-2: loading weights with BiReNet codes:
# birefnet = BiRefNet.from_pretrained('zhengpeng7/BiRefNet')
imgs = tu.next_files("./in_the_wild", ".png")


birefnet = BiRefNet(bb_pretrained=False)
state_dict = torch.load("./BiRefNet-general-epoch_244.pth", map_location="cpu")
state_dict = check_state_dict(state_dict)
birefnet.load_state_dict(state_dict)


# Load Model
device = "cuda"
torch.set_float32_matmul_precision(["high", "highest"][0])

birefnet.to(device)
birefnet.eval()
print("BiRefNet is ready to use.")

# Input Data
transform_image = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


import os
from glob import glob

from image_proc import refine_foreground

src_dir = "./images_todo"
image_paths = glob(os.path.join(src_dir, "*"))
dst_dir = "./predictions"
os.makedirs(dst_dir, exist_ok=True)
for image_path in imgs:

    print("Processing {} ...".format(image_path))
    image = Image.open(image_path)
    input_images = transform_image(image).unsqueeze(0).to("cuda")

    # Prediction
    start = time.time()

    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()

    print(time.time() - start)
    pred = preds[0].squeeze()

    # Save Results
    file_ext = os.path.splitext(image_path)[-1]
    pred_pil = transforms.ToPILImage()(pred)
    pred_pil = pred_pil.resize(image.size)
    pred_pil.save(image_path.replace(src_dir, dst_dir).replace(file_ext, "-mask.png"))
    image_masked = refine_foreground(image, pred_pil)
    image_masked.putalpha(pred_pil)
    image_masked.save(
        image_path.replace(src_dir, dst_dir).replace(file_ext, "-subject.png")
    )

    # Save Results
    file_ext = os.path.splitext(image_path)[-1]
    pred_pil = transforms.ToPILImage()(pred)
    pred_pil = pred_pil.resize(image.size)
    pred_pil.save(image_path.replace(src_dir, dst_dir).replace(file_ext, "-mask.png"))
