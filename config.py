from huggingface_hub import snapshot_download
import torch
from PIL import Image
import numpy as np
import folder_paths
import os

SCRIPT_DIR= os.path.dirname(os.path.abspath(__file__))
PROMPT_STYLES_DIR = os.path.join(SCRIPT_DIR, "style")
#loras stack number
MAX_LORA_NUM = 5
#controlnets stack number
MAX_CN_NUM = 3

NEGATIVE_PROMPT = "bad proportions, low resolution,worst quality, low quality, normal quality, lowres, inaccurate limb, bad, ugly, terrible,  extra fingers, fewer fingers, missing fingers, extra arms, extra legs, inaccurate eyes, bad composition, bad anatomy, error, extra digit, fewer digits, cropped, low res, jpeg artifacts, trademark,artist's name, username, watermarksignature, watermark,text, words"
#resolution list

BASE_RESOLUTIONS = [
    ("width", "height"),
    (512, 512),
    (512, 768),
    (576, 1024),
    (768, 512),
    (768, 768),
    (768, 1024),
    (768, 1280),
    (768, 1344),
    (768, 1536),
    (816, 1920),
    (832, 1152),
    (832, 1216),
    (896, 1152),
    (896, 1088),
    (1024, 1024),
    (1024, 576),
    (1024, 768),
    (1080, 1920),
    (1440, 2560),
    (1088, 896),
    (1216, 832),
    (1152, 832),
    (1152, 896),
    (1280, 768),
    (1344, 768),
    (1536, 640),
    (1536, 768),
    (1920, 816),
    (1920, 1080),
    (2560, 1440)
]

RESOLUTION_STRINGS = [f"{width} x {height}" if width == 'width' and height == 'height' else f"{width} x {height}" for width, height in BASE_RESOLUTIONS]

def load_model_from_hug(model):
    model_name = model.rsplit('/', 1)[-1]
    model_path = os.path.join(folder_paths.models_dir, "LLM", model_name)        
    if not os.path.exists(model_path):
        print(f"Downloading model to: {model_path}")            
        snapshot_download(repo_id=model,
                        local_dir=model_path,
                        local_dir_use_symlinks=False)
    return model_path

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
def resize_image(image, max_size=768):
    """
    调整图像大小，使得长边不超过max_size像素。
    """
    width, height = image.size
    if width > height:
        new_width = max_size
        new_height = int(height * (new_width / width))
    elif height > width:
        new_height = max_size
        new_width = int(width * (new_height / height))
    else:
        new_width = max_size
        new_height = max_size
    return image.resize((new_width, new_height), Image.LANCZOS)