from huggingface_hub import snapshot_download
import torch
import random
import numpy as np
from skimage import color, filters, exposure
from PIL import Image, ImageFilter, ImageEnhance
from enum import Enum
from typing import List, Union
import numpy as np
import folder_paths
import os

SCRIPT_DIR= os.path.dirname(os.path.abspath(__file__))
PROMPT_STYLES_DIR = os.path.join(SCRIPT_DIR, "style")
FONT_STYLES_DIR = os.path.join(SCRIPT_DIR, "fonts")
FOOOCUS_STYLES_DIR = os.path.join(SCRIPT_DIR, "styles")
#text number
MAX_TEXT_NUM = 10
#loras stack number
MAX_LORA_NUM = 5
#controlnets stack number
MAX_CN_NUM = 3

NEGATIVE_PROMPT = "bad proportions, low resolution,worst quality, low quality, normal quality, lowres, inaccurate limb, bad, ugly, terrible,  extra fingers, fewer fingers, missing fingers, extra arms, extra legs, inaccurate eyes, bad composition, bad anatomy, error, extra digit, fewer digits, cropped, low res, jpeg artifacts, trademark,artist's name, username, watermarksignature, watermark,text, words"
#resolution list

BASE_RESOLUTIONS = [
    ("width", "height"),
    (128, 128),
    (256, 256),
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

# np to Tensor
def np2tensor(img_np: Union[np.ndarray, List[np.ndarray]]) -> torch.Tensor:
  if isinstance(img_np, list):
    return torch.cat([np2tensor(img) for img in img_np], dim=0)
  return torch.from_numpy(img_np.astype(np.float32) / 255.0).unsqueeze(0)
# Tensor to np
def tensor2np(tensor: torch.Tensor) -> List[np.ndarray]:
  if len(tensor.shape) == 3:  # Single image
    return np.clip(255.0 * tensor.cpu().numpy(), 0, 255).astype(np.uint8)
  else:  # Batch of images
    return [np.clip(255.0 * t.cpu().numpy(), 0, 255).astype(np.uint8) for t in tensor]

#素描效果
def image_sketch_photo(img):
    # 将RGB图像转换为灰度图
    gray_image = color.rgb2gray(img)

    # 使用Sobel算子提取边缘
    edges = filters.sobel(gray_image, mode='constant')  

    # 对边缘图像进行反色处理
    inverted_edges = 1 - edges

    # 调整反色后的边缘图像的亮度和对比度
    adjusted_image = exposure.adjust_gamma(inverted_edges, gamma=0.5)

    # 将处理后的图像像素值限制在0到1之间    
    adjusted_image = np.clip(adjusted_image, 0, 1)

    #sketch = (adjusted_image * 255).astype(np.uint8)
    # 将处理后的图像像素值转换为合适范围后再转换为8位无符号整数
    sketch = ((adjusted_image * 300).astype(np.uint8))

    # 使用直方图均衡化增强图像对比度
    #sketch = exposure.equalize_hist(sketch)
    # 使用PIL创建图像对象并显示
    sketch_pil = Image.fromarray(sketch)
    brightness_enhancer = ImageEnhance.Brightness(sketch_pil)
    sketch_pil = brightness_enhancer.enhance(4.5)

    if sketch_pil == None:
        return img

    return sketch_pil

#复古效果
def image_vintage_photo(img):

    # 调整颜色
    # 降低饱和度
    color_enhancer = ImageEnhance.Color(img)
    img = color_enhancer.enhance(0.7)
    # 调整色调
    hsv_img = img.convert('HSV')
    h, s, v = hsv_img.split()
    h = h.point(lambda i: (i + 30) % 256)
    hsv_img = Image.merge('HSV', (h, s, v))
    img = hsv_img.convert('RGB')

    # 调整亮度和对比度
    brightness_enhancer = ImageEnhance.Brightness(img)
    img = brightness_enhancer.enhance(1.1)
    contrast_enhancer = ImageEnhance.Contrast(img)
    img = contrast_enhancer.enhance(0.9)

    # 添加噪点
    width, height = img.size
    for _ in range(int(width * height * 0.005)):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        r, g, b = img.getpixel((x, y))
        noise = random.randint(-10, 10)
        r = max(min(r + noise, 255), 0)
        g = max(min(g + noise, 255), 0)
        b = max(min(b + noise, 255), 0)
        img.putpixel((x, y), (r, g, b))

    # 应用模糊滤镜
    img = img.filter(ImageFilter.GaussianBlur(radius=1))

    return img

def image_gaussian_noise(pil_image, mean=0, std=20):
    img_np = np.array(pil_image).astype(np.float32)
    noise = np.random.normal(mean, std, img_np.shape).astype(np.float32)
    noisy_img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
    noisy_image = Image.fromarray(noisy_img_np)
    return noisy_image
def get_new_bounds(width, height, left, right, top, bottom):
  """Returns the new bounds for an image with inset crop data."""
  left = 0 + left
  right = width - right
  top = 0 + top
  bottom = height - bottom
  return (left, right, top, bottom)

class ResizeMode(Enum):
  RESIZE = "Just Resize"
  INNER_FIT = "Crop and Resize"
  OUTER_FIT = "Resize and Fill"
  def int_value(self):
    if self == ResizeMode.RESIZE:
      return 0
    elif self == ResizeMode.INNER_FIT:
      return 1
    elif self == ResizeMode.OUTER_FIT:
      return 2
    assert False, "NOTREACHED"
def resize_image(image, max_size=768):
    """
    调整图像大小,使得长边不超过max_size像素。
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