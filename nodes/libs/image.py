import torch
from torchvision.utils import make_grid
import folder_paths
from skimage import color, filters, exposure
from PIL import Image,ImageFilter, ImageEnhance,ImageChops # ImageGrab, ImageSequence
import random,copy,math,os,hashlib
from pathlib import Path
import base64
from io import BytesIO
import numpy as np
from enum import Enum
from typing import List, Union
import cv2
from cv2.ximgproc import guidedFilter
from skimage import img_as_float, img_as_ubyte

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
def tensor_to_pil(img_tensor, batch_index=0):
    # Convert tensor of shape [batch_size, channels, height, width] at the batch_index to PIL Image
    img_tensor = img_tensor[batch_index].unsqueeze(0)
    i = 255. * img_tensor.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8).squeeze())
    return img
def batch_tensor_to_pil(img_tensor):
    # Convert tensor of shape [batch_size, channels, height, width] to a list of PIL Images
    return [tensor_to_pil(img_tensor, i) for i in range(img_tensor.shape[0])]

def batched_pil_to_tensor(images):
    # Takes a list of PIL images and returns a tensor of shape [batch_size, height, width, channels]
    return torch.cat([pil2tensor(image) for image in images], dim=0)

def get_image_md5hash(image: Image.Image):
    md5hash = hashlib.md5(image.tobytes())
    return md5hash.hexdigest()
def img2tensor(imgs, bgr2rgb=True, float32=True):

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)
def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError('Only support 4D, 3D or 2D tensor. ' f'But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result

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

def cv22pil(cv2_img:np.ndarray) -> Image:
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2_img)

def pil2cv2(pil_img:Image) -> np.array:
    np_img_array = np.asarray(pil_img)
    return cv2.cvtColor(np_img_array, cv2.COLOR_RGB2BGR)

def tensor2cv2(image:torch.Tensor) -> np.array:
    if image.dim() == 4:
        image = image.squeeze()
    npimage = image.numpy()
    cv2image = np.uint8(npimage * 255 / npimage.max())
    return cv2.cvtColor(cv2image, cv2.COLOR_RGB2BGR)

def cv22ski(cv2_image:np.ndarray) -> np.array:
    return img_as_float(cv2_image)

def ski2cv2(ski:np.array) -> np.ndarray:
    return img_as_ubyte(ski)
def image2base64(image_base64):
    image_bytes = base64.b64decode(image_base64)
    image_data = Image.open(BytesIO(image_bytes))
    return image_data
def image2mask(image:Image) -> torch.Tensor:
    _image = image.convert('RGBA')
    alpha = _image.split() [0]
    bg = Image.new("L", _image.size)
    _image = Image.merge('RGBA', (bg, bg, bg, alpha))
    ret_mask = torch.tensor([pil2tensor(_image)[0, :, :, 3].tolist()])
    return ret_mask

def mask2image(mask:torch.Tensor)  -> Image:
    masks = tensor2np(mask)
    for m in masks:
        _mask = Image.fromarray(m).convert("L")
        _image = Image.new("RGBA", _mask.size, color='white')
        _image = Image.composite(
            _image, Image.new("RGBA", _mask.size, color='black'), _mask)
    return _image
def gamma_trans(image:Image, gamma:float) -> Image:
    cv2_image = pil2cv2(image)
    gamma_table = [np.power(x/255.0,gamma)*255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    _corrected = cv2.LUT(cv2_image,gamma_table)
    return cv22pil(_corrected)
def RGB2RGBA(image:Image, mask:Image) -> Image:
    (R, G, B) = image.convert('RGB').split()
    return Image.merge('RGBA', (R, G, B, mask.convert('L')))

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

def resize_image_1024(image):
    image = image.convert('RGB')
    model_input_size = (1024, 1024)
    image = image.resize(model_input_size, Image.BILINEAR)
    return image
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
#棕褐色调色转换
def sepia_image(img):
    # 将图像转换为RGB模式
    if img.mode!= 'RGB':
        img = img.convert('RGB')
    width, height = img.size
    pixels = img.load()
    for x in range(width):
        for y in range(height):
            r, g, b = pixels[x, y]
            # 应用棕褐色调转换公式
            new_r = int((r * 0.393 + g * 0.769 + b * 0.189))
            new_g = int((r * 0.349 + g * 0.686 + b * 0.168))
            new_b = int((r * 0.272 + g * 0.534 + b * 0.131))
            pixels[x, y] = (new_r if new_r < 255 else 255, new_g if new_g < 255 else 255, new_b if new_b < 255 else 255)
    sepia_img = img
    return sepia_img

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
    # 将处理后的图像像素值转换为合适范围后再转换为8位无符号整数
    sketch = ((adjusted_image * 300).astype(np.uint8))
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

def color_adapter(image:Image, ref_image:Image) -> Image:
    image = pil2cv2(image)
    ref_image = pil2cv2(ref_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image_mean, image_std = calculate_mean_std(image)
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2LAB)
    ref_image_mean, ref_image_std = calculate_mean_std(ref_image)
    _image = ((image - image_mean) * (ref_image_std / image_std)) + ref_image_mean
    np.putmask(_image, _image > 255, values=255)
    np.putmask(_image, _image < 0, values=0)
    ret_image = cv2.cvtColor(cv2.convertScaleAbs(_image), cv2.COLOR_LAB2BGR)
    return cv22pil(ret_image)

def calculate_mean_std(image:Image):
    mean, std = cv2.meanStdDev(image)
    mean = np.hstack(np.around(mean, decimals=2))
    std = np.hstack(np.around(std, decimals=2))
    return mean, std
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

def gamma_trans(image:Image, gamma:float) -> Image:
    cv2_image = pil2cv2(image)
    gamma_table = [np.power(x/255.0,gamma)*255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    _corrected = cv2.LUT(cv2_image,gamma_table)
    return cv22pil(_corrected)

# 颜色加深
def blend_color_burn(background_image:Image, layer_image:Image) -> Image:
    img_1 = cv22ski(pil2cv2(background_image))
    img_2 = cv22ski(pil2cv2(layer_image))
    img = 1 - (1 - img_2) / (img_1 + 0.001)
    mask_1 = img < 0
    mask_2 = img > 1
    img = img * (1 - mask_1)
    img = img * (1 - mask_2) + mask_2
    return cv22pil(ski2cv2(img))

# 颜色减淡
def blend_color_dodge(background_image:Image, layer_image:Image) -> Image:
    img_1 = cv22ski(pil2cv2(background_image))
    img_2 = cv22ski(pil2cv2(layer_image))
    img = img_2 / (1.0 - img_1 + 0.001)
    mask_2 = img > 1
    img = img * (1 - mask_2) + mask_2
    return cv22pil(ski2cv2(img))

# 线性加深
def blend_linear_burn(background_image:Image, layer_image:Image) -> Image:
    img_1 = cv22ski(pil2cv2(background_image))
    img_2 = cv22ski(pil2cv2(layer_image))
    img = img_1 + img_2 - 1
    mask_1 = img < 0
    img = img * (1 - mask_1)
    return cv22pil(ski2cv2(img))

# 线性减淡
def blend_linear_dodge(background_image:Image, layer_image:Image) -> Image:
    img_1 = cv22ski(pil2cv2(background_image))
    img_2 = cv22ski(pil2cv2(layer_image))
    img = img_1 + img_2
    mask_2 = img > 1
    img = img * (1 - mask_2) + mask_2
    return cv22pil(ski2cv2(img))

# 变亮
def blend_lighten(background_image:Image, layer_image:Image) -> Image:
    img_1 = cv22ski(pil2cv2(background_image))
    img_2 = cv22ski(pil2cv2(layer_image))
    img = img_1 - img_2
    mask = img > 0
    img = img_1 * mask + img_2 * (1 - mask)
    return cv22pil(ski2cv2(img))

# 变暗
def blend_dark(background_image:Image, layer_image:Image) -> Image:
    img_1 = cv22ski(pil2cv2(background_image))
    img_2 = cv22ski(pil2cv2(layer_image))
    img = img_1 - img_2
    mask = img < 0
    img = img_1 * mask + img_2 * (1 - mask)
    return cv22pil(ski2cv2(img))

# 滤色
def blend_screen(background_image:Image, layer_image:Image) -> Image:
    img_1 = cv22ski(pil2cv2(background_image))
    img_2 = cv22ski(pil2cv2(layer_image))
    img = 1 - (1 - img_1) * (1 - img_2)
    return cv22pil(ski2cv2(img))

# 叠加
def blend_overlay(background_image:Image, layer_image:Image) -> Image:
    img_1 = cv22ski(pil2cv2(background_image))
    img_2 = cv22ski(pil2cv2(layer_image))
    mask = img_2 < 0.5
    img = 2 * img_1 * img_2 * mask + (1 - mask) * (1 - 2 * (1 - img_1) * (1 - img_2))
    return cv22pil(ski2cv2(img))

# 柔光
def blend_soft_light(background_image:Image, layer_image:Image) -> Image:
    img_1 = cv22ski(pil2cv2(background_image))
    img_2 = cv22ski(pil2cv2(layer_image))
    mask = img_1 < 0.5
    T1 = (2 * img_1 - 1) * (img_2 - img_2 * img_2) + img_2
    T2 = (2 * img_1 - 1) * (np.sqrt(img_2) - img_2) + img_2
    img = T1 * mask + T2 * (1 - mask)
    return cv22pil(ski2cv2(img))

# 强光
def blend_hard_light(background_image:Image, layer_image:Image) -> Image:
    img_1 = cv22ski(pil2cv2(background_image))
    img_2 = cv22ski(pil2cv2(layer_image))
    mask = img_1 < 0.5
    T1 = 2 * img_1 * img_2
    T2 = 1 - 2 * (1 - img_1) * (1 - img_2)
    img = T1 * mask + T2 * (1 - mask)
    return cv22pil(ski2cv2(img))

# 亮光
def blend_vivid_light(background_image:Image, layer_image:Image) -> Image:
    img_1 = cv22ski(pil2cv2(background_image))
    img_2 = cv22ski(pil2cv2(layer_image))
    mask = img_1 < 0.5
    T1 = 1 - (1 - img_2) / (2 * img_1 + 0.001)
    T2 = img_2 / (2 * (1 - img_1) + 0.001)
    mask_1 = T1 < 0
    mask_2 = T2 > 1
    T1 = T1 * (1 - mask_1)
    T2 = T2 * (1 - mask_2) + mask_2
    img = T1 * mask + T2 * (1 - mask)
    return cv22pil(ski2cv2(img))

# 点光
def blend_pin_light(background_image:Image, layer_image:Image) -> Image:
    img_1 = cv22ski(pil2cv2(background_image))
    img_2 = cv22ski(pil2cv2(layer_image))
    mask_1 = img_2 < (img_1 * 2 - 1)
    mask_2 = img_2 > 2 * img_1
    T1 = 2 * img_1 - 1
    T2 = img_2
    T3 = 2 * img_1
    img = T1 * mask_1 + T2 * (1 - mask_1) * (1 - mask_2) + T3 * mask_2
    return cv22pil(ski2cv2(img))

# 线性光
def blend_linear_light(background_image:Image, layer_image:Image) -> Image:
    img_1 = cv22ski(pil2cv2(background_image))
    img_2 = cv22ski(pil2cv2(layer_image))
    img = img_2 + img_1 * 2 - 1
    mask_1 = img < 0
    mask_2 = img > 1
    img = img * (1 - mask_1)
    img = img * (1 - mask_2) + mask_2
    return cv22pil(ski2cv2(img))

def blend_hard_mix(background_image:Image, layer_image:Image) -> Image:
    img_1 = cv22ski(pil2cv2(background_image))
    img_2 = cv22ski(pil2cv2(layer_image))
    img = img_1 + img_2
    mask = img_1 + img_2 > 1
    img = img * (1 - mask) + mask
    img = img * mask
    return cv22pil(ski2cv2(img))

def chop_image(background_image:Image, layer_image:Image, blend_mode:str, opacity:int) -> Image:
    ret_image = background_image
    if blend_mode == 'normal':
        ret_image = copy.deepcopy(layer_image)
    if blend_mode == 'multply':
        ret_image = ImageChops.multiply(background_image,layer_image)
    if blend_mode == 'screen':
        ret_image = ImageChops.screen(background_image, layer_image)
    if blend_mode == 'add':
        ret_image = ImageChops.add(background_image, layer_image, 1, 0)
    if blend_mode == 'subtract':
        ret_image = ImageChops.subtract(background_image, layer_image, 1, 0)
    if blend_mode == 'difference':
        ret_image = ImageChops.difference(background_image, layer_image)
    if blend_mode == 'darker':
        ret_image = ImageChops.darker(background_image, layer_image)
    if blend_mode == 'lighter':
        ret_image = ImageChops.lighter(background_image, layer_image)
    if blend_mode == 'color_burn':
        ret_image = blend_color_burn(background_image, layer_image)
    if blend_mode == 'color_dodge':
        ret_image = blend_color_dodge(background_image, layer_image)
    if blend_mode == 'linear_burn':
        ret_image = blend_linear_burn(background_image, layer_image)
    if blend_mode == 'linear_dodge':
        ret_image = blend_linear_dodge(background_image, layer_image)
    if blend_mode == 'overlay':
        ret_image = blend_overlay(background_image, layer_image)
    if blend_mode == 'soft_light':
        ret_image = blend_soft_light(background_image, layer_image)
    if blend_mode == 'hard_light':
        ret_image = blend_hard_light(background_image, layer_image)
    if blend_mode == 'vivid_light':
        ret_image = blend_vivid_light(background_image, layer_image)
    if blend_mode == 'pin_light':
        ret_image = blend_pin_light(background_image, layer_image)
    if blend_mode == 'linear_light':
        ret_image = blend_linear_light(background_image, layer_image)
    if blend_mode == 'hard_mix':
        ret_image = blend_hard_mix(background_image, layer_image)
    # opacity
    if opacity == 0:
        ret_image = background_image
    elif opacity < 100:
        alpha = 1.0 - float(opacity) / 100
        ret_image = Image.blend(ret_image, background_image, alpha)
    return ret_image

# 颜色加深
def blend_color_burn(background_image:Image, layer_image:Image) -> Image:
    img_1 = cv22ski(pil2cv2(background_image))
    img_2 = cv22ski(pil2cv2(layer_image))
    img = 1 - (1 - img_2) / (img_1 + 0.001)
    mask_1 = img < 0
    mask_2 = img > 1
    img = img * (1 - mask_1)
    img = img * (1 - mask_2) + mask_2
    return cv22pil(ski2cv2(img))

# 颜色减淡
def blend_color_dodge(background_image:Image, layer_image:Image) -> Image:
    img_1 = cv22ski(pil2cv2(background_image))
    img_2 = cv22ski(pil2cv2(layer_image))
    img = img_2 / (1.0 - img_1 + 0.001)
    mask_2 = img > 1
    img = img * (1 - mask_2) + mask_2
    return cv22pil(ski2cv2(img))

# 线性加深
def blend_linear_burn(background_image:Image, layer_image:Image) -> Image:
    img_1 = cv22ski(pil2cv2(background_image))
    img_2 = cv22ski(pil2cv2(layer_image))
    img = img_1 + img_2 - 1
    mask_1 = img < 0
    img = img * (1 - mask_1)
    return cv22pil(ski2cv2(img))

# 线性减淡
def blend_linear_dodge(background_image:Image, layer_image:Image) -> Image:
    img_1 = cv22ski(pil2cv2(background_image))
    img_2 = cv22ski(pil2cv2(layer_image))
    img = img_1 + img_2
    mask_2 = img > 1
    img = img * (1 - mask_2) + mask_2
    return cv22pil(ski2cv2(img))

# 变亮
def blend_lighten(background_image:Image, layer_image:Image) -> Image:
    img_1 = cv22ski(pil2cv2(background_image))
    img_2 = cv22ski(pil2cv2(layer_image))
    img = img_1 - img_2
    mask = img > 0
    img = img_1 * mask + img_2 * (1 - mask)
    return cv22pil(ski2cv2(img))

# 变暗
def blend_dark(background_image:Image, layer_image:Image) -> Image:
    img_1 = cv22ski(pil2cv2(background_image))
    img_2 = cv22ski(pil2cv2(layer_image))
    img = img_1 - img_2
    mask = img < 0
    img = img_1 * mask + img_2 * (1 - mask)
    return cv22pil(ski2cv2(img))

# 滤色
def blend_screen(background_image:Image, layer_image:Image) -> Image:
    img_1 = cv22ski(pil2cv2(background_image))
    img_2 = cv22ski(pil2cv2(layer_image))
    img = 1 - (1 - img_1) * (1 - img_2)
    return cv22pil(ski2cv2(img))

# 叠加
def blend_overlay(background_image:Image, layer_image:Image) -> Image:
    img_1 = cv22ski(pil2cv2(background_image))
    img_2 = cv22ski(pil2cv2(layer_image))
    mask = img_2 < 0.5
    img = 2 * img_1 * img_2 * mask + (1 - mask) * (1 - 2 * (1 - img_1) * (1 - img_2))
    return cv22pil(ski2cv2(img))

# 柔光
def blend_soft_light(background_image:Image, layer_image:Image) -> Image:
    img_1 = cv22ski(pil2cv2(background_image))
    img_2 = cv22ski(pil2cv2(layer_image))
    mask = img_1 < 0.5
    T1 = (2 * img_1 - 1) * (img_2 - img_2 * img_2) + img_2
    T2 = (2 * img_1 - 1) * (np.sqrt(img_2) - img_2) + img_2
    img = T1 * mask + T2 * (1 - mask)
    return cv22pil(ski2cv2(img))

# 强光
def blend_hard_light(background_image:Image, layer_image:Image) -> Image:
    img_1 = cv22ski(pil2cv2(background_image))
    img_2 = cv22ski(pil2cv2(layer_image))
    mask = img_1 < 0.5
    T1 = 2 * img_1 * img_2
    T2 = 1 - 2 * (1 - img_1) * (1 - img_2)
    img = T1 * mask + T2 * (1 - mask)
    return cv22pil(ski2cv2(img))

# 亮光
def blend_vivid_light(background_image:Image, layer_image:Image) -> Image:
    img_1 = cv22ski(pil2cv2(background_image))
    img_2 = cv22ski(pil2cv2(layer_image))
    mask = img_1 < 0.5
    T1 = 1 - (1 - img_2) / (2 * img_1 + 0.001)
    T2 = img_2 / (2 * (1 - img_1) + 0.001)
    mask_1 = T1 < 0
    mask_2 = T2 > 1
    T1 = T1 * (1 - mask_1)
    T2 = T2 * (1 - mask_2) + mask_2
    img = T1 * mask + T2 * (1 - mask)
    return cv22pil(ski2cv2(img))

# 点光
def blend_pin_light(background_image:Image, layer_image:Image) -> Image:
    img_1 = cv22ski(pil2cv2(background_image))
    img_2 = cv22ski(pil2cv2(layer_image))
    mask_1 = img_2 < (img_1 * 2 - 1)
    mask_2 = img_2 > 2 * img_1
    T1 = 2 * img_1 - 1
    T2 = img_2
    T3 = 2 * img_1
    img = T1 * mask_1 + T2 * (1 - mask_1) * (1 - mask_2) + T3 * mask_2
    return cv22pil(ski2cv2(img))

# 线性光
def blend_linear_light(background_image:Image, layer_image:Image) -> Image:
    img_1 = cv22ski(pil2cv2(background_image))
    img_2 = cv22ski(pil2cv2(layer_image))
    img = img_2 + img_1 * 2 - 1
    mask_1 = img < 0
    mask_2 = img > 1
    img = img * (1 - mask_1)
    img = img * (1 - mask_2) + mask_2
    return cv22pil(ski2cv2(img))

def blend_hard_mix(background_image:Image, layer_image:Image) -> Image:
    img_1 = cv22ski(pil2cv2(background_image))
    img_2 = cv22ski(pil2cv2(layer_image))
    img = img_1 + img_2
    mask = img_1 + img_2 > 1
    img = img * (1 - mask) + mask
    img = img * mask
    return cv22pil(ski2cv2(img))

def shift_image(image:Image, distance_x:int, distance_y:int, background_color:str='#000000', cyclic:bool=False) -> Image:
    width = image.width
    height = image.height
    ret_image = Image.new('RGB', size=(width, height), color=background_color)
    for x in range(width):
        for y in range(height):
            if cyclic:
                    orig_x = x + distance_x
                    if orig_x > width-1 or orig_x < 0:
                        orig_x = abs(orig_x % width)
                    orig_y = y + distance_y
                    if orig_y > height-1 or orig_y < 0:
                        orig_y = abs(orig_y % height)

                    pixel = image.getpixel((orig_x, orig_y))
                    ret_image.putpixel((x, y), pixel)
            else:
                if x > -distance_x and y > -distance_y:  # 防止回转
                    if x + distance_x < width and y + distance_y < height:  # 防止越界
                        pixel = image.getpixel((x + distance_x, y + distance_y))
                        ret_image.putpixel((x, y), pixel)
    return ret_image

def chop_image(background_image:Image, layer_image:Image, blend_mode:str, opacity:int) -> Image:
    ret_image = background_image
    if blend_mode == 'normal':
        ret_image = copy.deepcopy(layer_image)
    if blend_mode == 'multply':
        ret_image = ImageChops.multiply(background_image,layer_image)
    if blend_mode == 'screen':
        ret_image = ImageChops.screen(background_image, layer_image)
    if blend_mode == 'add':
        ret_image = ImageChops.add(background_image, layer_image, 1, 0)
    if blend_mode == 'subtract':
        ret_image = ImageChops.subtract(background_image, layer_image, 1, 0)
    if blend_mode == 'difference':
        ret_image = ImageChops.difference(background_image, layer_image)
    if blend_mode == 'darker':
        ret_image = ImageChops.darker(background_image, layer_image)
    if blend_mode == 'lighter':
        ret_image = ImageChops.lighter(background_image, layer_image)
    if blend_mode == 'color_burn':
        ret_image = blend_color_burn(background_image, layer_image)
    if blend_mode == 'color_dodge':
        ret_image = blend_color_dodge(background_image, layer_image)
    if blend_mode == 'linear_burn':
        ret_image = blend_linear_burn(background_image, layer_image)
    if blend_mode == 'linear_dodge':
        ret_image = blend_linear_dodge(background_image, layer_image)
    if blend_mode == 'overlay':
        ret_image = blend_overlay(background_image, layer_image)
    if blend_mode == 'soft_light':
        ret_image = blend_soft_light(background_image, layer_image)
    if blend_mode == 'hard_light':
        ret_image = blend_hard_light(background_image, layer_image)
    if blend_mode == 'vivid_light':
        ret_image = blend_vivid_light(background_image, layer_image)
    if blend_mode == 'pin_light':
        ret_image = blend_pin_light(background_image, layer_image)
    if blend_mode == 'linear_light':
        ret_image = blend_linear_light(background_image, layer_image)
    if blend_mode == 'hard_mix':
        ret_image = blend_hard_mix(background_image, layer_image)
    # opacity
    if opacity == 0:
        ret_image = background_image
    elif opacity < 100:
        alpha = 1.0 - float(opacity) / 100
        ret_image = Image.blend(ret_image, background_image, alpha)
    return ret_image


def remove_background(image:Image, mask:Image, color:str) -> Image:
    width = image.width
    height = image.height
    ret_image = Image.new('RGB', size=(width, height), color=color)
    ret_image.paste(image, mask=mask)
    return ret_image

def sharpen(image:Image) -> Image:
    img = pil2cv2(image)
    Laplace_kernel = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]], dtype=np.float32)
    ret_image = cv2.filter2D(img, -1, Laplace_kernel)
    return cv22pil(ret_image)

def gaussian_blur(image:Image, radius:int) -> Image:
    # image = image.convert("RGBA")
    ret_image = image.filter(ImageFilter.GaussianBlur(radius=radius))
    return ret_image

def motion_blur(image:Image, angle:int, blur:int) -> Image:
    angle += 45
    blur *= 5
    image = np.array(pil2cv2(image))
    M = cv2.getRotationMatrix2D((blur / 2, blur / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(blur))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (blur, blur))
    motion_blur_kernel = motion_blur_kernel / blur
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)
    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    ret_image = cv22pil(blurred)
    return ret_image

def guided_filter_alpha(image:torch.Tensor, mask:torch.Tensor, filter_radius:int) -> torch.Tensor:
    sigma = 0.15
    d = filter_radius + 1
    mask = pil2tensor(tensor2pil(mask).convert('RGB'))
    if not bool(d % 2):
        d += 1
    s = sigma / 10
    i_dup = copy.deepcopy(image.cpu().numpy())
    a_dup = copy.deepcopy(mask.cpu().numpy())
    for index, image in enumerate(i_dup):
        alpha_work = a_dup[index]
        i_dup[index] = guidedFilter(image, alpha_work, d, s)
    return torch.from_numpy(i_dup)

class VITMatteModel:
    def __init__(self,model,processor):
        self.model = model
        self.processor = processor
def load_VITMatte_model(model_name:str, local_files_only:bool=False) -> object:
    # if local_files_only:
    #     model_name = Path(os.path.join(folder_paths.models_dir, "vitmatte"))
    model_name = Path(os.path.join(folder_paths.models_dir, "vitmatte"))
    from transformers import VitMatteImageProcessor, VitMatteForImageMatting
    model = VitMatteForImageMatting.from_pretrained(model_name, local_files_only=local_files_only)
    processor = VitMatteImageProcessor.from_pretrained(model_name, local_files_only=local_files_only)
    vitmatte = VITMatteModel(model, processor)
    return vitmatte

def generate_VITMatte(image:Image, trimap:Image, local_files_only:bool=False, device:str="cpu", max_megapixels:float=2.0) -> Image:
    if image.mode != 'RGB':
        image = image.convert('RGB')
    if trimap.mode != 'L':
        trimap = trimap.convert('L')
    max_megapixels *= 1048576
    width, height = image.size
    ratio = width / height
    target_width = math.sqrt(ratio * max_megapixels)
    target_height = target_width / ratio
    target_width = int(target_width)
    target_height = int(target_height)
    if width * height > max_megapixels:
        image = image.resize((target_width, target_height), Image.BILINEAR)
        trimap = trimap.resize((target_width, target_height), Image.BILINEAR)
        print(f"vitmatte image size {width}x{height} too large, resize to {target_width}x{target_height} for processing.")
    model_name = "hustvl/vitmatte-small-composition-1k"
    if device=="cpu":
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            print("vitmatte device is set to cuda, but not available, using cpu instead.")
            device = torch.device('cpu')
    vit_matte_model = load_VITMatte_model(model_name=model_name, local_files_only=local_files_only)
    vit_matte_model.model.to(device)
    print(f"vitmatte processing, image size = {image.width}x{image.height}, device = {device}.")
    inputs = vit_matte_model.processor(images=image, trimaps=trimap, return_tensors="pt")
    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in inputs.items()}
        predictions = vit_matte_model.model(**inputs).alphas
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    mask = tensor2pil(predictions).convert('L')
    mask = mask.crop(
        (0, 0, image.width, image.height))  # remove padding that the prediction appends (works in 32px tiles)
    if width * height > max_megapixels:
        mask = mask.resize((width, height), Image.BILINEAR)
    return mask

def generate_VITMatte_trimap(mask:torch.Tensor, erode_kernel_size:int, dilate_kernel_size:int) -> Image:
    def g_trimap(mask, erode_kernel_size=10, dilate_kernel_size=10):
        erode_kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
        dilate_kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
        eroded = cv2.erode(mask, erode_kernel, iterations=5)
        dilated = cv2.dilate(mask, dilate_kernel, iterations=5)
        trimap = np.zeros_like(mask)
        trimap[dilated == 255] = 128
        trimap[eroded == 255] = 255
        return trimap

    mask = mask.squeeze(0).cpu().detach().numpy().astype(np.uint8) * 255
    trimap = g_trimap(mask, erode_kernel_size, dilate_kernel_size).astype(np.float32)
    trimap[trimap == 128] = 0.5
    trimap[trimap == 255] = 1
    trimap = torch.from_numpy(trimap).unsqueeze(0)

    return tensor2pil(trimap).convert('L')

def mask_edge_detail(image:torch.Tensor, mask:torch.Tensor, detail_range:int=8, black_point:float=0.01, white_point:float=0.99) -> torch.Tensor:
    from pymatting import fix_trimap, estimate_alpha_cf
    d = detail_range * 5 + 1
    mask = pil2tensor(tensor2pil(mask).convert('RGB'))
    if not bool(d % 2):
        d += 1
    i_dup = copy.deepcopy(image.cpu().numpy().astype(np.float64))
    a_dup = copy.deepcopy(mask.cpu().numpy().astype(np.float64))
    for index, img in enumerate(i_dup):
        trimap = a_dup[index][:, :, 0]  # convert to single channel
        if detail_range > 0:
            trimap = cv2.GaussianBlur(trimap, (d, d), 0)
        trimap = fix_trimap(trimap, black_point, white_point)
        alpha = estimate_alpha_cf(img, trimap, laplacian_kwargs={"epsilon": 1e-6},
                                  cg_kwargs={"maxiter": 500})
        a_dup[index] = np.stack([alpha, alpha, alpha], axis=-1)  # convert back to rgb
    return torch.from_numpy(a_dup.astype(np.float32))
def mask_fix(images:torch.Tensor, radius:int, fill_holes:int, white_threshold:float, extra_clip:float) -> torch.Tensor:
    d = radius * 2 + 1
    i_dup = copy.deepcopy(images.cpu().numpy())
    for index, image in enumerate(i_dup):
        cleaned = cv2.bilateralFilter(image, 9, 0.05, 8)
        alpha = np.clip((image - white_threshold) / (1 - white_threshold), 0, 1)
        rgb = image * alpha
        alpha = cv2.GaussianBlur(alpha, (d, d), 0) * 0.99 + np.average(alpha) * 0.01
        rgb = cv2.GaussianBlur(rgb, (d, d), 0) * 0.99 + np.average(rgb) * 0.01
        rgb = rgb / np.clip(alpha, 0.00001, 1)
        rgb = rgb * extra_clip
        cleaned = np.clip(cleaned / rgb, 0, 1)
        if fill_holes > 0:
            fD = fill_holes * 2 + 1
            gamma = cleaned * cleaned
            kD = np.ones((fD, fD), np.uint8)
            kE = np.ones((fD + 2, fD + 2), np.uint8)
            gamma = cv2.dilate(gamma, kD, iterations=1)
            gamma = cv2.erode(gamma, kE, iterations=1)
            gamma = cv2.GaussianBlur(gamma, (fD, fD), 0)
            cleaned = np.maximum(cleaned, gamma)
        i_dup[index] = cleaned
    return torch.from_numpy(i_dup)

def histogram_remap(image:torch.Tensor, blackpoint:float, whitepoint:float) -> torch.Tensor:
    bp = min(blackpoint, whitepoint - 0.001)
    scale = 1 / (whitepoint - bp)
    i_dup = copy.deepcopy(image.cpu().numpy())
    i_dup = np.clip((i_dup - bp) * scale, 0.0, 1.0)
    return torch.from_numpy(i_dup)


def mask_invert(mask:torch.Tensor) -> torch.Tensor:
    return 1 - mask

def subtract_mask(masks_a:torch.Tensor, masks_b:torch.Tensor) -> torch.Tensor:
    return torch.clamp(masks_a - masks_b, 0, 255)

def add_mask(masks_a:torch.Tensor, masks_b:torch.Tensor) -> torch.Tensor:
    mask = chop_image(tensor2pil(masks_a), tensor2pil(masks_b), blend_mode='add', opacity=100)
    return image2mask(mask)

def RGB2RGBA(image:Image, mask:Image) -> Image:
    (R, G, B) = image.convert('RGB').split()
    return Image.merge('RGBA', (R, G, B, mask.convert('L')))

def mask_area(image:Image) -> tuple:
    cv2_image = pil2cv2(image.convert('RGBA'))
    gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, 0)
    locs = np.where(thresh == 255)
    x1 = np.min(locs[1]) if len(locs[1]) > 0 else 0
    x2 = np.max(locs[1]) if len(locs[1]) > 0 else image.width
    y1 = np.min(locs[0]) if len(locs[0]) > 0 else 0
    y2 = np.max(locs[0]) if len(locs[0]) > 0 else image.height
    x1, y1, x2, y2 = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
    return (x1, y1, x2 - x1, y2 - y1)

def min_bounding_rect(image:Image) -> tuple:
    cv2_image = pil2cv2(image)
    gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, 1, 2)
    x, y, width, height = 0, 0, 0, 0
    area = 0
    for contour in contours:
        _x, _y, _w, _h = cv2.boundingRect(contour)
        _area = _w * _h
        if _area > area:
            area = _area
            x, y, width, height = _x, _y, _w, _h
    return (x, y, width, height)

def max_inscribed_rect(image:Image) -> tuple:
    img = pil2cv2(image)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_bin = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(img_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0].reshape(len(contours[0]), 2)
    rect = []
    for i in range(len(contour)):
        x1, y1 = contour[i]
        for j in range(len(contour)):
            x2, y2 = contour[j]
            area = abs(y2 - y1) * abs(x2 - x1)
            rect.append(((x1, y1), (x2, y2), area))
    all_rect = sorted(rect, key=lambda x: x[2], reverse=True)
    if all_rect:
        best_rect_found = False
        index_rect = 0
        nb_rect = len(all_rect)
        while not best_rect_found and index_rect < nb_rect:
            rect = all_rect[index_rect]
            (x1, y1) = rect[0]
            (x2, y2) = rect[1]
            valid_rect = True
            x = min(x1, x2)
            while x < max(x1, x2) + 1 and valid_rect:
                if any(img[y1, x]) == 0 or any(img[y2, x]) == 0:
                    valid_rect = False
                x += 1
            y = min(y1, y2)
            while y < max(y1, y2) + 1 and valid_rect:
                if any(img[y, x1]) == 0 or any(img[y, x2]) == 0:
                    valid_rect = False
                y += 1
            if valid_rect:
                best_rect_found = True
            index_rect += 1
    #较小的数值排前面
    x1, y1, x2, y2 = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
    return (x1, y1, x2 - x1, y2 - y1)

def gray_threshold(image:Image, thresh:int=127, otsu:bool=False) -> Image:
    cv2_image = pil2cv2(image)
    gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
    if otsu:
        _, thresh =  cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    else:
        _, thresh = cv2.threshold(gray, thresh, 255, cv2.THRESH_TOZERO)
    return cv22pil(thresh).convert('L')

def image_to_colormap(image:Image, index:int) -> Image:
    return cv22pil(cv2.applyColorMap(pil2cv2(image), index))

# 检查mask有效区域面积比例
def mask_white_area(mask:Image, white_point:int) -> float:
    if mask.mode != 'L':
        mask.convert('L')
    white_pixels = 0
    for y in range(mask.height):
        for x in range(mask.width):
            mask.getpixel((x, y)) > 16
            if mask.getpixel((x, y)) > white_point:
                white_pixels += 1
    return white_pixels / (mask.width * mask.height)

'''Color Functions'''


def RGB_to_Hex(RGB:tuple) -> str:
    color = '#'
    for i in RGB:
        num = int(i)
        color += str(hex(num))[-2:].replace('x', '0').upper()
    return color

def Hex_to_RGB(inhex:str) -> tuple:
    if not inhex.startswith('#'):
        raise ValueError(f'Invalid Hex Code in {inhex}')
    else:
        rval = inhex[1:3]
        gval = inhex[3:5]
        bval = inhex[5:]
        rgb = (int(rval, 16), int(gval, 16), int(bval, 16))
    return tuple(rgb)

def RGB_to_HSV(RGB:tuple) -> list:
    HSV = rgb_to_hsv(RGB[0] / 255.0, RGB[1] / 255.0, RGB[2] / 255.0)
    return [int(x * 360) for x in HSV]

def Hex_to_HSV_255level(inhex:str) -> list:
    if not inhex.startswith('#'):
        raise ValueError(f'Invalid Hex Code in {inhex}')
    else:
        rval = inhex[1:3]
        gval = inhex[3:5]
        bval = inhex[5:]
        RGB = (int(rval, 16), int(gval, 16), int(bval, 16))
        HSV = rgb_to_hsv(RGB[0] / 255.0, RGB[1] / 255.0, RGB[2] / 255.0)
    return [int(x * 255) for x in HSV]


def HSV_255level_to_Hex(HSV: list) -> str:
    if len(HSV) != 3 or any((not isinstance(v, int) or v < 0 or v > 255) for v in HSV):
        raise ValueError('Invalid HSV values, each value should be an integer between 0 and 255')

    H, S, V = HSV
    RGB = tuple(int(x * 255) for x in hsv_to_rgb(H / 255.0, S / 255.0, V / 255.0))

    # Convert RGB values to hexadecimal format
    hex_r = format(RGB[0], '02x')
    hex_g = format(RGB[1], '02x')
    hex_b = format(RGB[2], '02x')

    return '#' + hex_r + hex_g + hex_b

'''Value Functions'''
def is_valid_mask(tensor:torch.Tensor) -> bool:
    return not bool(torch.all(tensor == 0).item())

def step_value(start_value, end_value, total_step, step) -> float:  # 按当前步数在总步数中的位置返回比例值
    factor = step / total_step
    return (end_value - start_value) * factor + start_value

def step_color(start_color_inhex:str, end_color_inhex:str, total_step:int, step:int) -> str:  # 按当前步数在总步数中的位置返回比例颜色
    start_color = tuple(Hex_to_RGB(start_color_inhex))
    end_color = tuple(Hex_to_RGB(end_color_inhex))
    start_R, start_G, start_B = start_color[0], start_color[1], start_color[2]
    end_R, end_G, end_B = end_color[0], end_color[1], end_color[2]
    ret_color = (int(step_value(start_R, end_R, total_step, step)),
                 int(step_value(start_G, end_G, total_step, step)),
                 int(step_value(start_B, end_B, total_step, step)),
                 )
    return RGB_to_Hex(ret_color)
