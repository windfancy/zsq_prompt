import torch
import os 
import random
import numpy as np
from nodes import MAX_RESOLUTION,PreviewImage, SaveImage
import comfy.model_management
from .config import tensor2pil,pil2tensor,SCRIPT_DIR,FONT_STYLES_DIR,image_vintage_photo,image_gaussian_noise,image_sketch_photo
from PIL import Image,ImageDraw, ImageFont, ImageFilter, ImageEnhance, ImageOps # ImageGrab, ImageSequence,  ImageChops
import folder_paths
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur
import torchvision.transforms as transforms


def SaveImageUI(images, filename_prefix, output_type, prompt=None, extra_pnginfo=None):
    """Save or Preview Image"""
    if output_type in ["Hide", "None"]:
        return list()
    elif output_type in ["Preview", "Preview&Choose"]:
        filename_prefix = 'Preview'
        results = PreviewImage().save_images(images, filename_prefix, prompt, extra_pnginfo)
        return results['ui']['images']
    else:
        results = SaveImage().save_images(images, filename_prefix, prompt, extra_pnginfo)
        return results['ui']['images']

class SaveJpgImage:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.quality = 85

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."})
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "ZSQ/Image"
    DESCRIPTION = "Saves the input images to your ComfyUI output directory."

    def save_images(self, images, filename_prefix="ComfyUI"):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))  
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.jpg"
            img.save(os.path.join(full_output_folder, file), 'JPEG', quality=self.quality)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return { "ui": { "images": results } }

# 图像数量
class imageCount:
  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
        "images": ("IMAGE",),
      }
    }

  CATEGORY = "ZSQ/Image"

  RETURN_TYPES = ("INT",)
  RETURN_NAMES = ("count",)
  FUNCTION = "get_count"

  def get_count(self, images):
    return (images.size(0),)

# 图像裁切
class imageCrop:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", { "default": 256, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
                "height": ("INT", { "default": 256, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
                "position": (["top-left", "top-center", "top-right", "right-center", "bottom-right", "bottom-center", "bottom-left", "left-center", "center"],),
                "x_offset": ("INT", { "default": 0, "min": -99999, "step": 1, }),
                "y_offset": ("INT", { "default": 0, "min": -99999, "step": 1, }),
            }
        }

    RETURN_TYPES = ("IMAGE","INT","INT",)
    RETURN_NAMES = ("IMAGE","x","y",)
    FUNCTION = "execute"
    CATEGORY = "ZSQ/Image"

    def execute(self, image, width, height, position, x_offset, y_offset):
        _, oh, ow, _ = image.shape

        width = min(ow, width)
        height = min(oh, height)

        if "center" in position:
            x = round((ow-width) / 2)
            y = round((oh-height) / 2)
        if "top" in position:
            y = 0
        if "bottom" in position:
            y = oh-height
        if "left" in position:
            x = 0
        if "right" in position:
            x = ow-width

        x += x_offset
        y += y_offset

        x2 = x+width
        y2 = y+height

        if x2 > ow:
            x2 = ow
        if x < 0:
            x = 0
        if y2 > oh:
            y2 = oh
        if y < 0:
            y = 0

        image = image[:, y:y2, x:x2, :]

        return(image, x, y, )

# 图像尺寸
class imageSize:
  def __init__(self):
    pass

  @classmethod
  def INPUT_TYPES(cls):
    return {
      "required": {
        "image": ("IMAGE",),
      }
    }

  RETURN_TYPES = ("INT", "INT")
  RETURN_NAMES = ("width_int", "height_int")
  OUTPUT_NODE = True
  FUNCTION = "GetImageSize"

  CATEGORY = "ZSQ/Image"

  def GetImageSize(self, image):
    _, raw_H, raw_W, _ = image.shape

    width = int(raw_W)
    height = int(raw_H)

    if width is not None and height is not None:
      return (width, height,)
    else:
      return (0, 0)

class imageResize:
  upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]
  crop_methods = ["disabled", "center"]
  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
        "image": ("IMAGE",),
        "side": (["WIDTH", "HEIGHT"], { "default": "WIDTH"}),
        "max_side": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8,}),
        "ratio": ("FLOAT", {"default": 1.0, "step": 0.1, "min": 0.0, "max": 8.0, "display": "slider"}),
        "upscale_method": (s.upscale_methods,),
        "crop": (s.crop_methods,),
      }
    }

  RETURN_TYPES = ("IMAGE", "INT", "INT",)
  RETURN_NAMES = ("image", "width", "height",)
  FUNCTION = "execute"
  CATEGORY = "ZSQ/Image"

  def execute(self, image, side,max_side, ratio, crop, upscale_method):
      _, oh, ow, _ = image.shape
      if ratio == 0:
            out = image
            return (out,ow,oh)
      else:
          samples = image.movedim(-1,1)
          if side  == "WIDTH":
            side_length = ow
            width = side_length * ratio
            width = min(max_side, width)
            height = width * float(oh/ow)
          else:
            side_length = oh
            height = side_length * ratio 
            height = min(max_side, height)
            width = height * float(ow/oh)
          width = int(width)
          height = int(height)
          out = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
          out = out.movedim(1,-1)

      return (out,width,height)

# 图像缩放
class imageScaleDown:
  crop_methods = ["disabled", "center"]

  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
        "images": ("IMAGE",),
        "width": (
          "INT",
          {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1},
        ),
        "height": (
          "INT",
          {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1},
        ),
        "crop": (s.crop_methods,),
      }
    }

  RETURN_TYPES = ("IMAGE",)
  CATEGORY = "ZSQ/Image"
  FUNCTION = "image_scale_down"

  def image_scale_down(self, images, width, height, crop):
    if crop == "center":
      old_width = images.shape[2]
      old_height = images.shape[1]
      old_aspect = old_width / old_height
      new_aspect = width / height
      x = 0
      y = 0
      if old_aspect > new_aspect:
        x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
      elif old_aspect < new_aspect:
        y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
      s = images[:, y: old_height - y, x: old_width - x, :]
    else:
      s = images

    results = []
    for image in s:
      img = tensor2pil(image).convert("RGB")
      img = img.resize((width, height), Image.LANCZOS)
      results.append(pil2tensor(img))

    return (torch.cat(results, dim=0),)

# 图像缩放比例
class imageScaleDownBy(imageScaleDown):
  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
        "images": ("IMAGE",),
        "scale_by": (
          "FLOAT",
          {"default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01,"display": "slider"},
        ),
      }
    }

  RETURN_TYPES = ("IMAGE",)
  CATEGORY = "ZSQ/Image"
  FUNCTION = "image_scale_down_by"

  def image_scale_down_by(self, images, scale_by):
    width = images.shape[2]
    height = images.shape[1]
    new_width = int(width * scale_by)
    new_height = int(height * scale_by)
    return self.image_scale_down(images, new_width, new_height, "center")

# 图像比率
class imageRatio:
  def __init__(self):
    pass

  @classmethod
  def INPUT_TYPES(cls):
    return {
      "required": {
        "image": ("IMAGE",),
      }
    }

  RETURN_TYPES = ("INT", "INT", "FLOAT", "FLOAT")
  RETURN_NAMES = ("width_ratio_int", "height_ratio_int", "width_ratio_float", "height_ratio_float")
  OUTPUT_NODE = True
  FUNCTION = "image_ratio"

  CATEGORY = "ZSQ/Image"

  def gcf(self, a, b):
    while b:
      a, b = b, a % b
    return a

  def image_ratio(self, image):
    _, raw_H, raw_W, _ = image.shape

    width = raw_W
    height = raw_H

    ratio = self.gcf(width, height)

    if width is not None and height is not None:
      width_ratio = width // ratio
      height_ratio = height // ratio
      return(width_ratio, height_ratio, width_ratio, height_ratio)
    else:
      width_ratio = 0
      height_ratio = 0
      return(0, 0, 0.0, 0.0)


# 图像保存 (简易JPG)
class imageSaveSimple(SaveJpgImage):

  def __init__(self):
    pass
  @classmethod
  def INPUT_TYPES(s):
    return {"required":
              {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "only_preview": ("BOOLEAN", {"default": False}),
              },
            }

  RETURN_TYPES = ()
  FUNCTION = "save"
  OUTPUT_NODE = True
  CATEGORY = "ZSQ/Image"

  def save(self, images, filename_prefix="ComfyUI", only_preview=False):
    if only_preview:
      self.output_dir = folder_paths.get_temp_directory()
      self.type = "temp"
      self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
      self.quality = 85
      return self.save_images(images, filename_prefix)
    else:
      self.output_dir = folder_paths.get_output_directory()
      self.type = "output"
      self.prefix_append = ""
      self.quality = 85
      return self.save_images(images, filename_prefix)
  

# # easy use 图像批次合并
class JoinImageBatch:
  """Turns an image batch into one big image."""

  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
        "images": ("IMAGE",),
        "mode": (("horizontal", "vertical"), {"default": "horizontal"}),
      },
    }

  RETURN_TYPES = ("IMAGE",)
  RETURN_NAMES = ("image",)
  FUNCTION = "join"
  CATEGORY = "ZSQ/Image"

  def join(self, images, mode):
    n, h, w, c = images.shape
    image = None
    if mode == "vertical":
      # for vertical we can just reshape
      image = images.reshape(1, n * h, w, c)
    elif mode == "horizontal":
      # for horizontal we have to swap axes
      image = torch.transpose(torch.transpose(images, 1, 2).reshape(1, n * w, h, c), 1, 2)
    return (image,)

# easy use
class imageTilesFromBatch:
  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
        "tiles": ("IMAGE",),
        "masks": ("MASK",),
        "overlap": ("OVERLAP",),
        "index":("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
      },
    }

  RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
  RETURN_NAMES = ("image", "mask", "x", "y")
  FUNCTION = "doit"
  CATEGORY = "ZSQ/Image"

  def imageFromBatch(self, image, batch_index, length=1):
    s_in = image
    batch_index = min(s_in.shape[0] - 1, batch_index)
    length = min(s_in.shape[0] - batch_index, length)
    s = s_in[batch_index:batch_index + length].clone()
    return s

  def maskFromBatch(self, mask, start, length=1):
    if length > mask.shape[0]:
        length = mask.shape[0]
    start = min(start, mask.shape[0]-1)
    length = min(mask.shape[0]-start, length)
    return mask[start:start + length]

  def doit(self, tiles, masks, overlap, index):
    tile = self.imageFromBatch(tiles, index)
    mask = self.maskFromBatch(masks, index)
    overlap_w, overlap_h, tile_w, tile_h, tiles_rows, tiles_cols = overlap

    x = tile_w * (index % tiles_cols) - overlap_w if (index % tiles_cols) > 0 else 0
    y = tile_h * (index // tiles_cols) - overlap_h if tiles_rows > 1 and index > tiles_cols - 1 else 0

    return (tile, mask, x, y)


# easy use
class imagesSplitImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
          "required": {
              "images": ("IMAGE",),
          }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("image1", "image2", "image3", "image4", "image5")
    FUNCTION = "split"
    CATEGORY = "ZSQ/Image"

    def split(self, images,):
      new_images = torch.chunk(images, len(images), dim=0)
      return new_images
# easy use
class imageConcat:
  @classmethod
  def INPUT_TYPES(s):
    return {"required": {
      "image1": ("IMAGE",),
      "image2": ("IMAGE",),
      "direction": (['right','down','left','up',],{"default": 'right'}),
      "match_image_size": ("BOOLEAN", {"default": False}),
    }}

  RETURN_TYPES = ("IMAGE",)
  FUNCTION = "concat"
  CATEGORY = "ZSQ/Image"

  def concat(self, image1, image2, direction, match_image_size):
    if image1 is None:
      return (image2,)
    elif image2 is None:
      return (image1,)
    if match_image_size:
      image2 = torch.nn.functional.interpolate(image2, size=(image1.shape[2], image1.shape[3]), mode="bilinear")
    if direction == 'right':
      row = torch.cat((image1, image2), dim=2)
    elif direction == 'down':
      row = torch.cat((image1, image2), dim=1)
    elif direction == 'left':
      row = torch.cat((image2, image1), dim=2)
    elif direction == 'up':
      row = torch.cat((image2, image1), dim=1)
    return (row,)

# easy use
class imageDetailTransfer:
  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
        "target": ("IMAGE",),
        "source": ("IMAGE",),
        "mode": (["add", "multiply", "screen", "overlay", "soft_light", "hard_light", "color_dodge", "color_burn", "difference", "exclusion", "divide",],{"default": "add"}),
        "blur_sigma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 100.0, "step": 0.01}),
        "blend_factor": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001, "round": 0.001}),
        "image_output": (["Hide", "Preview", "Save", "Hide/Save"], {"default": "Preview"}),
        "save_prefix": ("STRING", {"default": "ComfyUI"}),
      },
      "optional": {
        "mask": ("MASK",),
      },
      "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
    }

  RETURN_TYPES = ("IMAGE",)
  RETURN_NAMES = ("image",)
  OUTPUT_NODE = True
  FUNCTION = "execute"
  CATEGORY = "ZSQ/Image"


  def execute(self, target, source, mode, blur_sigma, blend_factor, image_output, save_prefix, mask=None, prompt=None, extra_pnginfo=None):
    batch_size, height, width, _ = source.shape
    device = comfy.model_management.get_torch_device()
    target_tensor = target.permute(0, 3, 1, 2).clone().to(device)
    source_tensor = source.permute(0, 3, 1, 2).clone().to(device)

    if target.shape[1:] != source.shape[1:]:
      target_tensor = comfy.utils.common_upscale(target_tensor, width, height, "bilinear", "disabled")
    if mask is not None and target.shape[1:] != mask.shape[1:]:
      mask = mask.unsqueeze(1)
      mask = F.interpolate(mask, size=(height, width), mode="bilinear")
      mask = mask.squeeze(1)

    if source.shape[0] < batch_size:
      source = source[0].unsqueeze(0).repeat(batch_size, 1, 1, 1)

    kernel_size = int(6 * int(blur_sigma) + 1)

    gaussian_blur = GaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=(blur_sigma, blur_sigma))

    blurred_target = gaussian_blur(target_tensor)
    blurred_source = gaussian_blur(source_tensor)

    if mode == "add":
      new_image = (source_tensor - blurred_source) + blurred_target
    elif mode == "multiply":
      new_image = source_tensor * blurred_target
    elif mode == "screen":
      new_image = 1 - (1 - source_tensor) * (1 - blurred_target)
    elif mode == "overlay":
      new_image = torch.where(blurred_target < 0.5, 2 * source_tensor * blurred_target,
                               1 - 2 * (1 - source_tensor) * (1 - blurred_target))
    elif mode == "soft_light":
      new_image = (1 - 2 * blurred_target) * source_tensor ** 2 + 2 * blurred_target * source_tensor
    elif mode == "hard_light":
      new_image = torch.where(source_tensor < 0.5, 2 * source_tensor * blurred_target,
                               1 - 2 * (1 - source_tensor) * (1 - blurred_target))
    elif mode == "difference":
      new_image = torch.abs(blurred_target - source_tensor)
    elif mode == "exclusion":
      new_image = 0.5 - 2 * (blurred_target - 0.5) * (source_tensor - 0.5)
    elif mode == "color_dodge":
      new_image = blurred_target / (1 - source_tensor)
    elif mode == "color_burn":
      new_image = 1 - (1 - blurred_target) / source_tensor
    elif mode == "divide":
      new_image = (source_tensor / blurred_source) * blurred_target
    else:
      new_image = source_tensor

    new_image = torch.lerp(target_tensor, new_image, blend_factor)
    if mask is not None:
      mask = mask.to(device)
      new_image = torch.lerp(target_tensor, new_image, mask)
    new_image = torch.clamp(new_image, 0, 1)
    new_image = new_image.permute(0, 2, 3, 1).cpu().float()

    results = SaveImageUI(new_image, save_prefix, image_output, prompt, extra_pnginfo)

    if image_output in ("Hide", "Hide/Save"):
      return {"ui": {},
              "result": (new_image,)}

    return {"ui": {"images": results},
            "result": (new_image,)}
  

class ImageAddText:
    @classmethod
    def INPUT_TYPES(s):
        fonts = []
        fonts_dir = FONT_STYLES_DIR
        for file_name in os.listdir(fonts_dir):
            file = os.path.join(fonts_dir, file_name)
            if os.path.isfile(file) and (file_name.endswith(".ttf") or file_name.endswith(".otf") or file_name.endswith(".ttc")):
                fonts.append(file_name)
        return {"required": {
            "image":("IMAGE",),  
            "text_x": ("INT", {"default": 10, "min": 0, "max": 4096, "step": 1}),
            "text_y": ("INT", {"default": 10, "min": 0, "max": 4096, "step": 1}),
            "line_height": ("INT", {"default": 48, "min": 0, "max": 4096, "step": 1}),
            "font_size": ("INT", {"default": 32, "min": 0, "max": 4096, "step": 1}),            
            "font_r": ("INT", {"default": 0, "min": 0, "max":255, "step": 1, "display": "slider"}),
            "font_g": ("INT", {"default": 0, "min": 0, "max":255, "step": 1, "display": "slider"}),
            "font_b": ("INT", {"default": 0, "min": 0, "max":255, "step": 1, "display": "slider"}),
            "stroke_width": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            "stroke_r": ("INT", {"default": 0, "min": 0, "max":255, "step": 1, "display": "slider"}),
            "stroke_g": ("INT", {"default": 0, "min": 0, "max":255, "step": 1, "display": "slider"}),
            "stroke_b": ("INT", {"default": 0, "min": 0, "max":255, "step": 1, "display": "slider"}),
            "font": (fonts,),
            "text": ("STRING", {"default": "Text","multiline": True}),
            "image_output": (["Hide", "Preview", "Save", "Hide/Save"], {"default": "Preview"}),
            "save_prefix": ("STRING", {"default": "ComfyUI"}),
            },
          }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_NODE = True
    FUNCTION = "addlabel"
    CATEGORY = "ZSQ/Image"
    
    def addlabel(self, image, text_x, text_y, text, line_height, font,font_size,font_r,font_g,font_b,
                 stroke_width, stroke_r, stroke_g, stroke_b,save_prefix, image_output):
        _, oh, ow, _ = image.shape
        if text_x >= ow or text_y >= oh:
            return image
        font_path = os.path.join(FONT_STYLES_DIR,font)
        res_images = []
        for i in image:
          pil_image =tensor2pil(torch.unsqueeze(i, 0)).convert('RGB') 
          font_color = (font_r, font_g,font_b)
          stroke_color = (stroke_r, stroke_g, stroke_b) 
          for line in text.split('\n'):
            pil_image= self.write2image(pil_image,font_path,line,text_x, text_y,font_color,font_size,stroke_width,stroke_color)
            text_y += line_height
          res_images.append(pil2tensor(pil_image))        
        
        new_image = torch.cat(res_images, dim=0)
        results = SaveImageUI(new_image, save_prefix, image_output)

        if image_output in ("Hide", "Hide/Save"):
            return {"ui": {},
                    "result": (new_image,)}

        return {"ui": {"images": results},
                "result": (new_image,)}
    
    def write2image(self, img,font_path,text,x, y,font_color,font_size,stroke_width,stroke_color):
      """将文字写入图片"""    
      # 创建一个可以在图像上绘图的对象
      draw = ImageDraw.Draw(img)
      # 定义字体和字号
      font = ImageFont.truetype(font_path, font_size)
      # 定义文字内容和位置    
      # 在图像上添加文字    
      draw.text((x, y), text, font=font,fill=font_color,stroke_width=stroke_width, stroke_fill=stroke_color)
      # 显示图像
      return img

#图像锐化    
class imageSharpen:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "image": ("IMAGE", ),  #
                "percent": ("INT", {"default": 150, "min": 100, "max": 300, "step": 1, "display": "slider"}),
                "threshold": ("INT", {"default": 30, "min": 0, "max": 100, "step": 1, "display": "slider"}),
                "radius": ("FLOAT", {"default": 1, "min": 0.00, "max": 5, "step": 0.1, "display": "slider"}),
                "image_output": (["Hide", "Preview", "Save", "Hide/Save"], {"default": "Preview"}),
                "save_prefix": ("STRING", {"default": "ComfyUI"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_NODE = True
    FUNCTION = 'execute'
    CATEGORY = 'ZSQ/Image'

    def execute(self, image,radius,percent,threshold,save_prefix,image_output):
        res_images = []
        for i in image:
          pil_image =tensor2pil(torch.unsqueeze(i, 0)).convert('RGB')          
          res_images.append(pil2tensor(pil_image.filter(ImageFilter.UnsharpMask(radius=radius,percent=percent,threshold=threshold))))
        
        new_image = torch.cat(res_images, dim=0)
        results = SaveImageUI(new_image, save_prefix, image_output)

        if image_output in ("Hide", "Hide/Save"):
            return {"ui": {},
                    "result": (new_image,)}

        return {"ui": {"images": results},
                "result": (new_image,)}

#高斯模糊
class imageGaussianBlur:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "image": ("IMAGE", ),  #
                "radius": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5, "step": 0.01, "display": "slider"}),
                "image_output": (["Hide", "Preview", "Save", "Hide/Save"], {"default": "Preview"}),
                "save_prefix": ("STRING", {"default": "ComfyUI"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_NODE = True
    FUNCTION = 'execute'
    CATEGORY = 'ZSQ/Image'

    def execute(self, image,radius,save_prefix,image_output):
        res_images = []
        for i in image:
          pil_image =tensor2pil(torch.unsqueeze(i, 0)).convert('RGB')          
          res_images.append(pil2tensor(pil_image.filter(ImageFilter.GaussianBlur(radius=radius))))
        new_image = torch.cat(res_images, dim=0)
        results = SaveImageUI(new_image, save_prefix, image_output)

        if image_output in ("Hide", "Hide/Save"):
            return {"ui": {},
                    "result": (new_image,)}

        return {"ui": {"images": results},
                "result": (new_image,)}
    
#图像旋转
class imageRotate:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "image": ("IMAGE", ),  
                "expand": ("BOOLEAN", {"default": True}), #是否扩展输出图像的大小以容纳旋转后的图像
                "angle": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1, "display": "slider"}),
                "image_output": (["Hide", "Preview", "Save", "Hide/Save"], {"default": "Preview"}),
                "save_prefix": ("STRING", {"default": "ComfyUI"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_NODE = True
    FUNCTION = 'execute'
    CATEGORY = 'ZSQ/Image'

    def execute(self, image,save_prefix,image_output,angle=0.0, expand=True):
        res_images = []
        for i in image:
          pil_image =tensor2pil(torch.unsqueeze(i, 0)).convert('RGB')          
          res_images.append(pil2tensor(pil_image.rotate(angle=angle, expand=expand)))

        new_image = torch.cat(res_images, dim=0)
        results = SaveImageUI(new_image, save_prefix, image_output)

        if image_output in ("Hide", "Hide/Save"):
            return {"ui": {},
                    "result": (new_image,)}

        return {"ui": {"images": results},
                "result": (new_image,)} 

#图像翻转
class imageFlip:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "image": ("IMAGE", ),  
                "mode": (["LEFT_RIGHT", "TOP_BOTTOM", "LEFT_RIGHT_AND_TOP_BOTTOM"], {"default": "LEFT_RIGHT"}), #翻转模式，可以是左右翻转或上下翻转
                "image_output": (["Hide", "Preview", "Save", "Hide/Save"], {"default": "Preview"}),
                "save_prefix": ("STRING", {"default": "ComfyUI"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_NODE = True
    FUNCTION = 'execute'
    CATEGORY = 'ZSQ/Image'
    def execute(self,image,mode,save_prefix,image_output):
        res_images = []
        for i in image:
          pil_image =tensor2pil(torch.unsqueeze(i, 0)).convert('RGB')
          if mode == "LEFT_RIGHT":
            flipped_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
          elif mode == "TOP_BOTTOM":
            flipped_image = pil_image.transpose(Image.FLIP_TOP_BOTTOM)
          else: #左右和上下翻转
            flipped_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
            flipped_image = flipped_image.transpose(Image.FLIP_TOP_BOTTOM)
          res_images.append(pil2tensor(flipped_image))
        new_image = torch.cat(res_images, dim=0)
        results = SaveImageUI(new_image, save_prefix, image_output)

        if image_output in ("Hide", "Hide/Save"):
            return {"ui": {},
                    "result": (new_image,)}

        return {"ui": {"images": results},
                "result": (new_image,)}     
#滤镜
class imageFilter:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "image": ("IMAGE", ),  
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10, "step": 0.01, "display": "slider"}),
                "brightness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10, "step": 0.01, "display": "slider"}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10, "step": 0.01, "display": "slider"}),
                "filter": (["NONE","EMBOSS","VINTAGE","SKETCH","CONTOUR","NOISE", "GRAYSCALE","INVERT"], {"default": "NONE"}),
                "image_output": (["Hide", "Preview", "Save", "Hide/Save"], {"default": "Preview"}),
                "save_prefix": ("STRING", {"default": "ComfyUI"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_NODE = True
    FUNCTION = 'execute'
    CATEGORY = 'ZSQ/Image'

    def execute(self,image,contrast,brightness,saturation,filter,save_prefix,image_output):
        res_images = []
        for i in image:
            pil_image =tensor2pil(torch.unsqueeze(i, 0)).convert('RGB') 
            #对比度调整
            if contrast != 0.0:
                pil_image = ImageEnhance.Contrast(pil_image).enhance(contrast)
            #亮度调整
            if brightness != 0.0:
                pil_image = ImageEnhance.Brightness(pil_image).enhance(brightness)
            #饱和度调整
            if saturation != 0.0:
                pil_image = ImageEnhance.Color(pil_image).enhance(saturation)    

            #滤镜效果          
            if filter == "EMBOSS":         
                res_images.append(pil2tensor(pil_image.filter(ImageFilter.EMBOSS)))
            elif filter == "CONTOUR":
                res_images.append(pil2tensor(pil_image.filter(ImageFilter.CONTOUR)))
            elif filter == "NOISE":
                res_images.append(pil2tensor(image_gaussian_noise(pil_image)))
            elif filter == "GRAYSCALE":
                res_images.append(pil2tensor(pil_image.convert('L')))
            elif filter == "VINTAGE":
                res_images.append(pil2tensor(image_vintage_photo(pil_image)))
            elif filter == "SKETCH":
                res_images.append(pil2tensor(image_sketch_photo(pil_image)))
            elif filter == "INVERT":
                res_images.append(pil2tensor(ImageOps.invert(pil_image)))
            else:
                res_images.append(pil2tensor(pil_image))

        new_image = torch.cat(res_images, dim=0)
        results = SaveImageUI(new_image, save_prefix, image_output)

        if image_output in ("Hide", "Hide/Save"):
            return {"ui": {},
                    "result": (new_image,)}

        return {"ui": {"images": results},
                "result": (new_image,)}  
    
#色调调整    
class imageHug:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "image": ("IMAGE", ),  
                "hue_shift": ("FLOAT", {"default": 0.0, "min":-180.0, "max": 180.0, "step": 0.01, "display": "slider"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'execute'
    CATEGORY = 'ZSQ/Image'

    def execute(self, image,hue_shift):
        res_images = []
        for i in image:
          pil_image =tensor2pil(torch.unsqueeze(i, 0)).convert('HSV')    
          h, s, v = pil_image.split()
          h = h.point(lambda i: (i + hue_shift) % 256)
          hsv_image = Image.merge('HSV', (h, s, v))
          res_images.append(pil2tensor(hsv_image.convert('RGB')))

        return (torch.cat(res_images, dim=0),)
    
  #RGB调整    
class imageRGB:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "image": ("IMAGE", ),  
                "Red": ("FLOAT", {"default": 1.0, "min":0.0, "max": 10.0, "step": 0.1, "display": "slider"}),
                "Green": ("FLOAT", {"default": 1.0, "min":0.0, "max": 10.0, "step": 0.1, "display": "slider"}),
                "Blue": ("FLOAT", {"default": 1.0, "min":0.0, "max": 10.0, "step": 0.1, "display": "slider"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'execute'
    CATEGORY = 'ZSQ/Image'

    def execute(self, image,Red,Blue,Green):
        res_images = []
        for i in image:
          pil_image =tensor2pil(torch.unsqueeze(i, 0)).convert('RGB')    
          # 分离图像的RGB通道
          r, g, b = pil_image.split()
          # 降低红色和绿色通道的强度，增强蓝色通道的强度，以营造蓝调效果
          r = ImageEnhance.Brightness(r).enhance(Red)
          g = ImageEnhance.Brightness(g).enhance(Green)
          b = ImageEnhance.Brightness(b).enhance(Blue)
          # 合并调整后的通道
          bluish_image = Image.merge('RGB', (r, g, b))
          res_images.append(pil2tensor(bluish_image))

        return (torch.cat(res_images, dim=0),)

class ImageEmpty:
    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                { "width": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                  "height": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                  "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                  "color_r": ("INT", {"default": 0, "min": 0, "max":255, "step": 1, "display": "slider"}),
                  "color_g": ("INT", {"default": 0, "min": 0, "max":255, "step": 1, "display": "slider"}),
                  "color_b": ("INT", {"default": 0, "min": 0, "max":255, "step": 1, "display": "slider"}),
                              }}
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"

    CATEGORY = "ZSQ/Image"

    def execute(self, width, height, batch_size=1, color_r=0,color_g=0,color_b=0):
        r = torch.full([batch_size, height, width, 1], (color_r/255.0))
        g = torch.full([batch_size, height, width, 1], (color_g/255.0))
        b = torch.full([batch_size, height, width, 1], (color_b/255.0))
        return (torch.cat((r, g, b), dim=-1), )