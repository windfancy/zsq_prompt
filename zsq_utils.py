import torch
from .config import RESOLUTION_STRINGS
import comfy.model_management

class ZSQ_PixelLatent:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()
    @classmethod
    def INPUT_TYPES(self):
        return {"required": 
                {
                    "resolution": (RESOLUTION_STRINGS,{"default": "1024 x 1024"}),       
                    "batch_size": ("INT", {"default": 1, "step": 1, "min": 1, "max": 20, "display": "slider"}),
                 },
            }
        
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "run"
    CATEGORY = "ZSQ/utils"

    def run(self, resolution, batch_size):
        width, height = resolution.split(" x ")
        latent = torch.zeros([batch_size, 4, int(height) // 8, int(width) // 8], device=self.device)
        return ({"samples":latent}, )

class ZSQ_RatioLatent:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()
    @classmethod
    def INPUT_TYPES(self):
        return {"required": 
                {
                    "Width": (['384','512','768','1024','1280'],{"default": '512'}), 
                    "Ratio": (['1 : 1', '2 : 1', '3 : 2', '4 : 3', '16 : 9', '1 : 2', '2 : 3', '3 : 4', '9 : 16'],{"default": "1 : 1"}),       
                    "batch_size": ("INT", {"default": 1, "step": 1, "min": 1, "max": 20, "display": "slider"}),
                 },
            }
        
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "run"
    CATEGORY = "ZSQ/utils"

    def run(self, Width,Ratio, batch_size):
        width_Ratio, height_Ratio = Ratio.split(" : ")
        height = int(Width) * int(height_Ratio) // int(width_Ratio)
        width = int(Width)
        latent = torch.zeros([batch_size, 4, int(height) // 8, int(width) // 8], device=self.device)
        return ({"samples":latent}, )

class DoubleCLIPEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}),
                "negative": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}),  
                "clip": ("CLIP", {"tooltip": "The CLIP model used for encoding the text."})
            }
        }
    RETURN_TYPES = ("CONDITIONING","CONDITIONING",)
    RETURN_NAMES = ("positive","negative",)
    FUNCTION = "CLIPEncode"
    CATEGORY = "ZSQ/utils"
    
    def CLIPEncode(self, clip, positive, negative):
        positive_tokens = clip.tokenize(positive)
        negative_tokens = clip.tokenize(negative)
        positive_output = clip.encode_from_tokens(positive_tokens, return_pooled=True, return_dict=True)
        negative_output = clip.encode_from_tokens(negative_tokens, return_pooled=True, return_dict=True)
        positive_cond = positive_output.pop("cond")
        negative_cond = negative_output.pop("cond")
        return ([[positive_cond, positive_output]], [[negative_cond,negative_output]], )