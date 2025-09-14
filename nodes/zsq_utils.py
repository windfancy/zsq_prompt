import torch
import re,random
from ..config import RESOLUTION_STRINGS,MAX_TEXT_NUM
import comfy.model_management
from .libs.latent import reshape_latent_to

class IndexString:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "index": ("INT", {"default": 0, "min": 0, "max": MAX_TEXT_NUM-1, "step": 1}),
            },
            "optional": {
            }
        }
        for i in range(MAX_TEXT_NUM):
            inputs["optional"]["text%d" % i] = ("STRING", {"forceInput": True})
        return inputs

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "execute"

    CATEGORY = "ZSQ/Stinrg"

    def execute(self, index, **kwargs):
        key = "text%d" % index
        if kwargs.get(key, None) is None:
            return [key]
        return (kwargs[key],)

class ConnectionString:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {        
                "separator": ("STRING", {"default": ","}),       
            },
            "optional": {
            }
        }
        for i in range(MAX_TEXT_NUM):
            inputs["optional"]["text%d" % i] = ("STRING", {"forceInput": True})
        return inputs

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "execute"

    CATEGORY = "ZSQ/Stinrg"

    def execute(self,separator, **kwargs):
        value = ""
        for i in range(MAX_TEXT_NUM):
            if kwargs.get("text%d" % i, None) is not None:
                value = value + kwargs["text%d" % i] + separator
        return (value,)

# 字符串
class OptionString:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"input": ("STRING", {"multiline": True,"default": ""}),
                         "mode": (["Replace", "Connection","Regular"], {"default": "Replace"}),
                         "str1": ("STRING", {"default": ""}),
                         "str2": ("STRING", {"default": ""}),
                         },            
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "execute"
    CATEGORY = "ZSQ/Stinrg"

    def execute(self, input, str1, str2, mode):
        value = ""
        if mode == "Replace":
            value = input.replace(str1, str2)
        elif mode == "Connection":
            value = f"{input},{str1},{str2}"
        elif mode == "Regular":
            #input:"Hello 123, this is a test 456."  str1: \d+ str2: "NUMBER"
            #value: "Hello NUMBER, this is a test NUMBER."
            pattern = str1  # 假设 str1 是正则表达式模式
            replacement = str2  # 假设 str2 是替换字符串
            value = re.sub(pattern, replacement, input)
        else:
            value = input
        return (value,)

# 字符串输入
class StringInput:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"Input": ("STRING", {"multiline": True,"default": ""})},
            
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "execute"
    CATEGORY = "ZSQ/Stinrg"

    def execute(self, Input):
        return (Input,)

class IntMathOperation:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value1": ("INT", { "default": 0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "step": 1 }),
                "value2": ("INT", { "default": 0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "step": 1 }),
                "operation": (["add", "subtract", "multiply", "divide","max","min", "modulus", "random"], {"default": "add"}),
            },
        }

    RETURN_TYPES = ("INT","INT","INT", )
    RETURN_NAMES = ("result","value1","value2",)
    FUNCTION = "execute"
    CATEGORY = "ZSQ/utils"

    def execute(self, value1, value2, operation):
        output = None
        if operation == "add":
            output = value1 + value2
        elif operation == "subtract":
            output = value1 - value2
        elif operation == "multiply":
            output = value1 * value2
        elif operation == "divide":
            if value2 == 0: return (0,value1,value2,)
            output = value1 / value2
        elif operation == "max":
            output = max(value1, value2)
        elif operation == "min":
            output = min(value1, value2) 
        elif operation == "modulus":
            if value2 == 0: return (0,value1,value2,)
            output = value1 % value2
        elif operation == "random":
            if value1 > value2:
                output = random.randint(value2, value1)
            else:                
                output = random.randint(value1, value2)
        return (int(output), value1,value2,)

class FloatMathOperation:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value1": ("FLOAT", { "default": 0.0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "step": 0.001 }),
                "value2": ("FLOAT", { "default": 0.0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "step": 0.001 }),
                "operation": (["add", "subtract", "multiply", "divide","max","min", "random"], {"default": "add"}),
            },
        }

    RETURN_TYPES = ("FLOAT","FLOAT","FLOAT")
    RETURN_NAMES = ("result","value1","value2")
    FUNCTION = "execute"
    CATEGORY = "ZSQ/utils"

    def execute(self, value1, value2, operation):
        output = None
        if operation == "add":
            output = value1 + value2
        elif operation == "subtract":
            output = value1 - value2
        elif operation == "multiply":
            output = value1 * value2
        elif operation == "divide":
            if value2 == 0: return (0,value1,value2)
            output = value1 / value2
        elif operation == "max":
            output = max(value1, value2)
        elif operation == "min":
            output = min(value1, value2) 
        elif operation == "random":
            if value1 > value2:
                output = random.uniform(value2, value1)
            else:
                output = random.uniform(value1,value2)
        return (float(output),value1,value2)

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
                    "Width": ([384,512,768,1024,1280],{"default": 512}), 
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
    
class ShowText:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    FUNCTION = "notify"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)

    CATEGORY = "ZSQ/Stinrg"

    def notify(self, text):
        return {"ui": {"text": text}, "result": (text,)}

class ShowINT:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input": ("INT", {"forceInput": True}),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    FUNCTION = "notify"
    OUTPUT_NODE = True


    CATEGORY = "ZSQ/Stinrg"

    def notify(self, input):
        text = str(input)
        return {"ui": {"text": text}, "result": (text,)}
    
class ShowFloat:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input": ("FLOAT", {"forceInput": True}),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    FUNCTION = "notify"
    OUTPUT_NODE = True

    CATEGORY = "ZSQ/Stinrg"

    def notify(self, input):
        text = str(input)
        return {"ui": {"text": text}, "result": (text,)}
    
class ZsqLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    { 
                    "samples1": ("LATENT",), 
                    "samples2": ("LATENT",),
                    "mode": (["add", "subtract", "multiply", "interpolate"],),
                    "multiply_multiplier": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,"display": "slider"}),
                    "interpolate_ratio": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,"display": "slider"})
                     }
                }

    RETURN_TYPES = ("LATENT",)
            
    FUNCTION = "execute"
    CATEGORY = "ZSQ/utils"

    def execute(self, samples1, samples2, mode, multiply_multiplier, interpolate_ratio):
        if mode == "add":
            return self.add(samples1, samples2)
        elif mode == "subtract":
            return self.subtract(samples1, samples2)
        elif mode == "multiply":
            return self.multiply(samples1, multiply_multiplier)
        elif mode == "interpolate":
            return self.interpolate(samples1, interpolate_ratio)
        else:
            raise ValueError("Invalid mode: {}".format(mode))

    def add(self, samples1, samples2):
        samples_out = samples1.copy()
        s1 = samples1["samples"]
        s2 = samples2["samples"]
        s2 = reshape_latent_to(s1.shape, s2)
        samples_out["samples"] = s1 + s2
        return (samples_out,)

    def subtract(self, samples1, samples2):
        samples_out = samples1.copy()
        s1 = samples1["samples"]
        s2 = samples2["samples"]
        s2 = reshape_latent_to(s1.shape, s2)
        samples_out["samples"] = s1 - s2
        return (samples_out,)
    
    def multiply(self, samples1, multiply_multiplier):
        samples_out = samples1.copy()
        s1 = samples1["samples"]
        samples_out["samples"] = s1 * multiply_multiplier
        return (samples_out,)
    
    def interpolate(self, samples1, samples2, interpolate_ratio):
        samples_out = samples1.copy()
        s1 = samples1["samples"]
        s2 = samples2["samples"]
        s2 = reshape_latent_to(s1.shape, s2)
        m1 = torch.linalg.vector_norm(s1, dim=(1))
        m2 = torch.linalg.vector_norm(s2, dim=(1))
        s1 = torch.nan_to_num(s1 / m1)
        s2 = torch.nan_to_num(s2 / m2)
        t = (s1 * interpolate_ratio + s2 * (1.0 - interpolate_ratio))
        mt = torch.linalg.vector_norm(t, dim=(1))
        st = torch.nan_to_num(t / mt)
        samples_out["samples"] = st * (m1 * interpolate_ratio + m2 * (1.0 - interpolate_ratio))
        return (samples_out,)
    
class MaskToImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "mask": ("MASK",),
                    "image_output": (["Hide", "Preview"], {"default": "Preview"}),
                }
                
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "mask_to_image"
    CATEGORY = "ZSQ/mask"

    def mask_to_image(self, mask,image_output):
        result = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
        if image_output in ("Hide"):
            return {"ui": {}, "result": (result,)}
        return {"ui": {"images": result}, "result": (result,)}