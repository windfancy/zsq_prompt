from .zsq_prompt import PortraitStyler
from .zsq_llm import LLMText,LLMImage
from .zsq_loader import checkpoint_sampler,zsqcheckpoint,zsqcontrolnetStack,zsqcontrolnetstack_2,zsqcontrolnet,zsqloraStack,zsqloraStack_2,zsqsampler
from .zsq_utils import ZSQ_PixelLatent,ZSQ_RatioLatent,DoubleCLIPEncode

directory = ".\web"

WEB_DIRECTORY = directory


NODE_CLASS_MAPPINGS = {
    "PortraitStyler": PortraitStyler,
    "ZSQPixelLatent": ZSQ_PixelLatent,
    "ZSQRatioLatent": ZSQ_RatioLatent,
    "DoubleCLIPEncode":DoubleCLIPEncode,
    "LLMText": LLMText,
    "LLMImage": LLMImage,
    "zsqcheckpoint": zsqcheckpoint,
    "controlnetStack": zsqcontrolnetStack,
    "controlnetStack_2": zsqcontrolnetstack_2,
    "zsqcontrolnet": zsqcontrolnet,
    "loraStack": zsqloraStack,
    "loraStack_2": zsqloraStack_2,
    "zsqsampler": zsqsampler,
    "checkpoint_sampler": checkpoint_sampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PortraitStyler": "Portrait Styler",
    "ZSQPixelLatent": "Pixel Latent",
    "ZSQRatioLatent": "Ratio Latent",
    "DoubleCLIPEncode":"Double CLIP Encode",
    "LLMText": "LLM Text",
    "LLMImage": "LLM Image",
    "zsqcheckpoint": "Simple Checkpoint",
    "controlnetStack": "Controlnet Stack",
    "controlnetStack_2": "Simple Controlnet Stack",
    "zsqcontrolnet": "Simple Controlnet",
    "loraStack": "Lora Stack",
    "loraStack_2": "Simple Lora Stack",
    "zsqsampler": "Simple Sampler",
    "checkpoint_sampler": "Checkpoint & Sampler"
}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]








