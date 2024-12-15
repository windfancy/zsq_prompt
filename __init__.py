from .zsq_prompt import (
    PortraitStyler, 
    stylesPromptSelector,
    BatchPromptSelector,
    BatchPromptJson
    )
from .zsq_llm import LLMText,LLMImage
from .zsq_loader import (
    checkpoint_sampler,
    zsqcheckpoint,
    zsqcontrolnetStack,
    zsqcontrolnetstack_2,
    zsqcontrolnet,zsqloraStack,
    zsqloraStack_2,zsqsampler
    )   
from .zsq_utils import (
    ZSQ_PixelLatent,
    ZSQ_RatioLatent,
    DoubleCLIPEncode,
    ShowText,
    ShowINT,
    FloatMathOperation,
    IntMathOperation,
    OptionString,
    ConnectionString,
    StringInput,
    IndexString
    )
from .zsq_image import (
    imageCount,
    SaveJpgImage,
    imageCrop,
    imageResize,
    imageSize,
    imageScaleDown,
    imageScaleDownBy,
    imageRatio,
    imageSaveSimple,
    JoinImageBatch,
    imageTilesFromBatch,
    imagesSplitImage,
    imageConcat,
    imageDetailTransfer,
    ImageAddText,
    imageSharpen,
    imageGaussianBlur,
    imageRotate,
    imageFlip,
    imageFilter,
    imageHug,
    imageRGB,
    ImageEmpty
    )



directory = ".\web"

WEB_DIRECTORY = directory


NODE_CLASS_MAPPINGS = {
    #—————————————————————— zsq_prompt ———————————————————
    "PortraitStyler": PortraitStyler,
    "stylesSelector": stylesPromptSelector,
    "BatchPromptSelector": BatchPromptSelector,
    "BatchPromptJson": BatchPromptJson,
    #—————————————————————— zsq_llm ——————————————————————
    "LLMText": LLMText,
    "LLMImage": LLMImage,
    #—————————————————————— zsq_utils ————————————————————
    "ZSQPixelLatent": ZSQ_PixelLatent,
    "ZSQRatioLatent": ZSQ_RatioLatent,
    "DoubleCLIPEncode":DoubleCLIPEncode,
    "ZSQShowText": ShowText,
    "ZSQShowINT": ShowINT,
    "FloatMathOperation": FloatMathOperation,
    "IntMathOperation": IntMathOperation,
    "OptionString": OptionString,
    "ConnectionString": ConnectionString,
    "IndexString": IndexString,
    "StringInput": StringInput,
    #—————————————————————— zsq_loader ———————————————————
    "zsqcheckpoint": zsqcheckpoint,
    "controlnetStack": zsqcontrolnetStack,
    "controlnetStack_2": zsqcontrolnetstack_2,
    "zsqcontrolnet": zsqcontrolnet,
    "loraStack": zsqloraStack,
    "loraStack_2": zsqloraStack_2,
    "zsqsampler": zsqsampler,
    "checkpoint_sampler": checkpoint_sampler,
    #—————————————————————— zsq_image ————————————————————    
    "imageCount": imageCount,
    "imageResize": imageResize,
    "imageCrop": imageCrop,
    "imageSize": imageSize,
    "imageScaleDown": imageScaleDown,
    "imageScaleDownBy": imageScaleDownBy,
    "imageRatio": imageRatio,
    "imageSaveSimple":imageSaveSimple,
    "JoinImageBatch": JoinImageBatch,
    "imageTilesFromBatch": imageTilesFromBatch,
    "imagesSplitImage": imagesSplitImage,
    "imageConcat": imageConcat,
    "imageDetailTransfer": imageDetailTransfer,
    "SaveJpgImage": SaveJpgImage,
    "ImageAddText": ImageAddText,
    "imageSharpen": imageSharpen,
    "imageGaussianBlur": imageGaussianBlur,
    "imageRotate": imageRotate,
    "imageFlip": imageFlip,
    "imageFilter": imageFilter,
    "imageHug":imageHug,
    "imageRGB":imageRGB,
    "ImageEmpty":ImageEmpty
        
}

NODE_DISPLAY_NAME_MAPPINGS = {
    #—————————————————————— zsq_prompt ———————————————————
    "PortraitStyler": "Portrait Styler",
    "BatchPromptSelector": "Batch Prompt Selector",
    "stylesSelector": "Styles Selector",
    "BatchPromptJson": "Batch Prompt Json",
    #—————————————————————— zsq_llm ——————————————————————
    "LLMText": "LLM Text",
    "LLMImage": "LLM Image",
    #—————————————————————— zsq_utils ————————————————————
    "ZSQPixelLatent": "Pixel Latent",
    "ZSQRatioLatent": "Ratio Latent",
    "DoubleCLIPEncode":"Double CLIP Encode",
    "ZSQShowText": "Show Text",
    "ZSQShowINT": "Show INT",
    "FloatMathOperation": "Float Math Operation",
    "IntMathOperation": "Int Math Operation",
    "OptionString": "Option String",
    "ConnectionString": "Connection String",
    "IndexString": "Index String",
    "StringInput": "String Input",
    #—————————————————————— zsq_loader ———————————————————
    "zsqcheckpoint": "Simple Checkpoint",
    "controlnetStack": "Controlnet Stack",
    "controlnetStack_2": "Simple Controlnet Stack",
    "zsqcontrolnet": "Simple Controlnet",
    "loraStack": "Lora Stack",
    "loraStack_2": "Simple Lora Stack",
    "zsqsampler": "Simple Sampler",
    "checkpoint_sampler": "Checkpoint & Sampler",
    #—————————————————————— zsq_image ————————————————————
    "imageCount": "Image Count",
    "imageCrop": "Image Crop",
    "imageResize": "Image Resize",
    "imageSize": "Image Size",
    "imageScaleDown": "Image Scale Down",
    "imageScaleDownBy": "Image Scale Down By",
    "imageRatio": "Image Ratio",
    "imageSaveSimple": "Image Save Simple Jpg",
    "imageConcat": "Image Concat",
    "imageDetailTransfer": "Image Detail Transfer",
    "SaveJpgImage": "Image Save Jpg",
    "ImageAddText": "Image Add Text",
    "imageSharpen": "Image Sharpen",
    "imageGaussianBlur": "Image Gaussian Blur",
    "imageRotate": "Image Rotate",
    "imageFlip": "Image Flip",
    "imageFilter": "Image Filter",
    "imageHug": "Image Hug",
    "imageRGB": "Image RGB",
    "ImageEmpty": "Image Empty"
    }
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]








