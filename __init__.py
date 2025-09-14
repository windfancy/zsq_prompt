from .nodes.zsq_prompt import (
    PortraitStyler, 
    stylesPromptSelector,
    BatchPromptSelector,
    BatchPromptJson
    )
from .nodes.person_mask_ultra_v2 import PersonMaskUltraV2
from .nodes.zsq_llm import LLMText,LLMImage
from .nodes.zsq_face import (
    FaceRestoreCFWithModel,
    zsq_rmbg,face_reactor,
    face_reactor_plus,
    SaveFaceModel,
    BuildFaceModel,
    MakeFaceModelBatch
    )
from .nodes.zsq_loader import (
    checkpoint_sampler,
    checkpoint_sampler_tripleclip,
    checkpoint_sampler_dualclip,
    checkpoint_sampler_unet,
    zsqcheckpoint,
    zsqcontrolnetStack,
    zsqcontrolnetstack_2,
    zsqcontrolnet,
    zsqloraStack,
    zsqloraStack_2,
    zsqsampler
    )   
from .nodes.zsq_utils import (
    ZSQ_PixelLatent,
    ZSQ_RatioLatent,
    DoubleCLIPEncode,
    ZsqLatent,
    MaskToImage,
    ShowText,
    ShowINT,
    FloatMathOperation,
    IntMathOperation,
    OptionString,
    ConnectionString,
    StringInput,
    IndexString
    )
from .nodes.zsq_image import (
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
    imageRGB,
    ImageEmpty,
    LoadImagesFromFolder,
    ImageColorAdapter
    )
from .nodes.segformer_ultra import Segformer_B2_Clothes,SegformerUltraV2,SegformerClothesPipelineLoader,SegformerFashionPipelineLoader


directory = ".\web"

WEB_DIRECTORY = directory


NODE_CLASS_MAPPINGS = {
    #—————————————————————— zsq_prompt ———————————————————
    "PortraitStyler": PortraitStyler,
    "stylesSelector": stylesPromptSelector,
    "BatchPromptSelector": BatchPromptSelector,
    "BatchPromptJson": BatchPromptJson,
    #—————————————————————— PersonMaskUltraV2 ——————————————————————
    "PersonMaskUltraV2":PersonMaskUltraV2,
    #—————————————————————— zsq_llm ——————————————————————
    "LLMText": LLMText,
    "LLMImage": LLMImage,
    "FaceRestoreCFWithModel": FaceRestoreCFWithModel,
    "zsq_rmbg": zsq_rmbg,
    "face_reactor": face_reactor,
    "face_reactor_plus": face_reactor_plus,
    #—————————————————————— zsq_utils ————————————————————
    "ZsqLatent": ZsqLatent,
    "MaskToImage": MaskToImage,    
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
    "checkpoint_sampler_tripleclip": checkpoint_sampler_tripleclip,
    "checkpoint_sampler_dualclip": checkpoint_sampler_dualclip,
    "checkpoint_sampler_unet":checkpoint_sampler_unet,
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
    "imageRGB":imageRGB,
    "ImageEmpty":ImageEmpty,
    "LoadImagesFromFolder":LoadImagesFromFolder,
    "ImageColorAdapter":ImageColorAdapter,
    "ReActorSaveFaceModel": SaveFaceModel,
    "ReActorBuildFaceModel": BuildFaceModel,
    "ReActorMakeFaceModelBatch": MakeFaceModelBatch,
    "LayerMask: SegformerB2ClothesUltra": Segformer_B2_Clothes,
    "LayerMask: SegformerUltraV2": SegformerUltraV2,
    "LayerMask: SegformerClothesPipelineLoader": SegformerClothesPipelineLoader,
    "LayerMask: SegformerFashionPipelineLoader": SegformerFashionPipelineLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    #—————————————————————— zsq_prompt ———————————————————
    "PortraitStyler": "Portrait Styler",
    "BatchPromptSelector": "Batch Prompt Selector",
    "stylesSelector": "Styles Selector",
    "BatchPromptJson": "Batch Prompt Json",
    #—————————————————————— PersonMaskUltraV2 ——————————————————————
    "PersonMaskUltraV2": "Person Mask Ultra V2",
    #—————————————————————— zsq_llm ——————————————————————
    "LLMText": "LLM Text",
    "LLMImage": "LLM Image",
    "FaceRestoreCFWithModel": "Face Restore",
    "zsq_rmbg": "Remove Background",
    "face_reactor": "Face Reactor",
    "face_reactor_plus": "Reactor Plus",
    #—————————————————————— zsq_utils ————————————————————
    "ZsqLatent": "Zsq Latent",
    "MaskToImage": "Mask To Image",    
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
    "checkpoint_sampler_tripleclip": "Checkpoint & Sampler for TripleCLIP",
    "checkpoint_sampler_dualclip": "Checkpoint & Sampler for DualCLIP",
    "checkpoint_sampler_unet": "Checkpoint & Sampler for UNet",
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
    "imageRGB": "Image RGB",
    "ImageEmpty": "Image Empty",
    "LoadImagesFromFolder": "Load Images From Folder",
    "ImageColorAdapter": "Image Color Adapter",
    "ReActorSaveFaceModel": "Save Face Model ZSQ/ReActor",
    "ReActorBuildFaceModel": "Build Blended Face Model ZSQ/ReActor",
    "ReActorMakeFaceModelBatch": "Make Face Model Batch ZSQ/ReActor",
    "LayerMask: SegformerB2ClothesUltra": "LayerMask: Segformer B2 Clothes Ultra",
    "LayerMask: SegformerUltraV2": "LayerMask: Segformer Ultra V2",
    "LayerMask: SegformerClothesPipelineLoader": "LayerMask: Segformer Clothes Pipeline",
    "LayerMask: SegformerFashionPipelineLoader": "LayerMask: Segformer Fashion Pipeline"
    }

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]








