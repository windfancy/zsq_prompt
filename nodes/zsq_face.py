from torchvision.utils import make_grid
import torch, os,sys
from pathlib import Path
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import normalize
import numpy as np
import cv2
from PIL import Image
from spandrel import ModelLoader, ImageModelDescriptor

from codeformer import CodeFormer  # 导入模型类

from transparent_background import Remover
from tqdm import tqdm

import folder_paths
import comfy
from comfy import model_management

from insightface.app.common import Face
from typing import List
from .libs.briarmbg import BriaRMBG
from .libs.utils import logger
from ..config import FACE_MODELS_PATH

from .libs.face import (
    load_face_model,
    get_model_names,
    get_facemodels,
    get_current_faces_model,
    analyze_faces,
    half_det_size,
    model_names,
    check_models,
    save_face_model,
    MyFaceRestore,
    StableDiffusionProcessingImg2Img,
    FaceSwapScript
    )
from .libs.image import DEVICE,tensor2pil,pil2tensor,resize_image_1024,batch_tensor_to_pil,batched_pil_to_tensor,tensor2img,img2tensor

try:
    from spandrel_extra_arches import EXTRA_REGISTRY
    from spandrel import MAIN_REGISTRY
    MAIN_REGISTRY.add(*EXTRA_REGISTRY)
    print("Successfully imported spandrel_extra_arches: support for non commercial upscale models.")
except:
    pass


dir_facerestore_models = os.path.join(folder_paths.models_dir, "facerestore_models")
dir_facedetection_models = os.path.join(folder_paths.models_dir, "facedetection")
os.makedirs(dir_facerestore_models, exist_ok=True)
os.makedirs(dir_facedetection_models, exist_ok=True)

def restore_face(facerestore_path=None, image=None, facedetection='retinaface_resnet50', codeformer_fidelity=0.5,device='cuda',face_helper=None):
    print(f"Loading restore_face fuction")
    # weights
    weights_str = os.path.join(dir_facerestore_models, facerestore_path)
    # 转换为 Path 对象
    weights = Path(weights_str) 
    facerestore_model = CodeFormer(weights=weights).model.to(device)
    model_dir = dir_facedetection_models
    
    if face_helper is None:
        #self.face_helper = FaceRestoreHelper(1, face_size=512, crop_ratio=(1, 1), det_model=facedetection, save_ext='png', use_parse=True, device=device)
        face_helper = MyFaceRestore(1, face_size=512, crop_ratio=(1, 1), det_model=facedetection,model_dir=model_dir,save_ext='png', use_parse=True, device=device)
    
    image_np = (255. * image.cpu().numpy()).astype(np.uint8)
    total_images = image_np.shape[0]
    out_images = np.ndarray(shape=image_np.shape)

    for i in range(total_images):          
        cur_image_np = image_np[i,:, :, ::-1]

        original_resolution = cur_image_np.shape[0:2]

        if facerestore_model is None or face_helper is None:
            return image

        face_helper.clean_all()
        face_helper.read_image(cur_image_np)
        face_helper.get_face_landmarks_5(only_center_face=True, resize=640, eye_dist_threshold=5)
        face_helper.align_warp_face()

        restored_face = None
        for idx, cropped_face in enumerate(face_helper.cropped_faces):
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

            try:
                with torch.no_grad():
                    output = facerestore_model(cropped_face_t, w=codeformer_fidelity)[0]
                    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                del output
                torch.cuda.empty_cache()
            except Exception as error:
                print(f'\tFailed inference for CodeFormer: {error}', file=sys.stderr)
                restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

            restored_face = restored_face.astype('uint8')
            face_helper.add_restored_face(restored_face)

        face_helper.get_inverse_affine(None)

        restored_img = face_helper.paste_faces_to_input_image()
        restored_img = restored_img[:, :, ::-1]

        if original_resolution != restored_img.shape[0:2]:
            print(f'\tResizing to original resolution {original_resolution}')
            restored_img = cv2.resize(restored_img, (0, 0), fx=original_resolution[1]/restored_img.shape[1], fy=original_resolution[0]/restored_img.shape[0], interpolation=cv2.INTER_LINEAR)

        face_helper.clean_all()

        out_images[i] = restored_img

    restored_img_np = np.array(out_images).astype(np.float32) / 255.0
    restored_img_tensor = torch.from_numpy(restored_img_np)
    return restored_img_tensor       

def upscale_image(upscale_model_name=None, image=None,device="cuda"):
    print(f"Loading upscale_image fuction")
    model_path = folder_paths.get_full_path_or_raise("upscale_models", upscale_model_name)
    sd = comfy.utils.load_torch_file(model_path, safe_load=True)
    print(f"Loading upscale model: {model_path}")
    if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
        sd = comfy.utils.state_dict_prefix_replace(sd, {"module.":""})
    upscale_model = ModelLoader().load_from_state_dict(sd).eval()

    if not isinstance(upscale_model, ImageModelDescriptor):
        raise Exception("Upscale model must be a single-image model.")

    memory_required = model_management.module_size(upscale_model.model)
    memory_required += (512 * 512 * 3) * image.element_size() * max(upscale_model.scale, 1.0) * 384.0 #The 384.0 is an estimate of how much some of these models take, TODO: make it more accurate
    memory_required += image.nelement() * image.element_size()
    model_management.free_memory(memory_required, device)

    upscale_model.to(device)
    in_img = image.movedim(-1,-3).to(device)

    tile = 512
    overlap = 32

    oom = True
    while oom:
        try:
            steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap)
            pbar = comfy.utils.ProgressBar(steps)
            s = comfy.utils.tiled_scale(in_img, lambda a: upscale_model(a), tile_x=tile, tile_y=tile, overlap=overlap, upscale_amount=upscale_model.scale, pbar=pbar)
            oom = False
        except model_management.OOM_EXCEPTION as e:
            tile //= 2
            if tile < 128:
                raise e

    upscale_model.to("cpu")
    s = torch.clamp(s.movedim(-3,-1), min=0, max=1.0)
    print(f"Upscaled image shape: {s.shape}")
    return s

class FaceRestoreCFWithModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                      "first_facerestore":("BOOLEAN", {"default": True, "label_off": "OFF", "label_on": "ON"}),
                      "facerestore":("BOOLEAN", {"default": True, "label_off": "OFF", "label_on": "ON"}),
                      "facerestore_path": (["codeformer.pth"],),
                      "image": ("IMAGE",),
                      "facedetection": (["retinaface_resnet50", "retinaface_mobile0.25"],),
                      "codeformer_fidelity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1, "step": 0.05, "display": "slider"}),
                      "upscale":("BOOLEAN", {"default": False, "label_off": "OFF", "label_on": "ON"}),
                      "upscale_model_name": (folder_paths.get_filename_list("upscale_models"), ),
                    }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "ZSQ/Face"

    def __init__(self):
        self.face_helper = None

    def execute(self, first_facerestore,facerestore,facerestore_path, image, facedetection, codeformer_fidelity,upscale,upscale_model_name):
        device = DEVICE
        out_img_tensor = image
        if first_facerestore:
            if facerestore:
                out_img_tensor = restore_face(facerestore_path=facerestore_path, image=out_img_tensor, facedetection=facedetection, codeformer_fidelity=codeformer_fidelity,device=device,face_helper =self.face_helper)  
            if upscale:    # image upscale
                out_img_tensor = upscale_image(upscale_model_name=upscale_model_name,image=out_img_tensor, device=device)   
        else:
            if upscale:    # image upscale
                out_img_tensor = upscale_image(upscale_model_name=upscale_model_name,image=out_img_tensor, device=device) 
            if facerestore:
                out_img_tensor = restore_face(facerestore_path=facerestore_path, image=out_img_tensor, facedetection=facedetection, codeformer_fidelity=codeformer_fidelity,device=device,face_helper =self.face_helper)
        
        return (out_img_tensor,)    


# from BRIA_RMBG_Zho
# from InspyrenetRembg
class zsq_rmbg:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "rmbg_model": (["RMBG-1.4", "InspyrenetRembg"],{"default": "RMBG-1.4"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", )
    RETURN_NAMES = ("image", "mask", )
    FUNCTION = "remove_background"
    CATEGORY = "ZSQ/Image"
  
    def remove_background(self, image,rmbg_model):
        if rmbg_model == "RMBG-1.4":
            new_ims, new_masks = self.RMBG_1_4(image)
        elif rmbg_model == "InspyrenetRembg":
            new_ims, new_masks = self.RMBG_inspyrenet(image=image)
        else:
            raise Exception("Unknown RMBG model")

        return new_ims, new_masks
    def RMBG_1_4(self, image):
        rmbgmodel = BriaRMBG()
        rmbg_model_path = os.path.join(folder_paths.models_dir, "rmbg")
        model_path = os.path.join(rmbg_model_path, "RMBG-1.4/model.pth")
        rmbgmodel.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"Loading rmbg model: {model_path}")
        rmbgmodel.to(DEVICE)
        rmbgmodel.eval() 

        processed_images = []
        processed_masks = []

        for img in image:
            orig_image = tensor2pil(img)
            w,h = orig_image.size
            img = resize_image_1024(orig_image)
            im_np = np.array(img)
            im_tensor = torch.tensor(im_np, dtype=torch.float32).permute(2,0,1)
            im_tensor = torch.unsqueeze(im_tensor,0)
            im_tensor = torch.divide(im_tensor,255.0)
            im_tensor = normalize(im_tensor,[0.5,0.5,0.5],[1.0,1.0,1.0])
            if torch.cuda.is_available():
                im_tensor=im_tensor.cuda()

            result=rmbgmodel(im_tensor)
            result = torch.squeeze(F.interpolate(result[0][0], size=(h,w), mode='bilinear') ,0)
            ma = torch.max(result)
            mi = torch.min(result)
            result = (result-mi)/(ma-mi)    
            im_array = (result*255).cpu().data.numpy().astype(np.uint8)
            pil_im = Image.fromarray(np.squeeze(im_array))
            new_im = Image.new("RGBA", pil_im.size, (0,0,0,0))
            new_im.paste(orig_image, mask=pil_im)

            new_im_tensor = pil2tensor(new_im)  # 将PIL图像转换为Tensor
            pil_im_tensor = pil2tensor(pil_im)  # 同上

            processed_images.append(new_im_tensor)
            processed_masks.append(pil_im_tensor)

        new_ims = torch.cat(processed_images, dim=0)
        new_masks = torch.cat(processed_masks, dim=0)

        return new_ims, new_masks
    
    def RMBG_inspyrenet(self, image=None, torchscript_jit="default"):
        print(f"Loading rmbg model:Inspyrenet")
        if (torchscript_jit == "default"):
            remover = Remover()
        else:
            remover = Remover(jit=True)
        img_list = []
        for img in tqdm(image, "Inspyrenet Rembg"):
            mid = remover.process(tensor2pil(img), type='rgba')
            out =  pil2tensor(mid)
            img_list.append(out)
        new_ims = torch.cat(img_list, dim=0)
        new_masks = new_ims[:, :, :, 3]
        return new_ims, new_masks

check_models()
BLENDED_FACE_MODEL = None
FACE_SIZE: int = 512
FACE_HELPER = None

class face_reactor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "reactor_enabled": ("BOOLEAN", {"default": True, "label_off": "OFF", "label_on": "ON"}),
                "input_image": ("IMAGE",),
                "swap_model": (list(model_names().keys()),),
                "face_model": (get_model_names(get_facemodels),),
                "detect_gender_input": (["no","female","male"], {"default": "no"}),
                "detect_gender_source": (["no","female","male"], {"default": "no"}),
                "input_faces_index": ("STRING", {"default": "0"}),
                "source_faces_index": ("STRING", {"default": "0"}),
            },
            "optional": {
                "source_image": ("IMAGE",),                
            },
            "hidden": {"faces_order": "FACES_ORDER"},
        }

    RETURN_TYPES = ("IMAGE","FACE_MODEL")
    FUNCTION = "execute"
    CATEGORY = "ZSQ/Face"

    def __init__(self):
        # self.face_helper = None
        self.faces_order = ["large-small", "large-small"]
        # self.face_size = FACE_SIZE
        self.face_boost_enabled = False
        self.restore = True
        self.boost_model = None
        self.interpolation = "Bicubic"
        self.boost_model_visibility = 1
        self.boost_cf_weight = 0.5

    def execute(self, **required):
        #加载参数
        reactor_enabled = required.get("reactor_enabled", None)
        input_image = required.get("input_image", None)
        self.swap_model = required.get("swap_model", None)        
        self.detect_gender_source = required.get("detect_gender_source", "no")
        self.detect_gender_input = required.get("detect_gender_input", "no")
        self.source_faces_index = required.get("source_faces_index", "0")
        self.input_faces_index = required.get("input_faces_index", "0")
        self.source_image = required.get("source_image", None)        
        faces_order = required.get("faces_order", None)
        face_model = required.get("face_model", "none")
        
        self.face_boost_enabled = False

        if faces_order is None:
            faces_order = self.faces_order

        if not reactor_enabled:
            return (input_image,face_model)
        elif self.source_image is None and face_model is None:
            logger.error("Please provide 'source_image' or `face_model`")
            return (input_image,face_model)
        result,face_model_to_provide = self.face_execute(input_image,face_model)  

        return (result,face_model_to_provide)
    
    def face_execute(self,input_image,face_model):
        if face_model == "none":
            self.face_model = None
        else:
            self.face_model = self.load_model(face_model)
        script = FaceSwapScript()
        pil_images = batch_tensor_to_pil(input_image)
        if self.source_image is not None:
            source = tensor2pil(self.source_image)
        else:
            source = None
        p = StableDiffusionProcessingImg2Img(pil_images)
        script.process(
            p=p,
            img=source,
            enable=True,
            source_faces_index=self.source_faces_index,
            faces_index=self.input_faces_index,
            model=self.swap_model,
            swap_in_source=True,
            swap_in_generated=True,
            gender_source=self.detect_gender_source,
            gender_target=self.detect_gender_input,
            face_model=self.face_model,
            faces_order=self.faces_order,
            # face boost:
            face_boost_enabled=self.face_boost_enabled,
            face_restore_model=self.boost_model,
            face_restore_visibility=self.boost_model_visibility,
            codeformer_weight=self.boost_cf_weight,
            interpolation=self.interpolation,
        )
        result = batched_pil_to_tensor(p.init_images)

        if self.face_model is None:
            current_face_model = get_current_faces_model()
            face_model_to_provide = current_face_model[0] if (current_face_model is not None and len(current_face_model) > 0) else face_model
        else:
            face_model_to_provide = self.face_model

        return result,face_model_to_provide  

    def load_model(self, face_model):
        if face_model != "none":
            face_model_path = os.path.join(FACE_MODELS_PATH, face_model)
            out = load_face_model(face_model_path)
        else:
            out = None
        return out

class face_reactor_plus(face_reactor):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "reactor_enabled": ("BOOLEAN", {"default": True, "label_off": "OFF", "label_on": "ON"}),
                "input_image": ("IMAGE",),
                "swap_model": (list(model_names().keys()),),
                "face_model": (get_model_names(get_facemodels),),

                "facerestore_enabled":("BOOLEAN", {"default": True, "label_off": "OFF", "label_on": "ON"}),
                "facerestore_path": (["codeformer.pth"],),
                "facedetection": (["retinaface_resnet50", "retinaface_mobile0.25"],),
                "codeformer_fidelity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1, "step": 0.05, "display": "slider"}),

                "upscale_enabled":("BOOLEAN", {"default": False, "label_off": "OFF", "label_on": "ON"}),
                "upscale_model_name": (folder_paths.get_filename_list("upscale_models"), ),

                "detect_gender_input": (["no","female","male"], {"default": "no"}),
                "detect_gender_source": (["no","female","male"], {"default": "no"}),
                "input_faces_index": ("STRING", {"default": "0"}),
                "source_faces_index": ("STRING", {"default": "0"}),
            },
            "optional": {
                "source_image": ("IMAGE",),
            },
            "hidden": {"faces_order": "FACES_ORDER"},
        }

    RETURN_TYPES = ("IMAGE","FACE_MODEL")
    FUNCTION = "execute"
    CATEGORY = "ZSQ/Face"

    def __init__(self):
        self.face_helper = None
        self.faces_order = ["large-small", "large-small"]
        # self.face_size = FACE_SIZE
        self.face_boost_enabled = False
        self.restore = True
        self.boost_model = None
        self.interpolation = "Bicubic"
        self.boost_model_visibility = 1
        self.boost_cf_weight = 0.5

    def execute(self, **required):
        #加载参数
        reactor_enabled = required.get("reactor_enabled", True)
        facerestore_enabled = required.get("facerestore_enabled", True)
        upscale_enabled = required.get("upscale_enabled", False)

        input_image = required.get("input_image", None)

        self.swap_model = required.get("swap_model", None)        
        self.detect_gender_source = required.get("detect_gender_source", "no")
        self.detect_gender_input = required.get("detect_gender_input", "no")
        self.source_faces_index = required.get("source_faces_index", "0")
        self.input_faces_index = required.get("input_faces_index", "0")
        self.source_image = required.get("source_image", None)        
        faces_order = required.get("faces_order", None)
        face_model = required.get("face_model", "none")
        
        self.facerestore_path = required.get("facerestore_path", None)
        self.facedetection = required.get("facedetection", None)
        self.codeformer_fidelity = required.get("codeformer_fidelity", None)
        self.upscale_model_name = required.get("upscale_model_name", None)
        
        self.face_boost_enabled = False

        if faces_order is None:
            faces_order = self.faces_order

        result = input_image
        face_model_to_provide = face_model
        device = DEVICE
        if reactor_enabled:
            result,face_model_to_provide = self.face_execute(result,face_model_to_provide)  
        if facerestore_enabled:
            result = restore_face(facerestore_path=self.facerestore_path, image=result, facedetection=self.facedetection, codeformer_fidelity=self.codeformer_fidelity,device=device,face_helper=self.face_helper)  
        if upscale_enabled:
            result = upscale_image(upscale_model_name=self.upscale_model_name,image=result, device=device)

        return (result,face_model_to_provide)

class BuildFaceModel:
    def __init__(self):
        self.output_dir = FACE_MODELS_PATH
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "save_mode": ("BOOLEAN", {"default": True, "label_off": "OFF", "label_on": "ON"}),
                "send_only": ("BOOLEAN", {"default": False, "label_off": "NO", "label_on": "YES"}),
                "face_model_name": ("STRING", {"default": "default"}),
                "compute_method": (["Mean", "Median", "Mode"], {"default": "Mean"}),
            },
            "optional": {
                "images": ("IMAGE",),
                "face_models": ("FACE_MODEL",),
            }
        }

    RETURN_TYPES = ("FACE_MODEL",)
    FUNCTION = "blend_faces"

    OUTPUT_NODE = True

    CATEGORY = "ZSQ/Face"

    def build_face_model(self, image: Image.Image, det_size=(640, 640)):
        if image is None:
            error_msg = "Please load an Image"
            logger.error(error_msg)
            return error_msg
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        face_model = analyze_faces(image, det_size)

        if len(face_model) == 0:
            print("")
            det_size_half = half_det_size(det_size)
            face_model = analyze_faces(image, det_size_half)
            if face_model is not None and len(face_model) > 0:
                print("...........................................................", end=" ")
        
        if face_model is not None and len(face_model) > 0:
            return face_model[0]
        else:
            no_face_msg = "No face found, please try another image"
            # logger.error(no_face_msg)
            return no_face_msg
    
    def blend_faces(self, save_mode, send_only, face_model_name, compute_method, images=None, face_models=None):
        global BLENDED_FACE_MODEL
        blended_face: Face = BLENDED_FACE_MODEL

        if send_only and blended_face is None:
            send_only = False

        if (images is not None or face_models is not None) and not send_only:

            faces = []
            embeddings = []

            if images is not None:
                images_list: List[Image.Image] = batch_tensor_to_pil(images)

                n = len(images_list)

                for i,image in enumerate(images_list):
                    logger.status(f"Building Face Model {i+1} of {n}...")
                    face = self.build_face_model(image)
                    if isinstance(face, str):
                        logger.error(f"No faces found in image {i+1}, skipping")
                        continue
                    else:
                        print(f"{int(((i+1)/n)*100)}%")
                    faces.append(face)
                    embeddings.append(face.embedding)
            
            elif face_models is not None:

                n = len(face_models)

                for i,face_model in enumerate(face_models):
                    logger.status(f"Extracting Face Model {i+1} of {n}...")
                    face = face_model
                    if isinstance(face, str):
                        logger.error(f"No faces found for face_model {i+1}, skipping")
                        continue
                    else:
                        print(f"{int(((i+1)/n)*100)}%")
                    faces.append(face)
                    embeddings.append(face.embedding)

            if len(faces) > 0:
                # compute_method_name = "Mean" if compute_method == 0 else "Median" if compute_method == 1 else "Mode"
                logger.status(f"Blending with Compute Method '{compute_method}'...")
                blended_embedding = np.mean(embeddings, axis=0) if compute_method == "Mean" else np.median(embeddings, axis=0) if compute_method == "Median" else stats.mode(embeddings, axis=0)[0].astype(np.float32)
                blended_face = Face(
                    bbox=faces[0].bbox,
                    kps=faces[0].kps,
                    det_score=faces[0].det_score,
                    landmark_3d_68=faces[0].landmark_3d_68,
                    pose=faces[0].pose,
                    landmark_2d_106=faces[0].landmark_2d_106,
                    embedding=blended_embedding,
                    gender=faces[0].gender,
                    age=faces[0].age
                )
                if blended_face is not None:
                    BLENDED_FACE_MODEL = blended_face
                    if save_mode:
                        face_model_path = os.path.join(FACE_MODELS_PATH, face_model_name + ".safetensors")
                        save_face_model(blended_face,face_model_path)
                        # done_msg = f"Face model has been saved to '{face_model_path}'"
                        # logger.status(done_msg)
                    logger.status("--Done!--")
                    # return (blended_face,)
                else:
                    no_face_msg = "Something went wrong, please try another set of images"
                    logger.error(no_face_msg)
                    # return (blended_face,)
            # logger.status("--Done!--")
        if images is None and face_models is None:
            logger.error("Please provide `images` or `face_models`")
        return (blended_face,)


class SaveFaceModel:
    def __init__(self):
        self.output_dir = FACE_MODELS_PATH

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "save_mode": ("BOOLEAN", {"default": True, "label_off": "OFF", "label_on": "ON"}),
                "face_model_name": ("STRING", {"default": "default"}),
                "select_face_index": ("INT", {"default": 0, "min": 0}),
            },
            "optional": {
                "image": ("IMAGE",),
                "face_model": ("FACE_MODEL",),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_model"

    OUTPUT_NODE = True

    CATEGORY = "ZSQ/Face"

    def save_model(self, save_mode, face_model_name, select_face_index, image=None, face_model=None, det_size=(640, 640)):
        if save_mode and image is not None:
            source = tensor2pil(image)
            source = cv2.cvtColor(np.array(source), cv2.COLOR_RGB2BGR)
            logger.status("Building Face Model...")
            face_model_raw = analyze_faces(source, det_size)
            if len(face_model_raw) == 0:
                det_size_half = half_det_size(det_size)
                face_model_raw = analyze_faces(source, det_size_half)
            try:
                face_model = face_model_raw[select_face_index]
            except:
                logger.error("No face(s) found")
                return face_model_name
            logger.status("--Done!--")
        if save_mode and (face_model != "none" or face_model is not None):
            face_model_path = os.path.join(self.output_dir, face_model_name + ".safetensors")
            save_face_model(face_model,face_model_path)
        if image is None and face_model is None:
            logger.error("Please provide `face_model` or `image`")
        return face_model_name

class MakeFaceModelBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "face_model1": ("FACE_MODEL",), 
            },
            "optional": {
                "face_model2": ("FACE_MODEL",),
                "face_model3": ("FACE_MODEL",),
                "face_model4": ("FACE_MODEL",),
                "face_model5": ("FACE_MODEL",),
                "face_model6": ("FACE_MODEL",),
                "face_model7": ("FACE_MODEL",),
                "face_model8": ("FACE_MODEL",),
                "face_model9": ("FACE_MODEL",),
                "face_model10": ("FACE_MODEL",),
            },
        }

    RETURN_TYPES = ("FACE_MODEL",)
    RETURN_NAMES = ("FACE_MODELS",)
    FUNCTION = "execute"

    CATEGORY = "ZSQ/Face"

    def execute(self, **kwargs):
        if len(kwargs) > 0:
            face_models = [value for value in kwargs.values()]
            return (face_models,)
        else:
            logger.error("Please provide at least 1 `face_model`")
            return (None,)