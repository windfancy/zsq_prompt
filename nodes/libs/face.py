import glob,os,shutil
import torch
try:
    import torch.cuda as cuda
    
except:
    cuda = None    
import cv2
import numpy as np
from PIL import Image
from typing import List, Union
import insightface
from insightface.app.common import Face
from safetensors.torch import save_file,safe_open
import folder_paths
from ...config import REACTOR_MODELS_PATH,FACE_MODELS_PATH,MODELS_DIR
from .utils import State,logger,move_path
from .image import get_image_md5hash
import folder_paths
import comfy.model_management as model_management
from codeformer.facelib.utils.face_restoration_helper import FaceRestoreHelper
from codeformer.facelib.detection.retinaface.retinaface import RetinaFace
from codeformer.facelib.parsing.bisenet import BiSeNet
from codeformer.facelib.parsing.parsenet import ParseNet
from copy import deepcopy
import warnings

np.warnings = warnings
np.warnings.filterwarnings('ignore')
if cuda is not None:
    if cuda.is_available():
        providers = ["CUDAExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
else:
    providers = ["CPUExecutionProvider"]

models_path_old = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
insightface_path_old = os.path.join(models_path_old, "insightface")
insightface_models_path_old = os.path.join(insightface_path_old, "models")

models_path = folder_paths.models_dir
insightface_path = os.path.join(models_path, "insightface")
insightface_models_path = os.path.join(insightface_path, "models")

if os.path.exists(models_path_old):
    move_path(insightface_models_path_old, insightface_models_path)
    move_path(insightface_path_old, insightface_path)
    move_path(models_path_old, models_path)
if os.path.exists(insightface_path) and os.path.exists(insightface_path_old):
    shutil.rmtree(insightface_path_old)
    shutil.rmtree(models_path_old)

FS_MODEL = None
CURRENT_FS_MODEL_PATH = None

ANALYSIS_MODELS = {
    "640": None,
    "320": None,
}

SOURCE_FACES = None
SOURCE_IMAGE_HASH = None
TARGET_FACES = None
TARGET_IMAGE_HASH = None
TARGET_FACES_LIST = []
TARGET_IMAGE_LIST_HASH = []

def get_facemodels():
    models_path = os.path.join(FACE_MODELS_PATH, "*")
    models = glob.glob(models_path)
    models = [x for x in models if x.endswith(".safetensors")]
    return models


def get_model_names(get_models):
    models = get_models()
    names = []
    for x in models:
        names.append(os.path.basename(x))
    names.sort(key=str.lower)
    names.insert(0, "none")
    return names

def model_names():
    models = get_models()
    return {os.path.basename(x): x for x in models}

def get_restorers():
    models_path = os.path.join(MODELS_DIR, "facerestore_models/*")
    models = glob.glob(models_path)
    models = [x for x in models if (x.endswith(".pth") or x.endswith(".onnx"))]
    if len(models) == 0:
       logger.error("No face restoration models found in %s" % models_path)
       models = None 
    return models
# author: Trung0246 --->
def add_folder_path_and_extensions(folder_name, full_folder_paths, extensions):
    # Iterate over the list of full folder paths
    for full_folder_path in full_folder_paths:
        # Use the provided function to add each model folder path
        folder_paths.add_model_folder_path(folder_name, full_folder_path)

    # Now handle the extensions. If the folder name already exists, update the extensions
    if folder_name in folder_paths.folder_names_and_paths:
        # Unpack the current paths and extensions
        current_paths, current_extensions = folder_paths.folder_names_and_paths[folder_name]
        # Update the extensions set with the new extensions
        updated_extensions = current_extensions | extensions
        # Reassign the updated tuple back to the dictionary
        folder_paths.folder_names_and_paths[folder_name] = (current_paths, updated_extensions)
    else:
        # If the folder name was not present, add_model_folder_path would have added it with the last path
        # Now we just need to update the set of extensions as it would be an empty set
        # Also ensure that all paths are included (since add_model_folder_path adds only one path at a time)
        folder_paths.folder_names_and_paths[folder_name] = (full_folder_paths, extensions)

def get_models():
    models_path = os.path.join(folder_paths.models_dir,"insightface/*")
    models = glob.glob(models_path)
    models = [x for x in models if x.endswith(".onnx") or x.endswith(".pth")]
    return models

def save_face_model(face: Face, filename: str) -> None:
    try:
        tensors = {
            "bbox": torch.tensor(face["bbox"]),
            "kps": torch.tensor(face["kps"]),
            "det_score": torch.tensor(face["det_score"]),
            "landmark_3d_68": torch.tensor(face["landmark_3d_68"]),
            "pose": torch.tensor(face["pose"]),
            "landmark_2d_106": torch.tensor(face["landmark_2d_106"]),
            "embedding": torch.tensor(face["embedding"]),
            "gender": torch.tensor(face["gender"]),
            "age": torch.tensor(face["age"]),
        }
        save_file(tensors, filename)
        logger.info(f"Face model has been saved to '{filename}'")
    except Exception as e:
        logger.error(f"Error: {e}")

def load_face_model(filename: str):
    face = {}
    with safe_open(filename, framework="pt") as f:
        for k in f.keys():
            face[k] = f.get_tensor(k).numpy()
    return Face(face)

def check_models():
    if not os.path.exists(REACTOR_MODELS_PATH):
        os.makedirs(REACTOR_MODELS_PATH)
        if not os.path.exists(FACE_MODELS_PATH):
            os.makedirs(FACE_MODELS_PATH)

    dir_facerestore_models = os.path.join(MODELS_DIR, "facerestore_models")
    os.makedirs(dir_facerestore_models, exist_ok=True)
    folder_paths.folder_names_and_paths["facerestore_models"] = ([dir_facerestore_models], folder_paths.supported_pt_extensions)

    if "ultralytics" not in folder_paths.folder_names_and_paths:
        add_folder_path_and_extensions("ultralytics_bbox", [os.path.join(MODELS_DIR, "ultralytics", "bbox")], folder_paths.supported_pt_extensions)
        add_folder_path_and_extensions("ultralytics_segm", [os.path.join(MODELS_DIR, "ultralytics", "segm")], folder_paths.supported_pt_extensions)
        add_folder_path_and_extensions("ultralytics", [os.path.join(MODELS_DIR, "ultralytics")], folder_paths.supported_pt_extensions)
    if "sams" not in folder_paths.folder_names_and_paths:
        add_folder_path_and_extensions("sams", [os.path.join(MODELS_DIR, "sams")], folder_paths.supported_pt_extensions)

class StableDiffusionProcessing:
    def __init__(self, init_imgs):
        self.init_images = init_imgs
        self.width = init_imgs[0].width
        self.height = init_imgs[0].height
        self.extra_generation_params = {}


class StableDiffusionProcessingImg2Img(StableDiffusionProcessing):
    def __init__(self, init_img):
        super().__init__(init_img)


def get_models():
    models_path = os.path.join(folder_paths.models_dir,"insightface/*")
    models = glob.glob(models_path)
    models = [x for x in models if x.endswith(".onnx") or x.endswith(".pth")]
    return models


class FaceSwapScript():

    def process(
        self,
        p: StableDiffusionProcessing,
        img,
        enable,
        source_faces_index,
        faces_index,
        model,
        swap_in_source,
        swap_in_generated,
        gender_source,
        gender_target,
        face_model,
        faces_order,
        face_boost_enabled,
        face_restore_model,
        face_restore_visibility,
        codeformer_weight,
        interpolation,
    ):
        self.enable = enable
        if self.enable:

            self.source = img    
            self.swap_in_generated = swap_in_generated
            self.gender_source = gender_source
            self.gender_target = gender_target
            self.model = model
            self.face_model = face_model
            self.faces_order = faces_order
            self.face_boost_enabled = face_boost_enabled
            self.face_restore_model = face_restore_model
            self.face_restore_visibility = face_restore_visibility
            self.codeformer_weight = codeformer_weight
            self.interpolation = interpolation
            self.source_faces_index = [
                int(x) for x in source_faces_index.strip(",").split(",") if x.isnumeric()
            ]
            self.faces_index = [
                int(x) for x in faces_index.strip(",").split(",") if x.isnumeric()
            ]
            if len(self.source_faces_index) == 0:
                self.source_faces_index = [0]
            if len(self.faces_index) == 0:
                self.faces_index = [0]
            
            if self.gender_source is None or self.gender_source == "no":
                self.gender_source = 0
            elif self.gender_source  == "female":
                self.gender_source = 1
            elif self.gender_source  == "male":
                self.gender_source = 2
            
            if self.gender_target is None or self.gender_target == "no":
                self.gender_target = 0
            elif self.gender_target  == "female":
                self.gender_target = 1
            elif self.gender_target  == "male":
                self.gender_target = 2

            # if self.source is not None:
            if isinstance(p, StableDiffusionProcessingImg2Img) and swap_in_source:
                logger.status(f"Working: source face index %s, target face index %s", self.source_faces_index, self.faces_index)

                if len(p.init_images) == 1:

                    result = swap_face(
                        self.source,
                        p.init_images[0],
                        source_faces_index=self.source_faces_index,
                        faces_index=self.faces_index,
                        model=self.model,
                        gender_source=self.gender_source,
                        gender_target=self.gender_target,
                        face_model=self.face_model,
                        faces_order=self.faces_order,
                        face_boost_enabled=self.face_boost_enabled,
                        face_restore_model=self.face_restore_model,
                        face_restore_visibility=self.face_restore_visibility,
                        codeformer_weight=self.codeformer_weight,
                        interpolation=self.interpolation,
                    )
                    p.init_images[0] = result

                elif len(p.init_images) > 1:
                    result = swap_face_many(
                        self.source,
                        p.init_images,
                        source_faces_index=self.source_faces_index,
                        faces_index=self.faces_index,
                        model=self.model,
                        gender_source=self.gender_source,
                        gender_target=self.gender_target,
                        face_model=self.face_model,
                        faces_order=self.faces_order,
                        face_boost_enabled=self.face_boost_enabled,
                        face_restore_model=self.face_restore_model,
                        face_restore_visibility=self.face_restore_visibility,
                        codeformer_weight=self.codeformer_weight,
                        interpolation=self.interpolation,
                    )
                    p.init_images = result

                logger.status("--Done!--")
            # else:
            #     logger.error(f"Please provide a source face")

    def postprocess_batch(self, p, *args, **kwargs):
        if self.enable:
            images = kwargs["images"]


def get_current_faces_model():
    global SOURCE_FACES
    return SOURCE_FACES

def getAnalysisModel(det_size = (640, 640)):
    global ANALYSIS_MODELS
    ANALYSIS_MODEL = ANALYSIS_MODELS[str(det_size[0])]
    if ANALYSIS_MODEL is None:
        ANALYSIS_MODEL = insightface.app.FaceAnalysis(
            name="buffalo_l", providers=providers, root=insightface_path
        )
    ANALYSIS_MODEL.prepare(ctx_id=0, det_size=det_size)
    ANALYSIS_MODELS[str(det_size[0])] = ANALYSIS_MODEL
    return ANALYSIS_MODEL

def getFaceSwapModel(model_path: str):
    global FS_MODEL
    global CURRENT_FS_MODEL_PATH
    if CURRENT_FS_MODEL_PATH is None or CURRENT_FS_MODEL_PATH != model_path:
        CURRENT_FS_MODEL_PATH = model_path
        FS_MODEL = insightface.model_zoo.get_model(model_path, providers=providers)

    return FS_MODEL


def sort_by_order(face, order: str):
    if order == "left-right":
        return sorted(face, key=lambda x: x.bbox[0])
    if order == "right-left":
        return sorted(face, key=lambda x: x.bbox[0], reverse = True)
    if order == "top-bottom":
        return sorted(face, key=lambda x: x.bbox[1])
    if order == "bottom-top":
        return sorted(face, key=lambda x: x.bbox[1], reverse = True)
    if order == "small-large":
        return sorted(face, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    # if order == "large-small":
    #     return sorted(face, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse = True)
    # by default "large-small":
    return sorted(face, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse = True)

def get_face_gender(
        face,
        face_index,
        gender_condition,
        operated: str,
        order: str,
):
    gender = [
        x.sex
        for x in face
    ]
    gender.reverse()
    # If index is outside of bounds, return None, avoid exception
    if face_index >= len(gender):
        logger.status("Requested face index (%s) is out of bounds (max available index is %s)", face_index, len(gender))
        return None, 0
    face_gender = gender[face_index]
    logger.status("%s Face %s: Detected Gender -%s-", operated, face_index, face_gender)
    if (gender_condition == 1 and face_gender == "F") or (gender_condition == 2 and face_gender == "M"):
        logger.status("OK - Detected Gender matches Condition")
        try:
            faces_sorted = sort_by_order(face, order)
            return faces_sorted[face_index], 0
            # return sorted(face, key=lambda x: x.bbox[0])[face_index], 0
        except IndexError:
            return None, 0
    else:
        logger.status("WRONG - Detected Gender doesn't match Condition")
        faces_sorted = sort_by_order(face, order)
        return faces_sorted[face_index], 1
        # return sorted(face, key=lambda x: x.bbox[0])[face_index], 1

def half_det_size(det_size):
    logger.status("Trying to halve 'det_size' parameter")
    return (det_size[0] // 2, det_size[1] // 2)

def analyze_faces(img_data: np.ndarray, det_size=(640, 640)):
    face_analyser = getAnalysisModel(det_size)
    faces = face_analyser.get(img_data)

    # Try halving det_size if no faces are found
    if len(faces) == 0 and det_size[0] > 320 and det_size[1] > 320:
        det_size_half = half_det_size(det_size)
        return analyze_faces(img_data, det_size_half)

    return faces

def get_face_single(img_data: np.ndarray, face, face_index=0, det_size=(640, 640), gender_source=0, gender_target=0, order="large-small"):

    buffalo_path = os.path.join(insightface_models_path, "buffalo_l.zip")
    if os.path.exists(buffalo_path):
        os.remove(buffalo_path)

    if gender_source != 0:
        if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
            det_size_half = half_det_size(det_size)
            return get_face_single(img_data, analyze_faces(img_data, det_size_half), face_index, det_size_half, gender_source, gender_target, order)
        return get_face_gender(face,face_index,gender_source,"Source", order)

    if gender_target != 0:
        if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
            det_size_half = half_det_size(det_size)
            return get_face_single(img_data, analyze_faces(img_data, det_size_half), face_index, det_size_half, gender_source, gender_target, order)
        return get_face_gender(face,face_index,gender_target,"Target", order)
    
    if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
        det_size_half = half_det_size(det_size)
        return get_face_single(img_data, analyze_faces(img_data, det_size_half), face_index, det_size_half, gender_source, gender_target, order)

    try:
        faces_sorted = sort_by_order(face, order)
        return faces_sorted[face_index], 0
        # return sorted(face, key=lambda x: x.bbox[0])[face_index], 0
    except IndexError:
        return None, 0


def swap_face(
    source_img: Union[Image.Image, None],
    target_img: Image.Image,
    model: Union[str, None] = None,
    source_faces_index: List[int] = [0],
    faces_index: List[int] = [0],
    gender_source: int = 0,
    gender_target: int = 0,
    face_model: Union[Face, None] = None,
    faces_order: List = ["large-small", "large-small"],
    face_boost_enabled: bool = False,
    face_restore_model = None,
    face_restore_visibility: int = 1,
    codeformer_weight: float = 0.5,
    interpolation: str = "Bicubic",
):
    global SOURCE_FACES, SOURCE_IMAGE_HASH, TARGET_FACES, TARGET_IMAGE_HASH
    result_image = target_img

    if model is not None:

        if isinstance(source_img, str):  # source_img is a base64 string
            import base64, io
            if 'base64,' in source_img:  # check if the base64 string has a data URL scheme
                # split the base64 string to get the actual base64 encoded image data
                base64_data = source_img.split('base64,')[-1]
                # decode base64 string to bytes
                img_bytes = base64.b64decode(base64_data)
            else:
                # if no data URL scheme, just decode
                img_bytes = base64.b64decode(source_img)
            
            source_img = Image.open(io.BytesIO(img_bytes))
            
        target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)

        if source_img is not None:

            source_img = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)

            source_image_md5hash = get_image_md5hash(source_img)

            if SOURCE_IMAGE_HASH is None:
                SOURCE_IMAGE_HASH = source_image_md5hash
                source_image_same = False
            else:
                source_image_same = True if SOURCE_IMAGE_HASH == source_image_md5hash else False
                if not source_image_same:
                    SOURCE_IMAGE_HASH = source_image_md5hash

            logger.info("Source Image MD5 Hash = %s", SOURCE_IMAGE_HASH)
            logger.info("Source Image the Same? %s", source_image_same)

            if SOURCE_FACES is None or not source_image_same:
                logger.status("Analyzing Source Image...")
                source_faces = analyze_faces(source_img)
                SOURCE_FACES = source_faces
            elif source_image_same:
                logger.status("Using Hashed Source Face(s) Model...")
                source_faces = SOURCE_FACES

        elif face_model is not None:

            source_faces_index = [0]
            logger.status("Using Loaded Source Face Model...")
            source_face_model = [face_model]
            source_faces = source_face_model

        else:
            logger.error("Cannot detect any Source")

        if source_faces is not None:

            target_image_md5hash = get_image_md5hash(target_img)

            if TARGET_IMAGE_HASH is None:
                TARGET_IMAGE_HASH = target_image_md5hash
                target_image_same = False
            else:
                target_image_same = True if TARGET_IMAGE_HASH == target_image_md5hash else False
                if not target_image_same:
                    TARGET_IMAGE_HASH = target_image_md5hash

            logger.info("Target Image MD5 Hash = %s", TARGET_IMAGE_HASH)
            logger.info("Target Image the Same? %s", target_image_same)
            
            if TARGET_FACES is None or not target_image_same:
                logger.status("Analyzing Target Image...")
                target_faces = analyze_faces(target_img)
                TARGET_FACES = target_faces
            elif target_image_same:
                logger.status("Using Hashed Target Face(s) Model...")
                target_faces = TARGET_FACES

            # No use in trying to swap faces if no faces are found, enhancement
            if len(target_faces) == 0:
                logger.status("Cannot detect any Target, skipping swapping...")
                return result_image

            if source_img is not None:
                # separated management of wrong_gender between source and target, enhancement
                source_face, src_wrong_gender = get_face_single(source_img, source_faces, face_index=source_faces_index[0], gender_source=gender_source, order=faces_order[1])
            else:
                # source_face = sorted(source_faces, key=lambda x: x.bbox[0])[source_faces_index[0]]
                source_face = sorted(source_faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse = True)[source_faces_index[0]]
                src_wrong_gender = 0

            if len(source_faces_index) != 0 and len(source_faces_index) != 1 and len(source_faces_index) != len(faces_index):
                logger.status(f'Source Faces must have no entries (default=0), one entry, or same number of entries as target faces.')
            elif source_face is not None:
                result = target_img
                model_path = model_path = os.path.join(insightface_path, model)
                face_swapper = getFaceSwapModel(model_path)

                source_face_idx = 0

                for face_num in faces_index:
                    # No use in trying to swap faces if no further faces are found, enhancement
                    if face_num >= len(target_faces):
                        logger.status("Checked all existing target faces, skipping swapping...")
                        break

                    if len(source_faces_index) > 1 and source_face_idx > 0:
                        source_face, src_wrong_gender = get_face_single(source_img, source_faces, face_index=source_faces_index[source_face_idx], gender_source=gender_source, order=faces_order[1])
                    source_face_idx += 1

                    if source_face is not None and src_wrong_gender == 0:
                        target_face, wrong_gender = get_face_single(target_img, target_faces, face_index=face_num, gender_target=gender_target, order=faces_order[0])
                        if target_face is not None and wrong_gender == 0:
                            logger.status(f"Swapping...")
                            if face_boost_enabled:
                                logger.status(f"Face Boost is enabled")
                                bgr_fake, M = face_swapper.get(result, target_face, source_face, paste_back=False)
                                #bgr_fake, scale = restorer.get_restored_face(bgr_fake, face_restore_model, face_restore_visibility, codeformer_weight, interpolation)
                                #M *= scale
                                result = in_swap(target_img, bgr_fake, M)
                            else:
                                # logger.status(f"Swapping as-is")
                                result = face_swapper.get(result, target_face, source_face)
                        elif wrong_gender == 1:
                            wrong_gender = 0
                            # Keep searching for other faces if wrong gender is detected, enhancement
                            #if source_face_idx == len(source_faces_index):
                            #    result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                            #    return result_image
                            logger.status("Wrong target gender detected")
                            continue
                        else:
                            logger.status(f"No target face found for {face_num}")
                    elif src_wrong_gender == 1:
                        src_wrong_gender = 0
                        # Keep searching for other faces if wrong gender is detected, enhancement
                        #if source_face_idx == len(source_faces_index):
                        #    result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                        #    return result_image
                        logger.status("Wrong source gender detected")
                        continue
                    else:
                        logger.status(f"No source face found for face number {source_face_idx}.")

                result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

            else:
                logger.status("No source face(s) in the provided Index")
        else:
            logger.status("No source face(s) found")
    return result_image

def swap_face_many(
    source_img: Union[Image.Image, None],
    target_imgs: List[Image.Image],
    model: Union[str, None] = None,
    source_faces_index: List[int] = [0],
    faces_index: List[int] = [0],
    gender_source: int = 0,
    gender_target: int = 0,
    face_model: Union[Face, None] = None,
    faces_order: List = ["large-small", "large-small"],
    face_boost_enabled: bool = False,
    face_restore_model = None,
    face_restore_visibility: int = 1,
    codeformer_weight: float = 0.5,
    interpolation: str = "Bicubic",
):
    global SOURCE_FACES, SOURCE_IMAGE_HASH, TARGET_FACES, TARGET_IMAGE_HASH, TARGET_FACES_LIST, TARGET_IMAGE_LIST_HASH
    result_images = target_imgs

    if model is not None:

        if isinstance(source_img, str):  # source_img is a base64 string
            import base64, io
            if 'base64,' in source_img:  # check if the base64 string has a data URL scheme
                # split the base64 string to get the actual base64 encoded image data
                base64_data = source_img.split('base64,')[-1]
                # decode base64 string to bytes
                img_bytes = base64.b64decode(base64_data)
            else:
                # if no data URL scheme, just decode
                img_bytes = base64.b64decode(source_img)
            
            source_img = Image.open(io.BytesIO(img_bytes))
            
        target_imgs = [cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR) for target_img in target_imgs]

        if source_img is not None:

            source_img = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)

            source_image_md5hash = get_image_md5hash(source_img)

            if SOURCE_IMAGE_HASH is None:
                SOURCE_IMAGE_HASH = source_image_md5hash
                source_image_same = False
            else:
                source_image_same = True if SOURCE_IMAGE_HASH == source_image_md5hash else False
                if not source_image_same:
                    SOURCE_IMAGE_HASH = source_image_md5hash

            logger.info("Source Image MD5 Hash = %s", SOURCE_IMAGE_HASH)
            logger.info("Source Image the Same? %s", source_image_same)

            if SOURCE_FACES is None or not source_image_same:
                logger.status("Analyzing Source Image...")
                source_faces = analyze_faces(source_img)
                SOURCE_FACES = source_faces
            elif source_image_same:
                logger.status("Using Hashed Source Face(s) Model...")
                source_faces = SOURCE_FACES

        elif face_model is not None:

            source_faces_index = [0]
            logger.status("Using Loaded Source Face Model...")
            source_face_model = [face_model]
            source_faces = source_face_model

        else:
            logger.error("Cannot detect any Source")

        if source_faces is not None:

            target_faces = []
            for i, target_img in enumerate(target_imgs):
                if State.interrupted or model_management.processing_interrupted():
                    logger.status("Interrupted by User")
                    break
                
                target_image_md5hash = get_image_md5hash(target_img)
                if len(TARGET_IMAGE_LIST_HASH) == 0:
                    TARGET_IMAGE_LIST_HASH = [target_image_md5hash]
                    target_image_same = False
                elif len(TARGET_IMAGE_LIST_HASH) == i:
                    TARGET_IMAGE_LIST_HASH.append(target_image_md5hash)
                    target_image_same = False
                else:
                    target_image_same = True if TARGET_IMAGE_LIST_HASH[i] == target_image_md5hash else False
                    if not target_image_same:
                        TARGET_IMAGE_LIST_HASH[i] = target_image_md5hash
                
                logger.info("(Image %s) Target Image MD5 Hash = %s", i, TARGET_IMAGE_LIST_HASH[i])
                logger.info("(Image %s) Target Image the Same? %s", i, target_image_same)

                if len(TARGET_FACES_LIST) == 0:
                    logger.status(f"Analyzing Target Image {i}...")
                    target_face = analyze_faces(target_img)
                    TARGET_FACES_LIST = [target_face]
                elif len(TARGET_FACES_LIST) == i and not target_image_same:
                    logger.status(f"Analyzing Target Image {i}...")
                    target_face = analyze_faces(target_img)
                    TARGET_FACES_LIST.append(target_face)
                elif len(TARGET_FACES_LIST) != i and not target_image_same:
                    logger.status(f"Analyzing Target Image {i}...")
                    target_face = analyze_faces(target_img)
                    TARGET_FACES_LIST[i] = target_face
                elif target_image_same:
                    logger.status("(Image %s) Using Hashed Target Face(s) Model...", i)
                    target_face = TARGET_FACES_LIST[i]
                

                # logger.status(f"Analyzing Target Image {i}...")
                # target_face = analyze_faces(target_img)
                if target_face is not None:
                    target_faces.append(target_face)

            # No use in trying to swap faces if no faces are found, enhancement
            if len(target_faces) == 0:
                logger.status("Cannot detect any Target, skipping swapping...")
                return result_images

            if source_img is not None:
                # separated management of wrong_gender between source and target, enhancement
                source_face, src_wrong_gender = get_face_single(source_img, source_faces, face_index=source_faces_index[0], gender_source=gender_source, order=faces_order[1])
            else:
                # source_face = sorted(source_faces, key=lambda x: x.bbox[0])[source_faces_index[0]]
                source_face = sorted(source_faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse = True)[source_faces_index[0]]
                src_wrong_gender = 0

            if len(source_faces_index) != 0 and len(source_faces_index) != 1 and len(source_faces_index) != len(faces_index):
                logger.status(f'Source Faces must have no entries (default=0), one entry, or same number of entries as target faces.')
            elif source_face is not None:
                results = target_imgs
                model_path = model_path = os.path.join(insightface_path, model)
                face_swapper = getFaceSwapModel(model_path)

                source_face_idx = 0

                for face_num in faces_index:
                    # No use in trying to swap faces if no further faces are found, enhancement
                    if face_num >= len(target_faces):
                        logger.status("Checked all existing target faces, skipping swapping...")
                        break

                    if len(source_faces_index) > 1 and source_face_idx > 0:
                        source_face, src_wrong_gender = get_face_single(source_img, source_faces, face_index=source_faces_index[source_face_idx], gender_source=gender_source, order=faces_order[1])
                    source_face_idx += 1

                    if source_face is not None and src_wrong_gender == 0:
                        # Reading results to make current face swap on a previous face result
                        for i, (target_img, target_face) in enumerate(zip(results, target_faces)):
                            target_face_single, wrong_gender = get_face_single(target_img, target_face, face_index=face_num, gender_target=gender_target, order=faces_order[0])
                            if target_face_single is not None and wrong_gender == 0:
                                result = target_img
                                logger.status(f"Swapping {i}...")
                                if face_boost_enabled:
                                    logger.status(f"Face Boost is enabled")
                                    bgr_fake, M = face_swapper.get(target_img, target_face_single, source_face, paste_back=False)
                                    #bgr_fake, scale = restorer.get_restored_face(bgr_fake, face_restore_model, face_restore_visibility, codeformer_weight, interpolation)
                                    #M *= scale
                                    result = in_swap(target_img, bgr_fake, M)
                                else:
                                    # logger.status(f"Swapping as-is")
                                    result = face_swapper.get(target_img, target_face_single, source_face)
                                results[i] = result
                            elif wrong_gender == 1:
                                wrong_gender = 0
                                logger.status("Wrong target gender detected")
                                continue
                            else:
                                logger.status(f"No target face found for {face_num}")
                    elif src_wrong_gender == 1:
                        src_wrong_gender = 0
                        logger.status("Wrong source gender detected")
                        continue
                    else:
                        logger.status(f"No source face found for face number {source_face_idx}.")

                result_images = [Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)) for result in results]

            else:
                logger.status("No source face(s) in the provided Index")
        else:
            logger.status("No source face(s) found")
    return result_images

def in_swap(img, bgr_fake, M):
    target_img = img
    IM = cv2.invertAffineTransform(M)
    img_white = np.full((bgr_fake.shape[0], bgr_fake.shape[1]), 255, dtype=np.float32)

    # Note the use of bicubic here; this is functionally the only change from the source code
    bgr_fake = cv2.warpAffine(bgr_fake, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0, flags=cv2.INTER_CUBIC)

    img_white = cv2.warpAffine(img_white, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
    img_white[img_white > 20] = 255
    img_mask = img_white
    mask_h_inds, mask_w_inds = np.where(img_mask == 255)
    mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
    mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
    mask_size = int(np.sqrt(mask_h * mask_w))
    k = max(mask_size // 10, 10)
    # k = max(mask_size//20, 6)
    # k = 6
    kernel = np.ones((k, k), np.uint8)
    img_mask = cv2.erode(img_mask, kernel, iterations=1)
    kernel = np.ones((2, 2), np.uint8)
    k = max(mask_size // 20, 5)
    # k = 3
    # k = 3
    kernel_size = (k, k)
    blur_size = tuple(2 * i + 1 for i in kernel_size)
    img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)
    k = 5
    kernel_size = (k, k)
    blur_size = tuple(2 * i + 1 for i in kernel_size)
    img_mask /= 255
    # img_mask = fake_diff
    img_mask = np.reshape(img_mask, [img_mask.shape[0], img_mask.shape[1], 1])
    fake_merged = img_mask * bgr_fake + (1 - img_mask) * target_img.astype(np.float32)
    fake_merged = fake_merged.astype(np.uint8)
    return fake_merged

#重构FaceRestoreHelper类，改变模型路径参数
class MyFaceRestore(FaceRestoreHelper):
    """Helper for the face restoration pipeline (base class)."""

    def __init__(self,
                 upscale_factor,
                 face_size=512,
                 crop_ratio=(1, 1),
                 det_model='retinaface_resnet50',
                 model_dir = None,
                 save_ext='png',
                 template_3points=False,
                 pad_blur=False,
                 use_parse=False,
                 device=None):
        self.template_3points = template_3points  # improve robustness
        self.upscale_factor = int(upscale_factor)
        self.model_dir = model_dir
        # the cropped face ratio based on the square face
        self.crop_ratio = crop_ratio  # (h, w)
        assert (self.crop_ratio[0] >= 1 and self.crop_ratio[1] >= 1), 'crop ration only supports >=1'
        self.face_size = (int(face_size * self.crop_ratio[1]), int(face_size * self.crop_ratio[0]))
        self.det_model = det_model

        if self.det_model == 'dlib':
            # standard 5 landmarks for FFHQ faces with 1024 x 1024
            self.face_template = np.array([[686.77227723, 488.62376238], [586.77227723, 493.59405941],
                                        [337.91089109, 488.38613861], [437.95049505, 493.51485149],
                                        [513.58415842, 678.5049505]])
            self.face_template = self.face_template / (1024 // face_size)
        elif self.template_3points:
            self.face_template = np.array([[192, 240], [319, 240], [257, 371]])
        else:
            # standard 5 landmarks for FFHQ faces with 512 x 512 
            # facexlib
            self.face_template = np.array([[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
                                           [201.26117, 371.41043], [313.08905, 371.15118]])

        self.face_template = self.face_template * (face_size / 512.0)
        if self.crop_ratio[0] > 1:
            self.face_template[:, 1] += face_size * (self.crop_ratio[0] - 1) / 2
        if self.crop_ratio[1] > 1:
            self.face_template[:, 0] += face_size * (self.crop_ratio[1] - 1) / 2
        self.save_ext = save_ext
        self.pad_blur = pad_blur
        if self.pad_blur is True:
            self.template_3points = False

        self.all_landmarks_5 = []
        self.det_faces = []
        self.affine_matrices = []
        self.inverse_affine_matrices = []
        self.cropped_faces = []
        self.restored_faces = []
        self.pad_input_imgs = []

        if device is None:
            self.device = 'cuda'
        else:
            self.device = device

        self.face_detector = self.init_detection_model(det_model, half=False,model_dir=self.model_dir, device=self.device)

        # init face parsing model
        self.use_parse = use_parse
        self.face_parse = self.init_parsing_model(model_name='parsenet',model_dir=self.model_dir, device=self.device)

    def init_detection_model(self,model_name, half=False,model_dir=None,device='cuda'):
        if model_dir is None:
            raise NotImplementedError(f'{model_dir} is not implemented.')
        
        if model_name == 'retinaface_resnet50':
            model = RetinaFace(network_name='resnet50', half=half)
            model_path = os.path.join(model_dir, 'detection_Resnet50_Final.pth')            
        elif model_name == 'retinaface_mobile0.25':
            model = RetinaFace(network_name='mobile0.25', half=half)
            model_path = os.path.join(model_dir, 'detection_mobilenet0.25_Final.pth')            
        else:
            raise NotImplementedError(f'{model_name} is not implemented.')
        load_net = torch.load(model_path, map_location=lambda storage, loc: storage)
        
        print('load detection_model from ', model_path) 
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        model.load_state_dict(load_net, strict=True)
        model.eval()
        model = model.to(device)
        return model
    
    def init_parsing_model(self,model_name='bisenet', model_dir=None, device='cuda'):
        if model_dir is None:
            raise NotImplementedError(f'{model_dir} is not implemented.')
        if model_name == 'bisenet':
            model = BiSeNet(num_class=19)
            model_path = os.path.join(model_dir, 'parsing_bisenet.pth')            
        elif model_name == 'parsenet':
            model = ParseNet(in_size=512, out_size=512, parsing_ch=19)
            model_path = os.path.join(model_dir, 'parsing_parsenet.pth')            
        else:
            raise NotImplementedError(f'{model_name} is not implemented.')
        load_net = torch.load(model_path, map_location=lambda storage, loc: storage)
        print('load parsing_model from ', model_path)   
        model.load_state_dict(load_net, strict=True)
        model.eval()
        model = model.to(device)
        return model