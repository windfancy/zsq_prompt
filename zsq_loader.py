import torch
import folder_paths
import latent_preview
import comfy.utils, comfy.sample, comfy.samplers, comfy.model_management
from .config import RESOLUTION_STRINGS,MAX_LORA_NUM,MAX_CN_NUM,NEGATIVE_PROMPT

def prompt_to_conditioning(clip, text):
        tokens = clip.tokenize(text)
        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        cond = output.pop("cond")
        return ([[cond, output]])    
def load_checkpoint_Simple(ckpt_name):
    ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
    out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
    return (out[0], out[1], out[2])
def empty_latent(width=512, height=512, batch_size=1, device="cuda"):
    latent = torch.zeros([batch_size, 4, int(height) // 8, int(width) // 8], device=device)
    return ({"samples":latent},)
def load_controlnet(control_net_name):
        controlnet_path = folder_paths.get_full_path_or_raise("controlnet", control_net_name)
        print("Loading controlnet:", controlnet_path)   
        controlnet = comfy.controlnet.load_controlnet(controlnet_path)
        return (controlnet,)

def common_ksampler(model,seed, steps, cfg, sampler_name, scheduler, positive, negative, 
                    denoise=1.0, latent=None,
                    disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        #batch_inds = input_latent["batch_index"] if "batch_index" in input_latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed)

    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                force_full_denoise=force_full_denoise, noise_mask=None, callback=callback, disable_pbar=disable_pbar, seed=seed)

    out = latent.copy()
    out["samples"] = samples
    return (out, )
def get_content_from_key(list_value_dict):
    if isinstance(list_value_dict, dict):
        list_value = list_value_dict["content"]  
    else: 
        list_value = list_value_dict
    return list_value
def populate_items(names):
    for idx, item_name in enumerate(names):
        names[idx] = {"content": item_name}
    #names.sort(key=lambda i: i["content"].lower())

class checkpoint_sampler:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()
        self.model = None
        self.model_name = None
    @classmethod
    def INPUT_TYPES(cls):

        types = {"required": {
            "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
            "resolution": (RESOLUTION_STRINGS,),          
            "positive_text": ("STRING", {"default": "", "placeholder": "Positive", "multiline": True}),
            "negative_text": ("STRING", {"default": NEGATIVE_PROMPT, "placeholder": "Negative", "multiline": True}),
            "batch_size": ("INT", {"default": 1, "step": 1, "min": 1, "max": 20, "display": "slider"}),
            "steps": ("INT", {"default": 20, "step": 1, "min": 1, "max": 100, "display": "slider"}),
            "cfg": ("FLOAT", {"default": 7.5, "step": 0.1, "min": 0.0, "max": 30.0, "display": "slider"}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "The random seed used for creating the noise."}),
            "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
            "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
            "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),        
        },
        "optional": {"optional_lora_stack": ("ZSQ_LORA_STACK",), 
                     "optional_controlnet_stack": ("ZSQ_CONTROL_NET_STACK",),
            }  
        }
        
        names = types["required"]["ckpt_name"][0]
        populate_items(names)
        return types
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "ZSQ/Loaders"

    def run(self, resolution,**required,):
        # load required
        try:
            ckpt_name = required.get("ckpt_name", "")
            positive_text = required.get("positive_text", "")
            negative_text = required.get("negative_text", "")
            batch_size = required.get("batch_size", 1)
            seed = required.get("seed", 0)
            steps = required.get("steps", 50)
            cfg = required.get("cfg", 7.5)
            sampler_name = required.get("sampler_name", "euler")
            scheduler = required.get("scheduler", "normal")
            denoise = required.get("denoise", 1.0)
            optional_lora_stack = required.get("optional_lora_stack", None)
            optional_controlnet_stack = required.get("optional_controlnet_stack", None)
            #get ckpt_name from {"content": ckpt_name}
            ckpt_name = get_content_from_key(ckpt_name)
        except Exception as e:
            print(e)
            return empty_latent()

        images =None        

        # Clean models from loaded_objects
        if self.model_name != ckpt_name:
            self.model = None
            self.model_name = ckpt_name
        # Load models
        if self.model is None:
            print("Loading models:" + ckpt_name)
            self.model, self.clip, self.vae = load_checkpoint_Simple(ckpt_name)

        # Lora Stack Apply
        model,clip = self.apply_lorastack(optional_lora_stack)


        # Empty Latent width, height
        width, height = resolution.split(" x ")
        if width == 'width':
            width = 512
        if height == 'height':
            height = 512
        # Prompt to Conditioning
        positive_cond = prompt_to_conditioning(clip, positive_text)
        negative_cond = prompt_to_conditioning(clip, negative_text)

        # Controlnet Stack Apply
        positive_cond = self.apply_controlnetstack(positive_cond, optional_controlnet_stack)[0]

        latents = empty_latent(width, height, batch_size,self.device)
        samples = common_ksampler(model,seed, steps, cfg, sampler_name, scheduler,positive_cond, negative_cond, denoise,latents[0])
        images = self.vae.decode(samples[0]["samples"])
        return(images,)
    
    def apply_lorastack(self, optional_lora_stack):
        model = self.model
        clip = self.clip
        if optional_lora_stack is not None and len(optional_lora_stack) >0:
            model_lora= model
            clip_lora = clip
            for optional_lora in optional_lora_stack: 
                if optional_lora is not None:
                    lora_strength = optional_lora.get("lora_strength")
                    lora_path = optional_lora.get("lora_path")
                    if lora_path is not None:
                        print("Loading LoRA: " + lora_path)
                        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                        model_lora, clip_lora = comfy.sd.load_lora_for_models(model_lora, clip_lora, lora, lora_strength, lora_strength)
            del lora            
            return model_lora, clip_lora
        else:
            return model, clip

    def apply_controlnetstack(self, conditioning, control_net, image, strength):
        if strength == 0:
            return (conditioning, )
        c = []
        control_hint = image.movedim(-1,1)
        for t in conditioning:
            n = [t[0], t[1].copy()]
            c_net = control_net.copy().set_cond_hint(control_hint, strength)
            if 'control' in t[1]:
                c_net.set_previous_controlnet(t[1]['control'])
            n[1]['control'] = c_net
            n[1]['control_apply_to_uncond'] = True
            c.append(n)
        return (c, )
    def apply_controlnetstack(self,conditioning,optional_cn_stack):
        cond = conditioning
        if optional_cn_stack is not None and len(optional_cn_stack) >0:
            for optional_cn in optional_cn_stack:
                controlnet_strength = optional_cn.get("controlnet_strength", 1)
                if controlnet_strength == 0:
                    continue
                controlnet_name = optional_cn.get("controlnet_name", "")
                controlnet_image = optional_cn.get("image", None)

                control_net = load_controlnet(controlnet_name)[0]              

                control_hint = controlnet_image.movedim(-1,1)
                p = []
                for t in cond:
                    n = [t[0], t[1].copy()]
                    c_net = control_net.copy().set_cond_hint(control_hint, controlnet_strength)
                    if 'control' in t[1]:
                        c_net.set_previous_controlnet(t[1]['control'])
                    n[1]['control'] = c_net
                    n[1]['control_apply_to_uncond'] = True
                    p.append(n)
                cond = p

        return (cond, )


    
    
# lora simple
class zsqloraStack:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "num_loras": ("INT", {"default": 1, "step": 1, "min": 1, "max":  MAX_LORA_NUM, "display": "slider"}),
            },
            "optional": {

            },
        }

        for i in range(1, MAX_LORA_NUM+1):
            inputs["optional"][f"lora_{i}_name"] = (
            ["None"] + folder_paths.get_filename_list("loras"),)
            names = inputs["optional"][f"lora_{i}_name"][0]
            populate_items(names)
            inputs["optional"][f"lora_{i}_strength"] = (
            "FLOAT", {"default": 1.0, "min": 0.0, "max": 2.5, "step": 0.01,"display": "slider"})

        return inputs
    
    RETURN_TYPES = ("ZSQ_LORA_STACK",)
    RETURN_NAMES = ("lora_stack",)
    FUNCTION = "stack"
    CATEGORY = "ZSQ/Loaders"

    def stack(self, num_loras, optional_lora_stack=None, **kwargs):

        loras = []
        # Import Stack values
        if optional_lora_stack is not None:
            loras.extend([l for l in optional_lora_stack if l[0] != "None"])
        # Import Lora values
        for i in range(1, num_loras + 1):
            lora_name = kwargs.get(f"lora_{i}_name")
            #get lora_name from {"content": lora_name}
            lora_name = get_content_from_key(lora_name)
            if not lora_name or lora_name == "None":
                continue
            lora_strength = float(kwargs.get(f"lora_{i}_strength"))
            lora_path = folder_paths.get_full_path("loras", lora_name)
            loras.append({"lora_path":lora_path, "lora_strength":lora_strength})

        return (loras,)

# lora simple
class zsqloraStack_2:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP", {"tooltip": "The CLIP model used for encoding the text."}),
                "num_loras": ("INT", {"default": 1, "step": 1, "min": 1, "max":  MAX_LORA_NUM, "display": "slider"}),
            },
            "optional": {

            },
        }

        for i in range(1, MAX_LORA_NUM+1):
            inputs["optional"][f"lora_{i}_name"] = (["None"] + folder_paths.get_filename_list("loras"),)
            names = inputs["optional"][f"lora_{i}_name"][0]
            populate_items(names)
            inputs["optional"][f"lora_{i}_strength"] = (
            "FLOAT", {"default": 1.0, "min": 0.0, "max": 2.5, "step": 0.01,"display": "slider"})

        return inputs
    
    RETURN_TYPES = ("MODEL","CLIP",)
    RETURN_NAMES = ("model","clip",)
    FUNCTION = "apply_lora"
    CATEGORY = "ZSQ/Loaders"

    def apply_lora(self, num_loras, model,clip, **kwargs):
        loras = []
        # Import Stack values
        
        if num_loras is not None and num_loras >0:
            model_lora= model
            clip_lora = clip
            for i in range(1, num_loras + 1):
                lora_name = kwargs.get(f"lora_{i}_name")
                #get lora_name from {"content": lora_name}
                lora_name = get_content_from_key(lora_name)
                if not lora_name or lora_name == "None":
                    continue
                lora_strength = float(kwargs.get(f"lora_{i}_strength"))
                lora_path = folder_paths.get_full_path("loras", lora_name) 
                if lora_path is not None:
                    print("Loading LoRA: " + lora_path)
                    lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                    model_lora, clip_lora = comfy.sd.load_lora_for_models(model_lora, clip_lora, lora, lora_strength, lora_strength)
                    del lora            
            return model_lora, clip_lora
        else:
            return model, clip

# controlnetStack simple
class zsqcontrolnetStack:
    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "num_controlnet": ("INT", {"default": 1, "step": 1, "min": 1, "max":  MAX_CN_NUM, "display": "slider"}),
            },
            "optional": {                
            }
        }

        for i in range(1, MAX_CN_NUM+1):
            inputs["optional"][f"controlnet_{i}"] = (["None"] + folder_paths.get_filename_list("controlnet"), )
            names = inputs["optional"][f"controlnet_{i}"][0]
            populate_items(names)
            inputs["optional"][f"controlnet_{i}_strength"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,"display": "slider"})
            inputs["optional"][f"image_{i}"] = ("IMAGE",)
        return inputs

    RETURN_TYPES = ("ZSQ_CONTROL_NET_STACK",)
    RETURN_NAMES = ("controlnet_stack",)
    FUNCTION = "stack"
    CATEGORY = "ZSQ/Loaders"

    def stack(self, num_controlnet, optional_controlnet_stack=None, **kwargs):
        controlnets = []
        # Import Stack values
        if optional_controlnet_stack is not None:
            controlnets.extend([l for l in optional_controlnet_stack if l[0] != "None"])
        # Import Controlnet values
        for i in range(1, num_controlnet+1):
            controlnet_name = kwargs.get(f"controlnet_{i}")
            controlnet_name = get_content_from_key(controlnet_name)
            if not controlnet_name or controlnet_name == "None":
                continue
            controlnet_strength = float(kwargs.get(f"controlnet_{i}_strength"))
            image = kwargs.get(f"image_{i}")
            controlnet_style = {"controlnet_name": controlnet_name, "controlnet_strength": controlnet_strength,"image": image}
            controlnets.append(controlnet_style)

        return (controlnets,)    

class zsqcontrolnetstack_2:
    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "positive": ("CONDITIONING", ),
                "num_controlnet": ("INT", {"default": 1, "step": 1, "min": 1, "max":  MAX_CN_NUM, "display": "slider"}),
            },
            "optional": {                
            }
        }
        
        for i in range(1, MAX_CN_NUM+1):
            inputs["optional"][f"controlnet_{i}"] = (["None"] + folder_paths.get_filename_list("controlnet"),)
            names = inputs["optional"][f"controlnet_{i}"][0]
            populate_items(names)
            inputs["optional"][f"controlnet_{i}_strength"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,"display": "slider"})
            inputs["optional"][f"image_{i}"] = ("IMAGE",)

        return inputs

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("positive", )
    FUNCTION = "apply_controlnet"

    CATEGORY = "ZSQ/Loaders"

    def apply_controlnet(self, num_controlnet, positive, **kwargs):
        if positive is None:
            return None
        for i in range(1, num_controlnet+1):
            controlnet_name = kwargs.get(f"controlnet_{i}", "None")
            controlnet_strength = kwargs.get(f"controlnet_{i}_strength", 1.0)
            image = kwargs.get(f"image_{i}", None)

            controlnet_name = get_content_from_key(controlnet_name)
            if controlnet_name == "None":
                continue
            else:
                control_net = load_controlnet(controlnet_name)[0]

            if controlnet_strength == 0:
                continue

            control_hint = image.movedim(-1,1)
            cnets = {}

            c = []
            for t in positive:
                d = t[1].copy()
                prev_cnet = d.get('control', None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    c_net = control_net.copy().set_cond_hint(control_hint, controlnet_strength)
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net

                d['control'] = c_net
                d['control_apply_to_uncond'] = False
                n = [t[0], d]
                c.append(n)
            positive = c

        return (positive, )

class zsqcheckpoint:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()
        self.model = None
        self.model_name = None
    @classmethod
    def INPUT_TYPES(cls):

        types = {"required": {
            "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
            },
        }
        
        names = types["required"]["ckpt_name"][0]
        populate_items(names)
        return types
    
    RETURN_TYPES = ("MODEL","CLIP","VAE",)
    RETURN_NAMES = ("model","clip","vae",)
    FUNCTION = "run"
    CATEGORY = "ZSQ/Loaders"

    def run(self, ckpt_name,):
        # load ckpt_name
        ckpt_name = get_content_from_key(ckpt_name) 
        # Clean models from loaded_objects
        if self.model_name != ckpt_name:
            self.model = None
            self.model_name = ckpt_name
        # Load models
        if self.model is None:
            print("Loading models:" + ckpt_name)
            self.model, self.clip, self.vae = load_checkpoint_Simple(ckpt_name)
        return self.model, self.clip, self.vae

class zsqsampler:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()
        self.model = None
        self.model_name = None
    @classmethod
    def INPUT_TYPES(cls):

        types = {"required": {
            "model": ("MODEL",),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "vae": ("VAE", ),
            "latent": ("LATENT",),
            "steps": ("INT", {"default": 20, "step": 1, "min": 1, "max": 100, "display": "slider"}),
            "cfg": ("FLOAT", {"default": 7.5, "step": 0.1, "min": 0.0, "max": 30.0, "display": "slider"}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "The random seed used for creating the noise."}),
            "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
            "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
            "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),        
            },
        }
        return types
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "ZSQ/Loaders"

    def run(self, model,positive,negative,vae,latent,**required,):
        # load required
        seed = required.get("seed", 0)
        steps = required.get("steps", 50)
        cfg = required.get("cfg", 7.5)
        sampler_name = required.get("sampler_name", "euler")
        scheduler = required.get("scheduler", "normal")
        denoise = required.get("denoise", 1.0)
        images =None

        samples = common_ksampler(model,seed, steps, cfg, sampler_name, scheduler,positive, negative, denoise,latent)
        images = vae.decode(samples[0]["samples"])
        return(images,)
    

class zsqcontrolnet:
    @classmethod
    def INPUT_TYPES(s):
        inputs =  {"required": {"positive": ("CONDITIONING", ),
                             "image": ("IMAGE", ),
                             "controlnet_name":(["None"] + folder_paths.get_filename_list("controlnet"),),
                             "controlnet_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                             },
                    }
        names = inputs["required"]["controlnet_name"][0]
        populate_items(names)
        return inputs

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("positive", )
    FUNCTION = "apply_controlnet"

    CATEGORY = "ZSQ/Loaders"

    def apply_controlnet(self, controlnet_name,positive, image, controlnet_strength):
        controlnet_name = get_content_from_key(controlnet_name)
        if controlnet_name == "None":
            return (positive,)
        else:
            control_net = load_controlnet(controlnet_name)[0]

        if controlnet_strength == 0:
            return (positive,)

        control_hint = image.movedim(-1,1)
        cnets = {}

        c = []
        for t in positive:
            d = t[1].copy()
            prev_cnet = d.get('control', None)
            if prev_cnet in cnets:
                c_net = cnets[prev_cnet]
            else:
                c_net = control_net.copy().set_cond_hint(control_hint, controlnet_strength)
                c_net.set_previous_controlnet(prev_cnet)
                cnets[prev_cnet] = c_net

            d['control'] = c_net
            d['control_apply_to_uncond'] = False
            n = [t[0], d]
            c.append(n)
        return (c, )
