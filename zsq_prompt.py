import torch
import json
import shutil
import os
import yaml
import random as rd
import comfy.model_management
from .config import BASE_RESOLUTIONS,SCRIPT_DIR,RESOURCES_DIR,FOOOCUS_STYLES_DIR,PROMPT_STYLES_DIR,PROMPT_TEST_DIR

import numpy as np
from PIL import Image, ImageOps, ImageSequence
import hashlib
import inspect
from server import PromptServer
from aiohttp import web

def read_json_file(file_path):
    with open(file_path,'r',encoding='utf-8') as f:
        json_data = json.load(f) 
    f.close()
    return json_data
    
def get_all_json_files(directory):
    json_files = []
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if file.endswith('.json') and os.path.isfile(file_path):
            file_name =os.path.splitext(os.path.basename(file_path))[0]
            json_files.append({file_path:file_name})
    return json_files
      
def load_styles_from_directory(directory):
    json_files = get_all_json_files(directory)
    combined_data = {}
    style_names = []
    if isinstance(json_files, list) and all(isinstance(x, dict) for x in json_files):
        for json_dict in json_files:
            for json_file,json_name in json_dict.items():
                json_data = read_json_file(json_file)
                combined_sub=[]
                if json_data:  
                    combined_sub.append(json_data)
                combined_data[json_name]=combined_sub 
                style_names.append(json_name)  
    return combined_data,style_names

def get_list_name_dict(json_data,key):
    list_name = []
    value_list = json_data[key]    
    if isinstance(value_list, list):
        for index, item in enumerate(value_list):
            if isinstance(item, dict):
                for dict_key, dict_value in item.items():
                    list_name.append(dict_key)
    return list_name

def get_list_dict(json_data):
    list_name = []
    for key in json_data:
        list_name.append(key)
    return list_name

def select_value_by_name(json_data,key,tmp_name):
    value_list = json_data[key]
    res = ""
    if isinstance(value_list, list):
        for index, item in enumerate(value_list):
            if isinstance(item, dict):
                for dict_key, dict_value in item.items():
                    if dict_key == tmp_name:
                        res = dict_value
    return res
def select_name_by_value(json_data,key,tmp_value):
    value_list = json_data[key]
    res = ""
    if isinstance(value_list, list):
        for index, item in enumerate(value_list):
            if isinstance(item, dict):
                for dict_key, dict_value in item.items():
                    if dict_value == tmp_value:
                        res = dict_key
    return res

@PromptServer.instance.routes.get("/preview/{name}")
async def view(request):
    name = request.match_info["name"]

    image_path = name
    filename = os.path.basename(image_path)
    return web.FileResponse(image_path, headers={"Content-Disposition": f"filename=\"{filename}\""})
def populate_items(styles, item_type):
    for idx, item_name in enumerate(styles):
        if item_name!="-":
            current_directory = os.path.dirname(os.path.abspath(__file__))            

            if len(item_name.split('-')) > 1:
                preview_item = item_name.split('-')[-1]
                content = f"{item_name.split('-')[0]} /{item_name}"
                preview_path = os.path.join(current_directory, item_type, preview_item + ".png")
            else:
                content = item_name
                preview_path = ""

            if os.path.exists(preview_path):
                styles[idx] = {
                    "content": content,
                    "preview": preview_path
                }
            else:
                #print(f"Warning: Preview image '{preview_item}.png' not found for item '{preview_item}'")
                styles[idx] = {
                    "content": content,
                    "preview": None
                }

def get_prompt_from_key(key,required):
    list_value_dict = required.get(key, "")
    if isinstance(list_value_dict, dict):
        list_value = list_value_dict["content"].split("/")[-1]  
    else: 
        list_value = list_value_dict
    return list_value

def set_prompt_weight_from_key(key,prompt_tmp,config):                   
    value = config.get(key)                
    if value is not None:
        value_float = float(value)
        #value取值[0,2]
        if value_float > 2:value="2"
        if value_float < 0:value="0"
        res = f"({prompt_tmp}):{value}"                    
    else:
        res =  prompt_tmp
    return res
def get_prompt(key,required,options,config):  
    res = ""      
    list_value = get_prompt_from_key(key,required)
    prompt_tmp = ""        
    if list_value != "nothing":            
        prompt_tmp = select_value_by_name(options,key,list_value) 
        res = set_prompt_weight_from_key(key,prompt_tmp,config)                
    return res

def get_prompt_rd(key,options,config):
    res = ""
    prompt_tmp = ""
    list_dict = options[key][0]                             
    index = rd.randint(0, len(list_dict.items())-1)
    prompt_tmp = list(list_dict.values())[index] 
    if prompt_tmp != "nothing":              
        res = set_prompt_weight_from_key(key,prompt_tmp,config)               
    return res

def get_propmt_style(self,required,Random,Random_Flag):
    gender = required['GO_gender']
    age = required['GO_age']
    Random_Item = required['GO_Random_Item']  
    list_keys = []
    for key in required.keys():            
        if not "GO_" in key:
            list_keys.append(key)
    config = self.config
    options = self.json_data
    res1 = ""
    if gender!= "-":
            res1= res1 + f",{gender},({str(age)}-year-old):1.5"
    if not Random_Flag:
        for key in list_keys:
            tmp = get_prompt(key,required,options,config)
            if tmp != "":
                res1 = res1 + "," + tmp
    elif Random == "ALL":
        for key in list_keys:
            res1 = res1 + "," + get_prompt_rd(key,options,config)                
    elif Random == "ONLY":
        for key in list_keys:
            if Random_Item == key:
                res1 = res1 + "," + get_prompt_rd(key,options,config)
            else:
                tmp = get_prompt(key,required,options,config)
                if tmp != "":
                    res1 = res1 + "," + tmp
    elif Random == "EXCEPT":
        for key in list_keys:
            if Random_Item != key:
                res1 = res1 + "," + get_prompt_rd(key,options,config)
            else:
                tmp = get_prompt(key,required,options,config)
                if tmp != "":
                    res1 = res1 + "," + tmp
    elif Random == "NOSELECT":
        for key in list_keys:
            value_tmp = get_prompt_from_key(key,required)
            if value_tmp == "nothing":
                res1 = res1 + "," + get_prompt_rd(key,options,config)
            else:
                tmp = get_prompt(key,required,options,config)
                if tmp != "":
                    res1 = res1 + "," + tmp
    elif Random == "SELECTED":
        for key in list_keys:
            value_tmp = get_prompt_from_key(key,required)
            if value_tmp != "nothing":
                res1 = res1 + "," + get_prompt_rd(key,options,config)            
    else:
        pass
    return res1    

class PromptStyler:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(self):
        # 获取当前脚本所在的路径
        config_file_path = os.path.join(SCRIPT_DIR, 'prompt_weight.json')
        self.config = read_json_file(config_file_path)

        styles,style_names=load_styles_from_directory(PROMPT_STYLES_DIR)
        self.json_data = styles

        widgets = {}
        for key in styles:
            _list = get_list_name_dict(styles,key)
            _list.insert(0, 'nothing')
            widgets[key] = ((_list),)

        types = {
            "required":{
                        "GO_positive": ("STRING", {"default": "", "multiline": True}),
                        "GO_negative": ("STRING", {"default": "", "multiline": True}),
                        "GO_Random": (["NO", "ONLY", "EXCEPT", "NOSELECT","SELECTED", "ALL"], {"default": "NO"}), 
                        "GO_ImageNum": ("INT", {"default": 1, "step": 1, "min": 1, "max": 50, "display": "slider"}),                   
                        "GO_Random_Item":((style_names),),
                        "GO_gender": (["-", "Man", "Woman"], {"default": "-"}),
                        "GO_age": ("INT", {"default": 30, "step": 1, "min": 18, "max": 80, "display": "slider"}),
                        **widgets,},
            }
        for key in style_names:
            style = types["required"][key][0]
            populate_items(style, "images")
 
        return types
    
    RETURN_TYPES = ("LIST","STRING","STRING",)
    RETURN_NAMES = ("positive_list","positive","negative",)
    OUTPUT_IS_LIST = (True,True,False)
    FUNCTION = "prompt_styler"
    CATEGORY = "ZSQ"
    
    def prompt_styler(self,**required):
        Random = required["GO_Random"]
        ImageNum = required["GO_ImageNum"]        
        positive = required['GO_positive']
        negative = required["GO_negative"]
    
        prompts = []
        if Random != "NO": 
            for i in range(ImageNum):
                res1 = get_propmt_style(self,required,Random,True)
                res1 = positive + "," + res1
                prompts.append(res1)                
        else:    
            res1 = get_propmt_style(self,required,Random,False)
            res1 = positive + "," + res1
            prompts.append(res1)
                
        return(prompts,prompts,negative)

    
resolution_strings = [f"{width} x {height} (custom)" if width == 'width' and height == 'height' else f"{width} x {height}" for width, height in BASE_RESOLUTIONS]
class PromptLatent:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(self):
        return {"required": 
                {
                    "resolution": (resolution_strings,{"default": "1024 x 1024"}),       
                    "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096, "tooltip": "The number of latent images in the batch."}) 
                 },
            }
        
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "run"
    CATEGORY = "ZSQ"

    def run(self, resolution, batch_size):
        width, height = resolution.split(" x ")
        latent = torch.zeros([batch_size, 4, int(height) // 8, int(width) // 8], device=self.device)
        return ({"samples":latent}, )

class PromptCLIPEncode:
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
    CATEGORY = "ZSQ"
    
    def CLIPEncode(self, clip, positive, negative):
        positive_tokens = clip.tokenize(positive)
        negative_tokens = clip.tokenize(negative)
        positive_output = clip.encode_from_tokens(positive_tokens, return_pooled=True, return_dict=True)
        negative_output = clip.encode_from_tokens(negative_tokens, return_pooled=True, return_dict=True)
        positive_cond = positive_output.pop("cond")
        negative_cond = negative_output.pop("cond")
        return ([[positive_cond, positive_output]], [[negative_cond,negative_output]], )
    
class PromptSelector:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        styles = ["fooocus_styles"]
        styles_dir = FOOOCUS_STYLES_DIR
        for file_name in os.listdir(styles_dir):
            file = os.path.join(styles_dir, file_name)
            if os.path.isfile(file) and file_name.endswith(".json"):
                styles.append(file_name.split(".")[0])
        return {
            "required": {
               "styles": (styles, {"default": "fooocus_styles"}),
            },
            "optional": {
                "positive": ("STRING", {"forceInput": True}),
                "negative": ("STRING", {"forceInput": True}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("positive", "negative",)

    CATEGORY = 'ZSQ'
    FUNCTION = 'run'

    def run(self, styles, positive='', negative='', prompt=None, extra_pnginfo=None, my_unique_id=None):
        values = []
        all_styles = {}
        positive_prompt, negative_prompt = '', negative
        if styles == "fooocus_styles":
            file = os.path.join(RESOURCES_DIR,  styles + '.json')
        else:
            file = os.path.join(FOOOCUS_STYLES_DIR, styles + '.json')
        data = read_json_file(file)
        print(data)
        for d in data:
            all_styles[d['name']] = d
        if my_unique_id in prompt:
            if prompt[my_unique_id]["inputs"]['select_styles']:
                values = prompt[my_unique_id]["inputs"]['select_styles'].split(',')

        has_prompt = False
        if len(values) == 0:
            return (positive, negative)

        for index, val in enumerate(values):
            if 'prompt' in all_styles[val]:
                if "{prompt}" in all_styles[val]['prompt'] and has_prompt == False:
                    positive_prompt = all_styles[val]['prompt'].replace('{prompt}', positive)
                    has_prompt = True
                else:
                    positive_prompt += ', ' + all_styles[val]['prompt'].replace(', {prompt}', '').replace('{prompt}', '')
            if 'negative_prompt' in all_styles[val]:
                negative_prompt += ', ' + all_styles[val]['negative_prompt'] if negative_prompt else all_styles[val]['negative_prompt']

        if has_prompt == False and positive:
            positive_prompt = positive + ', '

        return (positive_prompt, negative_prompt)

NODE_CLASS_MAPPINGS = {
    "PromptStyler": PromptStyler,
    "PromptLatent": PromptLatent,
    "PromptCLIPEncode":PromptCLIPEncode,
    "PromptSelector": PromptSelector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptStyler": "PromptStyler",
    "PromptLatent": "PromptLatent",
    "PromptCLIPEncode":"PromptCLIPEncode",
    "PromptSelector": "PromptSelector"
}
