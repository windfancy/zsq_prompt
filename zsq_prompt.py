import torch
import json
import os
import random as rd
import comfy.model_management
from .config import SCRIPT_DIR,PROMPT_STYLES_DIR,FOOOCUS_STYLES_DIR,RESOURCES_DIR

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

def populate_items(styles):
    for idx, item_name in enumerate(styles):
        if item_name!="-":
            if len(item_name.split('-')) > 1:
                content = f"{item_name.split('-')[0]} /{item_name}"
            else:
                content = item_name

            styles[idx] = {"content": content}

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

class PortraitStyler:
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
            populate_items(style)
 
        return types
    
    RETURN_TYPES = ("STRING","STRING",)
    RETURN_NAMES = ("positive","negative",)
    OUTPUT_IS_LIST = (True,False)
    FUNCTION = "run"
    CATEGORY = "ZSQ/Prompt"
    
    def run(self,**required):
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
                
        return(prompts,negative)



    


