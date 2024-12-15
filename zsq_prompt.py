import json
import os
import random as rd
from aiohttp import web
from server import PromptServer
from .config import SCRIPT_DIR,PROMPT_STYLES_DIR,FOOOCUS_STYLES_DIR

#get style list
@PromptServer.instance.routes.get("/prompt/styles")
async def getStylesList(request):
    if "name" in request.rel_url.query:
        name = request.rel_url.query["name"]
        if name == 'fooocus_styles':
            file = os.path.join(FOOOCUS_STYLES_DIR, name+'.json')
            cn_file = os.path.join(FOOOCUS_STYLES_DIR, name + '_cn.json')
        else:
            file = os.path.join(FOOOCUS_STYLES_DIR, name+'.json')
            cn_file = os.path.join(FOOOCUS_STYLES_DIR, name + '_cn.json')
        cn_data = None
        if os.path.isfile(cn_file):
            f = open(cn_file, 'r', encoding='utf-8')
            cn_data = json.load(f)
            f.close()
        if os.path.isfile(file):
            f = open(file, 'r', encoding='utf-8')
            data = json.load(f)
            f.close()
            if data:
                ndata = []
                for d in data:
                    nd = {}
                    name = d['name'].replace('-', ' ')
                    words = name.split(' ')
                    key = ' '.join(
                        word.upper() if word.lower() in ['mre', 'sai', '3d'] else word.capitalize() for word in
                        words)
                    img_name = '_'.join(words).lower()
                    if "name_cn" in d:
                        nd['name_cn'] = d['name_cn']
                    elif cn_data:
                        nd['name_cn'] = cn_data[key] if key in cn_data else key
                    nd["name"] = d['name']
                    nd['imgName'] = img_name
                    if "prompt" in d:
                        nd['prompt'] = d['prompt']
                    if "negative_prompt" in d:
                        nd['negative_prompt'] = d['negative_prompt']
                    ndata.append(nd)
                return web.json_response(ndata)
    return web.Response(status=400)

# get style preview image
@PromptServer.instance.routes.get("/prompt/styles/image")
async def getStylesImage(request):
    styles_name = request.rel_url.query["styles_name"] if "styles_name" in request.rel_url.query else None
    if "name" in request.rel_url.query:
        name = request.rel_url.query["name"]
        if os.path.exists(os.path.join(FOOOCUS_STYLES_DIR, 'samples')):
            file = os.path.join(FOOOCUS_STYLES_DIR, 'samples', name + '.jpg')
            file1 = os.path.join(FOOOCUS_STYLES_DIR, 'samples', name + '.png')
            if os.path.isfile(file) or os.path.isfile(file1):
                file = file1 if os.path.isfile(file1) else file
                return web.FileResponse(file)
            elif styles_name == 'fooocus_styles':
                return web.Response(status=400)
        elif styles_name == 'fooocus_styles':
            return web.Response(status=400)
    return web.Response(status=400)

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
            res1= res1 + f",A {gender},({str(age)}-year-old):1.5"
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

# 风格提示词选择器
# easy use：stylesPromptSelector
class stylesPromptSelector:

    @classmethod
    def INPUT_TYPES(s):
        #styles = ["fooocus_styles"]
        styles = []
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

    CATEGORY = 'ZSQ/Prompt'
    FUNCTION = 'run'

    def run(self, styles, positive='', negative='', prompt=None, extra_pnginfo=None, my_unique_id=None):
        values = []
        all_styles = {}
        positive_prompt, negative_prompt = '', negative
        file = os.path.join(FOOOCUS_STYLES_DIR, styles + '.json')
        f = open(file, 'r', encoding='utf-8')
        data = json.load(f)
        f.close()
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

    
class BatchPromptSelector:
    @classmethod
    def INPUT_TYPES(s):
        #styles = ["fooocus_styles"]
        styles = []
        styles_dir = FOOOCUS_STYLES_DIR
        for file_name in os.listdir(styles_dir):
            file = os.path.join(styles_dir, file_name)
            if os.path.isfile(file) and file_name.endswith(".json"):
                styles.append(file_name.split(".")[0])
        return {
            "required": {
               "positive": ("STRING", {"default":""}),
               "styles": (styles, {"default": "fooocus_styles"}),
               "batch_start": ("INT", {"default": 0}),
               "batch_length": ("INT", {"default": 100}),
            },
        }

    RETURN_TYPES = ("STRING","STRING", "STRING",)
    RETURN_NAMES = ("name", "positive", "negative",)
    OUTPUT_IS_LIST = (True,True,True)
    CATEGORY = 'ZSQ/Prompt'
    FUNCTION = 'run'

    def run(self, styles, positive='', batch_start=0, batch_length=100):
        values = []
        prompt_positive=[]
        prompt_negative=[]
        prompt_name = []
        file = os.path.join(FOOOCUS_STYLES_DIR, styles + '.json')
        f = open(file, 'r', encoding='utf-8')
        data = json.load(f)
        f.close()
        for d in data:
            values.append(d)
        max_lenth = len(values) 
        batch_end= min(max_lenth,batch_start + batch_length)
        for i in range(batch_start, batch_end):
            name_tmp = values[i]["name"]
            name_tmp = name_tmp.replace(' ', '_')
            name_tmp = name_tmp.replace('-', '_')
            prompt_name.append(name_tmp)
            if 'prompt' in values[i]["prompt"]:
                prompt_positive.append(values[i]["prompt"].replace("{prompt}", f"{positive},"))
            else:
                prompt_positive.append(values[i]["prompt"] + ',')
            prompt_negative.append(values[i]["negative_prompt"])



        return (prompt_name,prompt_positive, prompt_negative)
    
class BatchPromptJson:
    @classmethod
    def INPUT_TYPES(s):
        #styles = ["fooocus_styles"]
        styles = []
        styles_dir = PROMPT_STYLES_DIR
        for file_name in os.listdir(styles_dir):
            file = os.path.join(styles_dir, file_name)
            if os.path.isfile(file) and file_name.endswith(".json"):
                styles.append(file_name.split(".")[0])
        return {
            "required": {
               "positive": ("STRING", {"default":""}),
               "styles": (styles, {"default": "fooocus_styles"}),
               "batch_start": ("INT", {"default": 0}),
               "batch_length": ("INT", {"default": 100}),
            },
        }

    RETURN_TYPES = ("STRING","STRING",)
    RETURN_NAMES = ("name", "positive",)
    OUTPUT_IS_LIST = (True,True,)
    CATEGORY = 'ZSQ/Prompt'
    FUNCTION = 'run'

    def run(self, styles, positive='', batch_start=0, batch_length=100):
        values = []
        prompt_positive=[]
        prompt_name = []
        file = os.path.join(PROMPT_STYLES_DIR, styles + '.json')
        f = open(file, 'r', encoding='utf-8')
        data = json.load(f)
        f.close()
        values = list(data.items())
        max_lenth = len(values) 
        batch_end= min(max_lenth,batch_start + batch_length)
        for i in range(batch_start, batch_end):
            name, value = values[i]
            name = name.replace(' ', '_')
            name = name.replace('-', '_')
            prompt_name.append(name)
            prompt_positive.append(f"{positive},{value}")

        return (prompt_name,prompt_positive,)