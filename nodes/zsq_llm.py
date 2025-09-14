from transformers import AutoModelForCausalLM, AutoTokenizer,AutoProcessor, AutoModelForVision2Seq
from ..config import load_model_from_hug
from .libs.image import tensor2pil, resize_image
import torch

class LLMText:
    @classmethod
    def INPUT_TYPES(self):
        self.model = None
        self.tokenizer = None
        self.modelname = None
        return {
            "required": {                
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "default":"请输入中文提示词"}),
                "max_tokens": ("INT", {"default": 128, "step": 1, "min": 1, "max": 1024, "display": "slider"}),                   
                "system_role": (['中译英助手','提示词工程师',"提示词提炼助手"], {"default":"中译英助手"}),
                "modelname": (['Qwen/Qwen2.5-1.5B-Instruct','cognitivecomputations/dolphin-2.9.4-gemma2-2b'], {"default":'Qwen/Qwen2.5-1.5B-Instruct'}),
                "device": (['cuda', 'cpu'], {"default": 'cuda'}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("outtext",)
    FUNCTION = "run"
    CATEGORY = "ZSQ/LLM"
    
    def run(self, modelname, device,prompt, system_role,max_tokens):
        if system_role == '中译英助手':
            system_instruction = 'You are a Chinese-to-English translator,Only translate the following from Chinese to English and output：'
        elif system_role == '提示词工程师':
            system_instruction = 'You are a stable diffusion prompt word engineer, please conceive a picture according to the prompt word, describe the details in 5 paragraphs, without your subjective thoughts, translated into English output.'
        elif system_role == '提示词提炼助手':
            system_instruction = 'Please describe a picture in detail according to the keywords, and then refine the keywords in the description and separated by commas.'
            prompt = "output it into English:" + prompt
        else:
            system_instruction = system_instruction
        if self.model is None or self.modelname != modelname:
            self.unload_model()  # 卸载之前的模型        
            self.model, self.tokenizer = self.load_model(modelname, device) 
            self.modelname = modelname
        
        response = self.generate_content(self.model, self.tokenizer, prompt, system_instruction, device,max_tokens)
        response = response.replace("\n", "")
        response = response.strip()
        return(response,)
    def load_model(self,model_name, device):
        model_path = load_model_from_hug(model_name)
        model = AutoModelForCausalLM.from_pretrained(
                                model_path, 
                                device_map=device, 
                                torch_dtype="auto", 
                                trust_remote_code=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        torch.cuda.empty_cache()
        return model, tokenizer
    def generate_content(self,model, tokenizer, prompt, system_instruction, device,max_tokens=128):
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_tokens
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]    
        return response
    def unload_model(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        torch.cuda.empty_cache()  # 释放未使用的 CUDA 缓存


class LLMImage:
    def __init__(self):
        self.model = None
        self.processor = None
        self.modelname = None
    @classmethod
    def INPUT_TYPES(s):
        
        return {
            "required": {                
                "Input_Image": ("IMAGE",),
                "max_tokens": ("INT", {"default": 128, "step": 1, "min": 1, "max": 1024, "display": "slider"}),                   
                "modelname": (['Qwen/Qwen2-VL-2B-Instruct','HuggingFaceTB/SmolVLM-Instruct'], {"default": 'Qwen/Qwen2-VL-2B-Instruct'}),
                "device": (['cuda', 'cpu'], {"default": 'cuda'}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("image2text",)
    FUNCTION = "run"
    CATEGORY = "ZSQ/LLM"
    
    def run(self, modelname, device, max_tokens,Input_Image=None): 
        if self.model is None or self.modelname != modelname:
            self.unload_model()  # 卸载之前的模型
            print("Loading model:" + modelname)       
            self.model, self.processor = self.load_model(modelname, device) 
            self.modelname = modelname
        image = Input_Image[0]            
        image = tensor2pil(image)
        image = image.convert('RGB')
        image = resize_image(image)
        prompt = self.generate_content(self.model, self.processor, image, device,max_tokens)
        prompt = self.replace_image_text(prompt)
        return(prompt,)

    def load_model(self,model_name, device):
        dtype = torch.float16
        model_path = load_model_from_hug(model_name)
        model = AutoModelForVision2Seq.from_pretrained(model_path, 
                                                device_map=device, 
                                                torch_dtype=dtype,
                                                trust_remote_code=True).to(0)
        processor = AutoProcessor.from_pretrained(model_path, 
                                            trust_remote_code=True)
        return model, processor

    def generate_content(self,model, processor, image, device,max_tokens=128):
        messages = [
            {
                "role": "user",
                "content": [
                        {
                        "type": "image",
                        "image": image,  
                        },
                        {"type": "text", 
                        "text": "Can you describe the image?"
                        },
                    ],
                }
            ]

        # Preparation for inference
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        if prompt is None:
            raise ValueError("Prompt is None, check the messages format or processor configuration.")
        inputs = processor(
            text=prompt, 
            images=image, 
            return_tensors="pt").to(device)
        generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)
        output_text = processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        return output_text    
    def unload_model(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        torch.cuda.empty_cache()  # 释放未使用的 CUDA 缓存
    
    def replace_image_text(self,image_text):
        image_text = image_text.replace("User:<image>","")
        image_text = image_text.replace("Assistant:","")
        image_text = image_text.replace("system","")
        image_text = image_text.replace("You are a helpful assistant.","") 
        image_text = image_text.replace("user","")   
        image_text = image_text.replace("Can you describe the image?","") 
        image_text = image_text.replace("assistant","")
        image_text = image_text.replace('\n',"")
        return image_text
