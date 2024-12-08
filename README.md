
1.prompt
PortraitStyler
参考项目：
ComfyUI-Easy-Use：https://github.com/yolain/ComfyUI-Easy-Use.git
ComfyUI-SDXL-Style-Preview：https://github.com/xuyiqing88/ComfyUI-SDXL-Style-Preview.git
参考肖像大师风格，参考SDXL-Style-Preview子菜单生成，改为动态加载人物年龄、国籍、容貌、形态、场所、服饰等提示词，可以动态无限扩充词条，参考comfyui-sixgod_prompt，ComfyUi_PromptStylers项目的风格，其他数据大部分来自于语言模型生成。
![image](https://github.com/user-attachments/assets/a3f26cc7-ff5e-4cda-9f4b-9523a9f2657e)
2.llm
提示词翻译，使用qwen2.5 1.5b模型进行提示词翻译，也可以进行扩句，中文效果很不错的。模型第一次会从huggingface，下载到comfyui模型目录下的LLM目录。
反推提示词：使用Qwen2-VL-2B，SmolVLM两个模型实现提示词反推，在小参数模型中效果非常好。8G 显存，反推6秒左右，SDXL跑图在20秒左右。模型第一次会从huggingface，下载到comfyui模型目录下的LLM目录。
使用SmolVLM模型需要transformers>=4.46.3，当transformers==4.45时会报错。
3.loader
参考ComfyUI-Easy-Use，对常用的checkpoint、sample、lora、controlnet进行了调整，只保留了常用的参数，所有数值输入都采用滑块方式，解放键盘，模型选择全部采样子菜单方式方便选择。
![image](https://github.com/user-attachments/assets/0ac4379a-92c1-46a7-92b4-a05e9ff51328)
4.utils
对原生的Latent、CLIPEncode进行优化
![image](https://github.com/user-attachments/assets/3c33710f-dfc1-4f0a-832b-ed0dab2a951e)




