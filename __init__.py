import os
from .zsq_prompt import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
cwd_path = os.path.dirname(os.path.realpath(__file__))
directory = ".\web"

WEB_DIRECTORY = directory

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
print(f'\033[34m[zsq_prompt] web root: \033[0m{os.path.join(cwd_path, directory)} \033[92mLoaded\033[0m')








