import os

SCRIPT_DIR= os.path.dirname(os.path.abspath(__file__))
RESOURCES_DIR = os.path.join(SCRIPT_DIR, "style")
FOOOCUS_STYLES_DIR = os.path.join(SCRIPT_DIR, "style")
PROMPT_STYLES_DIR = os.path.join(SCRIPT_DIR, "style")
PROMPT_TEST_DIR = os.path.join(SCRIPT_DIR, "style")

BASE_RESOLUTIONS = [
    ("width", "height"),
    (512, 512),
    (512, 768),
    (576, 1024),
    (768, 512),
    (768, 768),
    (768, 1024),
    (768, 1280),
    (768, 1344),
    (768, 1536),
    (816, 1920),
    (832, 1152),
    (832, 1216),
    (896, 1152),
    (896, 1088),
    (1024, 1024),
    (1024, 576),
    (1024, 768),
    (1080, 1920),
    (1440, 2560),
    (1088, 896),
    (1216, 832),
    (1152, 832),
    (1152, 896),
    (1280, 768),
    (1344, 768),
    (1536, 640),
    (1536, 768),
    (1920, 816),
    (1920, 1080),
    (2560, 1440)
]
