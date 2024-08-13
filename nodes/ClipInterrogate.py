import torch
from PIL import Image
import numpy as np
from clip_interrogator import Config, Interrogator as ClipInterrogator

ci = None
BLIP_CAPTIONING_MAX_LENGTH = 32
CURRENT_DEVICE = "cuda:0"

CAPTION_MODEL_NAME = ("blip-base", 
              "blip-large", "blip2-2.7b",
              "blip2-flan-t5-xl", "git-large-coco")

CAPTION_MODEL_NAME = ("blip-base", 
              "blip-large", "blip2-2.7b",
              "blip2-flan-t5-xl", "git-large-coco")

DEVICE_NAME = ("cpu", "cuda:0","cuda:1","cuda:2","cuda:3","cuda:4")
              
CAPTION_MODE = (
    "fast", "best", "classic", "caption", "negative")

CLIP_MODEL_NAME = ("RN50/openai",
                                    "RN50/yfcc15m",
                                    "RN50/cc12m",
                                    "RN50-quickgelu/openai",
                                    "RN50-quickgelu/yfcc15m",
                                    "RN50-quickgelu/cc12m",
                                    "RN101/openai",
                                    "RN101/yfcc15m",
                                    "RN101-quickgelu/openai",
                                    "RN101-quickgelu/yfcc15m",
                                    "RN50x4/openai",
                                    "RN50x16/openai",
                                    "RN50x64/openai",
                                    "ViT-B-32/openai",
                                    "ViT-B-32/laion400m_e31",
                                    "ViT-B-32/laion400m_e32",
                                    "ViT-B-32/laion2b_e16",
                                    "ViT-B-32/laion2b_s34b_b79k",
                                    "ViT-B-32-quickgelu/openai",
                                    "ViT-B-32-quickgelu/laion400m_e31",
                                    "ViT-B-32-quickgelu/laion400m_e32",
                                    "ViT-B-16/openai",
                                    "ViT-B-16/laion400m_e31",
                                    "ViT-B-16/laion400m_e32",
                                    "ViT-B-16-plus-240/laion400m_e31",
                                    "ViT-B-16-plus-240/laion400m_e32",
                                    "ViT-L-14/openai",
                                    "ViT-L/14@336px",
                                    "ViT-L-14/laion400m_e31",
                                    "ViT-L-14/laion400m_e32",
                                    "ViT-L-14/laion2b_s32b_b82k",
                                    "ViT-L-14-336/openai",
                                    "ViT-H-14/laion2b_s32b_b79k",
                                    "ViT-g-14/laion2b_s12b_b42k",
                                    "roberta-ViT-B-32/laion2b_s12b_b32k",
                                    "xlm-roberta-base-ViT-B-32/laion5b_s13b_b90k",
                                    "xlm-roberta-large-ViT-H-14/frozen_laion5b_s13b_b90k")


def tensor_to_pil(img_tensor, batch_index=0):
    # Takes an image in a batch in the form of a tensor of shape [batch_size, channels, height, width]
    # and returns an PIL Image with the corresponding mode deduced by the number of channels

    # Take the image in the batch given by batch_index
    img_tensor = img_tensor[batch_index].unsqueeze(0)
    i = 255. * img_tensor.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8).squeeze())
    return img


def pil_to_tensor(image):
    # Takes a PIL image and returns a tensor of shape [1, height, width, channels]
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image).unsqueeze(0)
    if len(image.shape) == 3:  # If the image is grayscale, add a channel dimension
        image = image.unsqueeze(-1)
    return image


def resize_image_max_size(image, max_size=768, multiple_of=8):
    width, height = image.size
    if width > height:
        max_width = max_size
        max_height = int((height / width) * max_size)
    else:
        max_width = int((width / height) * max_size)
        max_height = max_size

    max_width, max_height = resize_multiple_of((max_width, max_height), multiple_of)
    resized_image = image.resize((max_width, max_height))
    return resized_image

def resize_multiple_of(size, multiple_of=8):
    width, height = size
    new_width = (width // multiple_of) * multiple_of
    new_height = (height // multiple_of) * multiple_of
    return (new_width, new_height)



				
				
def load(clip_model_name='ViT-L-14/openai', caption_model_name='blip-base'):
    global ci
    if ci is None:
        #print(f"Loading CLIP Interrogator {clip_interrogator.__version__}...")
        config = Config(
            device=CURRENT_DEVICE,
            cache_path='models/clip-interrogator',
            clip_model_name=clip_model_name
        )
        config.caption_model_name = caption_model_name
        config.caption_max_length = BLIP_CAPTIONING_MAX_LENGTH
        config.caption_offload = False
        config.clip_offload = False
        # config.chunk_size = 1024
        # config.flavor_intermediate_count = 1024
        ci = ClipInterrogator(config)
    if caption_model_name and caption_model_name != ci.config.caption_model_name:
        ci.config.caption_model_name = caption_model_name
        ci.load_caption_model()
    if clip_model_name != ci.config.clip_model_name:
        ci.config.clip_model_name = clip_model_name
        ci.load_clip_model()


def interrogate(image, mode, caption=None):
    if mode == 'best':
        prompt = ci.interrogate(image, caption=caption)
    elif mode == 'caption':
        prompt = ci.generate_caption(image) if caption is None else caption
    elif mode == 'classic':
        prompt = ci.interrogate_classic(image, caption=caption)
    elif mode == 'fast':
        prompt = ci.interrogate_fast(image, caption=caption)
    elif mode == 'negative':
        prompt = ci.interrogate_negative(image)
    else:
        raise Exception(f"Unknown mode {mode}")
    return prompt


# def unload_interrogate():
#     global ci
#     if ci is not None:
#         print("Offloading CLIP Interrogator...")
#         ci.caption_model = ci.caption_model.to(devices.cpu)
#         ci.clip_model = ci.clip_model.to(devices.cpu)
#         ci.caption_offloaded = True
#         ci.clip_offloaded = True
#         devices.torch_gc()


def image_to_prompt(image, mode, clip_model_name, caption_model_name):
    # try:
    # devices.torch_gc()
    global ci
    if ci is None:
        load(clip_model_name, caption_model_name)
    image = image.convert('RGB')
    prompt = interrogate(image, mode)
    # except torch.cuda.OutOfMemoryError as e:
    #     prompt = "Ran out of VRAM"
    #     print(e)
    # except RuntimeError as e:
    #     prompt = f"Exception {type(e)}"
    #     print(e)

    # unload_interrogate()
    return prompt



class ClipInterrogate:
    """
    A example node

    Class methods
    -------------
    INPUT_TYPES (dict): 
        Tell the main program input parameters of nodes.

    Attributes
    ----------
    RETURN_TYPES (`tuple`): 
        The type of each element in the output tulple.
    RETURN_NAMES (`tuple`):
        Optional: The name of each output in the output tulple.
    FUNCTION (`str`):
        The name of the entry-point method. For example, if `FUNCTION = "execute"` then it will run Example().execute()
    OUTPUT_NODE ([`bool`]):
        If this node is an output node that outputs a result/image from the graph. The SaveImage node is an example.
        The backend iterates on these output nodes and tries to execute all their parents if their parent graph is properly connected.
        Assumed to be False if not present.
    CATEGORY (`str`):
        The category the node should appear in the UI.
    execute(s) -> tuple || None:
        The entry point method. The name of this method must be the same as the value of property `FUNCTION`.
        For example, if `FUNCTION = "execute"` then this method's name must be `execute`, if `FUNCTION = "foo"` then it must be `foo`.
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "caption_mode": (CAPTION_MODE, {"default": "fast"}),
                "clip_model_name": (CLIP_MODEL_NAME, {"default": "ViT-L-14/openai"}),
                "caption_model_name": (CAPTION_MODEL_NAME, {"default": "blip2-2.7b"}),
                # "caption_min_length": ("INT", {"default": 5, "min": 0, "max": 256, "step": 1}),
                "caption_max_length": ("INT", {"default": 32, "min": 0, "max": 256, "step": 1}),
                "sentences_count": ("INT", {"default": 3, "min": 1, "max": 100, "step": 1}),
                "device_name": (DEVICE_NAME, {"default": "cpu"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "interrogate_image"

    # OUTPUT_IS_LIST = (True,)
    # OUTPUT_NODE = True

    CATEGORY = "image"

    def interrogate_image(self, image, caption_mode, clip_model_name, caption_model_name, caption_max_length, sentences_count, device_name):
        global BLIP_CAPTIONING_MAX_LENGTH
        global CURRENT_DEVICE

        CURRENT_DEVICE = device_name
        BLIP_CAPTIONING_MAX_LENGTH = caption_max_length


        img = tensor_to_pil(image)
        prompt_interrogate = image_to_prompt(img.convert('RGB'),
                                caption_mode,
                                clip_model_name=clip_model_name,
                                caption_model_name=caption_model_name)
        sentences = prompt_interrogate.split(", ")
        prompt_interrogate = ", ".join(sentences[:sentences_count])

        print(f"prompt_interrogate - {prompt_interrogate}")
        return {"ui": {"prompt": prompt_interrogate}, "result": (prompt_interrogate,)}


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "ClipInterrogate": ClipInterrogate
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ClipInterrogate": "Clip Interrogate"
}
