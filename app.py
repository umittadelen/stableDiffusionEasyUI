from flask import Flask, render_template, request, send_file, jsonify
import torch, random, os, math, time, threading, sys, subprocess, glob, gc, logging, cv2, json, requests, atexit, signal
from PIL import PngImagePlugin, Image
import numpy as np
from io import BytesIO
import traceback
from diffusers import (
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    StableDiffusionXLPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionXLControlNetPipeline,
    FluxPipeline,
    DiffusionPipeline,
    ControlNetModel,
    AutoencoderKL
)
from transformers import CLIPTokenizer, CLIPModel, CLIPProcessor
from compel import Compel, ReturnedEmbeddingsType, CompelForSDXL, CompelForSD
from diffusers.utils import load_image
from tools.downloadModelFromCivitai import downloadModelFromCivitai
from tools.DepthPro import DepthPro
from tools.BEiTDepthEstimation import BEiTDepthEstimation
from tools.NormalMap import NormalMap
import base64
import queue
from extension_loader import extension_loader, AppAPI

app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

def isDirectory(a):
    return os.path.isdir(a)

def isFile(a):
    return os.path.isfile(a)

def resize_image(image, width, height):
    return image.resize((width, height), resample=Image.Resampling.BICUBIC)

def check_online():
    try:
        # Try to connect to a known server
        response = requests.get("https://www.google.com", timeout=5)
        if response.status_code == 200:
            return True  # Computer is online
    except requests.ConnectionError:
        pass  # Handle if there's no internet connection
    return False  # Computer is offline

torchdtype = torch.float16

gconfig = {
    "generation_stopped":False,
    "generating": False,
    "generated_dir": './generated/',
    "status": "",
    "progress": 0,
    "image_count": 0,
    "custom_seed": -1,
    "remainingImages": 0,
    "image_cache": {},
    "downloading": False,
    "generation_done": False,

    "theme": {"tone_1": "22, 18, 22","tone_2": "42, 32, 42","tone_3": "220, 140, 170"},
    "enable_attention_slicing": True,
    "enable_xformers_memory_efficient_attention": False,
    "enable_model_cpu_offload": True,
    "enable_sequential_cpu_offload": False,
    "use_long_clip": True,
    "update_page_in_background": True,
    "long_clip_model": "zer0int/LongCLIP-GmP-ViT-L-14",
    "fallback_vae_model": "madebyollin/sdxl-vae-fp16-fix",
    "default_clip_model": "openai/clip-vit-base-patch16",
    "fallback_tokenizer_model": "openai/clip-vit-base-patch16",
    "preview_size": "100",
    "update_interwal": "2500",
    "show_latents": False,
    "show_model_preview": True,
    "load_previous_data": True,
    "reset_on_new_request": False,
    "reverse_image_order": False,
    "use_multi_prompt": True,
    "multi_prompt_separator": "§",

    "enable_nsfw_blur": True,
    "nsfw_threshold": 0.3,

    "host":"localhost",
    "port":"8080",

    "SDXL":[
        "SDXL",
        "SDXL 1.0",
        "SDXL Lightning",
        "SDXL Hyper",
        "NoobAI",
        "Illustrious",
        "Pony"
    ],
    "SD 1.5":[
        "SD 1.5"
    ]
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cache for the last loaded pipeline to avoid reloading on every request
_pipeline_cache = {"pipe": None, "key": None}

if isFile("./static/json/settings.json"):
    gconfig.update(json.load(open('./static/json/settings.json', 'r', encoding='utf-8')))

_hf_token_path = os.path.expanduser("~/.cache/huggingface/token")
gconfig["HF_TOKEN"] = (
    open(_hf_token_path, 'r').read().strip()
    if os.path.exists(_hf_token_path) else
    json.load(open('./static/json/settings.json', 'r', encoding='utf-8'))["HF_TOKEN"]
    if isFile("./static/json/settings.json") else
    ""
)

def login_to_huggingface():
    from huggingface_hub import login
    if gconfig["HF_TOKEN"]:
        login(token=gconfig["HF_TOKEN"])
    else:
        login()

if check_online():
    print("Computer is connected to the internet")
    login_to_huggingface()
else:
    print("Computer is offline")

if not isDirectory(gconfig["generated_dir"]):
    os.mkdir(gconfig["generated_dir"])

class controlNets:
    def get_depth_map(image, width, height, colored=False):
        """DepthPro (Apple) — metric depth, colored heatmap or grayscale for ControlNet."""
        return DepthPro(image, colored=colored).resize((width, height), resample=Image.Resampling.BICUBIC)
    def get_depth_map_beit(image, width, height):
        """BEiT depth (Intel DPT-BEiT-Large-512) — grayscale depth map."""
        return BEiTDepthEstimation(image).resize((width, height), resample=Image.Resampling.BICUBIC)
    def get_normal_map(image, width, height):
        return NormalMap(image).resize((width, height), resample=Image.Resampling.BICUBIC)
    def get_canny_image(image, width, height):
        image = np.array(image)

        canny_edges = cv2.Canny(image, 100, 200)
        canny_edges = canny_edges[:, :, None]
        canny_edges = np.concatenate([canny_edges, canny_edges, canny_edges], axis=2)
        return Image.fromarray(canny_edges).resize((width, height), resample=Image.Resampling.BICUBIC)

def load_scheduler(pipe, scheduler_name):
    if   scheduler_name == "DPM++ 2M": pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif scheduler_name == "DPM++ 2M Karras": pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    elif scheduler_name == "DPM++ 2M SDE": pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type="sde-dpmsolver++")
    elif scheduler_name == "DPM++ 2M SDE Karras": pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++")
    elif scheduler_name == "DPM++ SDE": pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config)
    elif scheduler_name == "DPM++ SDE Karras": pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    elif scheduler_name == "DPM2": pipe.scheduler = KDPM2DiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler_name == "DPM2 Karras": pipe.scheduler = KDPM2DiscreteScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    elif scheduler_name == "DPM2 a": pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler_name == "DPM2 a Karras": pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    elif scheduler_name == "Euler": pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler_name == "Euler a": pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler_name == "Heun": pipe.scheduler = HeunDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler_name == "LMS": pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler_name == "LMS Karras": pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    return pipe

def load_pipeline(model_name, model_type, generation_type, scheduler_name, clip_skip):
    gconfig["status"] = "Loading New Pipeline... (loading Pipeline)"
    extension_loader.hooks.fire("before_load_pipeline", model_name=model_name, model_type=model_type)
    #TODO: Set the pipeline

    kwargs = {}

    if "controlnet" in generation_type:
        if "canny" in generation_type and model_type in gconfig["SDXL"]:
            controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torchdtype)
        if "canny" in generation_type and model_type in gconfig["SD 1.5"]:
            controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torchdtype)

        if "depth" in generation_type and model_type in gconfig["SDXL"]:
            controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-depth-sdxl-1.0", torch_dtype=torchdtype)
        if "depth" in generation_type and model_type in gconfig["SD 1.5"]:
            controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torchdtype)

        if "normal" in generation_type and model_type in gconfig["SDXL"]:
            #!controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-normal", torch_dtype=torchdtype)
            raise Exception("Normal Map is not supported for SDXL")
        if "normal" in generation_type and model_type in gconfig["SD 1.5"]:
            controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-normal", torch_dtype=torchdtype)

        kwargs["controlnet"] = controlnet

    if model_type in gconfig["SD 1.5"] and "txt2img" in generation_type:
        kwargs["custom_pipeline"] = "lpw_stable_diffusion"
        if clip_skip != 0:
            kwargs["clip_skip"] = clip_skip
    elif model_type in gconfig["SDXL"] and "txt2img" in generation_type:
        kwargs["custom_pipeline"] = "lpw_stable_diffusion_xl"
        if clip_skip == 0:
            kwargs["clip_skip"] = 2
        else:
            kwargs["clip_skip"] = clip_skip

    if "img2img" in generation_type:
        pipeline = (
            StableDiffusionXLImg2ImgPipeline.from_single_file
            if model_type in gconfig["SDXL"] and model_name.endswith((".ckpt", ".safetensors")) else

            StableDiffusionXLImg2ImgPipeline.from_pretrained
            if model_type in gconfig["SDXL"] else

            StableDiffusionImg2ImgPipeline.from_single_file
            if model_type in gconfig["SD 1.5"] and model_name.endswith((".ckpt", ".safetensors")) else

            StableDiffusionImg2ImgPipeline.from_pretrained
            if model_type in gconfig["SD 1.5"] else

            DiffusionPipeline.from_pretrained
        )
    elif "controlnet" in generation_type:
        pipeline = (
            StableDiffusionXLControlNetPipeline.from_single_file
            if model_type in gconfig["SDXL"] and model_name.endswith((".ckpt", ".safetensors")) else

            StableDiffusionXLControlNetPipeline.from_pretrained
            if model_type in gconfig["SDXL"] else

            StableDiffusionControlNetPipeline.from_single_file
            if model_type in gconfig["SD 1.5"] and model_name.endswith((".ckpt", ".safetensors")) else

            StableDiffusionControlNetPipeline.from_pretrained
            if model_type in gconfig["SD 1.5"] else

            DiffusionPipeline.from_pretrained
        )
    elif "FLUX" in generation_type:
        pipeline = FluxPipeline.from_pretrained
    else:
        pipeline = (
            StableDiffusionXLPipeline.from_single_file
            if model_type in gconfig["SDXL"] and model_name.endswith((".ckpt", ".safetensors")) else

            StableDiffusionXLPipeline.from_pretrained
            if model_type in gconfig["SDXL"] else

            StableDiffusionPipeline.from_single_file
            if model_type in gconfig["SD 1.5"] and model_name.endswith((".ckpt", ".safetensors")) else

            StableDiffusionPipeline.from_pretrained
            if model_type in gconfig["SD 1.5"] else

            DiffusionPipeline.from_pretrained
        )
    print("Pipeline class set to:", pipeline.__name__)

    gconfig["status"] = "Loading New Pipeline... (Pipeline loaded)"

    gconfig["status"] = "Loading New Pipeline... (pipe)"
    #TODO: Load the pipeline

    pipe = pipeline(
        model_name,
        torch_dtype=torchdtype,
        use_safetensors=True,
        add_watermarker=False,
        token=gconfig["HF_TOKEN"],
        safety_checker=None,
        **kwargs
    )

    gconfig["status"] = "Loading New Pipeline... (loading VAE)"
    if gconfig["use_long_clip"]:
        print(gconfig["long_clip_model"])
        clip_model = CLIPModel.from_pretrained(gconfig["long_clip_model"], dtype=torchdtype)
        clip_processor = CLIPProcessor.from_pretrained(gconfig["long_clip_model"], use_fast=True)
        print("max token limit:", clip_processor.tokenizer.model_max_length)
    else:
        print(gconfig["default_clip_model"])
        clip_model = CLIPModel.from_pretrained(gconfig["default_clip_model"], dtype=torchdtype)
        clip_processor = CLIPProcessor.from_pretrained(gconfig["default_clip_model"], use_fast=True)
        print("max token limit:", clip_processor.tokenizer.model_max_length)

    pipe.clip_model = clip_model
    pipe.clip_processor = clip_processor

    #TODO: Load the VAE model
    if not hasattr(pipe, "vae") or pipe.vae is None:
        print("Model does not include a VAE. Loading external VAE...")
        gconfig["status"] = "Model does not include a VAE. Loading external VAE..."
        vae = AutoencoderKL.from_pretrained(
            gconfig["fallback_vae_model"],
            torch_dtype=torchdtype,
        )
        pipe.vae = vae
        gconfig["status"] = "External VAE loaded."
    else:
        print("Model includes a VAE. Skipping external VAE loading...")
        gconfig["status"] = "Model includes a VAE. Skipping external VAE loading..."
    
    if not hasattr(pipe, "tokenizer") or pipe.tokenizer is None:
        print("Model does not include a tokenizer. Loading external tokenizer...")
        gconfig["status"] = "Model does not include a tokenizer. Loading external tokenizer..."
        tokenizer = CLIPTokenizer.from_pretrained(
            gconfig["fallback_tokenizer_model"],
        )
        pipe.tokenizer = tokenizer
        gconfig["status"] = "External tokenizer loaded."
    else:
        print("Model includes a tokenizer. Skipping external tokenizer loading...")
        gconfig["status"] = "Model includes a tokenizer. Skipping external tokenizer loading..."

    gconfig["status"] = "Loading New Pipeline... (VAE loaded)"

    if scheduler_name != "None":
        pipe = load_scheduler(pipe, scheduler_name)
    gconfig["status"] = "Loading New Pipeline... (pipe loaded)"

    pipe.to(device)

    if gconfig["enable_attention_slicing"]:
        pipe.enable_attention_slicing()
    if gconfig["enable_xformers_memory_efficient_attention"]:
        pipe.enable_xformers_memory_efficient_attention()
    if gconfig["enable_model_cpu_offload"]:
        pipe.enable_model_cpu_offload()
    if gconfig["enable_sequential_cpu_offload"]:
        pipe.enable_sequential_cpu_offload()

    gconfig["status"] = "Pipeline Loaded..."
    print("Pipeline Loaded...")
    extension_loader.hooks.fire("after_load_pipeline", pipe=pipe, model_name=model_name, model_type=model_type)
    return pipe

def latents_to_img(latents, pipe) -> Image:
    # Make sure latents are float
    latents = latents.float()

    # Try to get the model's latent_rgb_factors from the VAE
    # If not available, fall back to manual weights
    try:
        factors = pipe.vae.latent_format.latent_rgb_factors  # shape: [4,3] maybe
        biases = pipe.vae.latent_format.latent_rgb_biases   # optional, fallback if available
        if factors is None:
            raise AttributeError("No latent_rgb_factors in model")
        
        # Convert to torch tensors
        weights_tensor = torch.tensor(factors, dtype=latents.dtype, device=latents.device).t()
        if biases is not None:
            biases_tensor = torch.tensor(biases, dtype=latents.dtype, device=latents.device)
        else:
            biases_tensor = torch.zeros(3, dtype=latents.dtype, device=latents.device)

        rgb_tensor = torch.einsum("...lxy,lr -> ...rxy", latents, weights_tensor) + biases_tensor.unsqueeze(-1).unsqueeze(-1)

    except Exception:
        # fallback to manual weights
        weights = (
            (60, -60, 25, -70),
            (60,  -5, 15, -50),
            (60,  10, -5, -35),
        )
        weights_tensor = torch.t(torch.tensor(weights, dtype=latents.dtype).to(latents.device))
        biases_tensor = torch.tensor((150, 140, 130), dtype=latents.dtype).to(latents.device)
        rgb_tensor = torch.einsum("...lxy,lr -> ...rxy", latents, weights_tensor) + biases_tensor.unsqueeze(-1).unsqueeze(-1)

    # Clamp and convert to uint8
    image_array = rgb_tensor.clamp(0, 255).byte().cpu().numpy().transpose(1, 2, 0)
    return Image.fromarray(image_array)


# --- Latents saving queue and worker ---

latents_save_queue = queue.Queue()

def latents_saver_worker():
    while True:
        item = latents_save_queue.get()
        if item is None:
            break  # Shutdown signal
        latents, pipe, image_path, seed = item
        try:
            latents = latents.detach().cpu()
            image = latents_to_img(latents, pipe)
            if not gconfig["generation_stopped"] and not gconfig["generation_done"]:
                image.save(image_path, 'PNG')
                # Do NOT run NSFW check for latents; store None for nsfw_score
                gconfig["image_cache"][seed] = [image_path, None]
        except Exception:
            pass
        finally:
            latents_save_queue.task_done()

# Start the background worker thread (only once)
if not hasattr(globals(), "_latents_saver_thread"):
    _latents_saver_thread = threading.Thread(target=latents_saver_worker, daemon=True)
    _latents_saver_thread.start()

def save_latents_image(latents, pipe, image_path, seed):
    # Just enqueue the save request, don't block
    latents_save_queue.put((latents, pipe, image_path, seed))

def image_to_base64(img, temp_file=f"{gconfig['generated_dir']}temp_base64_image.png"):
    img = img.convert("RGB")
    img.save(temp_file, format="PNG")
    with open(temp_file, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode()
    os.remove(temp_file)
    return "data:image/png;base64," + img_base64

def generateImage(pipe, model:str, prompt:str, original_prompt:str, style_prompt:str, negative_prompt:str, seed, width, height, img_input, use_orig_img, strength, model_type, generation_type, image_size, cfg_scale, samplingSteps, scheduler_name, image_count, prompt_count):
    
    prompt = prompt.rstrip(",") + "," + style_prompt.lstrip(",") if prompt and style_prompt else prompt + style_prompt
    
    current_time = time.time()
    gconfig["generation_done"] = False

    if not isDirectory(gconfig["generated_dir"]):
        os.makedirs(gconfig["generated_dir"])

    def progress(pipe, step_index, timestep, callback_kwargs):
        gconfig["status"] = int(math.floor(step_index / samplingSteps * 100))
        gconfig["progress"] = int(math.floor(((image_count * prompt_count - gconfig["remainingImages"]) + (step_index / samplingSteps)) / (image_count * prompt_count) * 100))

        if gconfig["show_latents"]:
            image_path = os.path.join(gconfig["generated_dir"], f'image{current_time}_{seed}.png')
            # Enqueue the latents for saving (non-blocking)
            save_latents_image(callback_kwargs["latents"][0], pipe, image_path, seed)

        if gconfig["generation_stopped"]:
            gconfig["status"] = "Generation Stopped"
            gconfig["progress"] = 0
            gconfig["generating"] = False
            raise Exception("Generation Stopped")

        return callback_kwargs

    gconfig["status"] = "Generating Image..."
    kwargs = {}

    try:
        #! Pass the parameters to the pipeline - (default kwargs for all pipelines)
        kwargs["generator"] = torch.manual_seed(seed)
        kwargs["guidance_scale"] = cfg_scale
        kwargs["num_inference_steps"] = samplingSteps
        kwargs["callback_on_step_end"] = progress
        kwargs["num_images_per_prompt"] = 1

        if "controlnet" in generation_type:
            if img_input:
                print("Loading and preprocessing the input image for ControlNet...")
                try:
                    image = load_image(img_input).convert("RGB")
                    if image_size == "resize":
                        image = resize_image(image, width, height)

                except Exception:
                    #TODO: If the image is not valid, return False
                    gconfig["status"] = "Image Invalid"
                    traceback_details = traceback.format_exc()
                    print(f"Cannot acces to image:{traceback_details}")
                    return False

                if use_orig_img == "false":
                    if "canny" in generation_type:
                        new_image = controlNets.get_canny_image(image, width, height)
                    if "depth_pro" in generation_type:
                        new_image = controlNets.get_depth_map(image, width, height)
                    elif "depth_beit" in generation_type:
                        new_image = controlNets.get_depth_map_beit(image, width, height)
                    if "normal" in generation_type:
                        new_image = controlNets.get_normal_map(image, width, height)
                    else:
                        new_image = image
                else:
                    new_image = image

                #! Pass the image to pipeline - (kwargs for controlnet)
                kwargs["image"] = new_image
                kwargs["controlnet_conditioning_scale"] = float(strength)
            else:
                gconfig["status"] = "Image Not Provided"
                raise Exception("Image Not Provided")
        elif "img2img" in generation_type:
            if img_input:
                # Load and preprocess the image for img2img
                image = load_image(img_input).convert("RGB")
                if image_size == "resize":
                    image = resize_image(image, width, height)

                #! Pass the image to pipeline - (kwargs for img2img)
                kwargs["image"] = image
                kwargs["strength"] = 1.0 - strength
        else:
            #! Pass the parameters to the pipeline - (kwargs for txt2img)
            kwargs["width"] = width
            kwargs["height"] = height
        
        def pad_embeddings(embeds, target_length):
            current_length = embeds.shape[1]
            if current_length < target_length:
                padding = torch.zeros((embeds.shape[0], target_length - current_length, embeds.shape[2]), dtype=embeds.dtype, device=embeds.device)
                embeds = torch.cat([embeds, padding], dim=1)
            return embeds

        if not gconfig["enable_sequential_cpu_offload"]:
            if hasattr(pipe, "tokenizer_2"):
                compel = CompelForSDXL(pipe=pipe, device=device)
                result = compel(prompt, negative_prompt=negative_prompt)
                kwargs["prompt_embeds"] = result.embeds
                kwargs["pooled_prompt_embeds"] = result.pooled_embeds
                kwargs["negative_prompt_embeds"] = result.negative_embeds
                kwargs["negative_pooled_prompt_embeds"] = result.negative_pooled_embeds
            else:
                compel = CompelForSD(pipe=pipe, device=device)
                result = compel(prompt, negative_prompt=negative_prompt)
                kwargs["prompt_embeds"] = result.embeds
                kwargs["negative_prompt_embeds"] = result.negative_embeds
        else:
            kwargs["prompt"] = prompt
            kwargs["negative_prompt"] = negative_prompt

        try:
            with torch.no_grad():
                image = pipe(
                    **kwargs
                ).images[0]
        except Exception:
            if gconfig["generation_stopped"]:
                print("Generation Stopped", flush=True)
            else:
                traceback_details = traceback.format_exc()
                gconfig["status"] = f"Generation Stopped with reason:<br>{traceback_details}"
                print(f"Generation Stopped with reason:\n{traceback_details}", flush=True)
            gconfig["generation_stopped"] = True
            gconfig["generating"] = False
            return False

        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("Prompt", prompt)
        metadata.add_text("StylePrompt", style_prompt)
        metadata.add_text("OriginalPrompt", original_prompt)
        metadata.add_text("NegativePrompt", negative_prompt)
        metadata.add_text("Width", str(width))
        metadata.add_text("Height", str(height))
        metadata.add_text("CFGScale", str(cfg_scale))
        metadata.add_text("Strength", str(strength) if "img2img" in model_type else "N/A")
        metadata.add_text("Seed", str(seed))
        metadata.add_text("SamplingSteps", str(samplingSteps))
        metadata.add_text("Model", str(model))
        metadata.add_text("Scheduler", str(scheduler_name))
        # Use load_image for URLs, Image.open for local files
        img_for_meta = (
                (
                    load_image(img_input) 
                    if isinstance(img_input, str) and img_input.startswith(("http://","https://"))
                    else Image.open(img_input)
                ).convert("RGB") 
                if img_input
                else None
            )
        metadata.add_text("ImgInput", str(image_to_base64(img_for_meta)) if img_for_meta else "N/A")
        metadata.add_text("ImgInputMetadata", json.dumps(getattr(img_for_meta, "info", {})) if img_for_meta else "N/A")

        #TODO: Save the image to the temporary directory
        image_path = os.path.join(gconfig["generated_dir"], f'image{current_time}_{seed}.png')
        if not gconfig["generation_stopped"]:
            gconfig["generation_done"] = True
            image.save(image_path, 'PNG', pnginfo=metadata)

        def _run_nsfw(path, img):
            try:
                from tools.NSFWClassifier import score as nsfw_score
                nsfw = nsfw_score(img)
                meta = PngImagePlugin.PngInfo()
                reloaded = Image.open(path)
                for k, v in reloaded.info.items():
                    if k == "NSFWScoreWD":
                        continue
                    if isinstance(v, str):
                        try:
                            meta.add_text(k, v)
                        except Exception:
                            pass
                meta.add_text("NSFWScoreWD", str(nsfw))
                reloaded.save(path, "PNG", pnginfo=meta)
                # update cache entry with score
                if seed in gconfig["image_cache"]:
                    gconfig["image_cache"][seed] = [path, nsfw]
            except Exception:
                pass

        if gconfig["enable_nsfw_blur"] and not gconfig["generation_stopped"]:
            threading.Thread(target=_run_nsfw, args=(image_path, image.copy()), daemon=True).start()

        gconfig["status"] = "DONE"
        gconfig["progress"] = 0

        return image_path

    except Exception:
        if gconfig["generation_stopped"]:
            print("Generation Stopped", flush=True)
        else:
            traceback_details = traceback.format_exc()
            gconfig["status"] = f"Generation Stopped with reason:<br>{traceback_details}"
            gconfig["generation_stopped"] = True
            gconfig["generating"] = False
            print(f"Generation Stopped with reason:\n{traceback_details}", flush=True)
        return False

@app.route('/generate', methods=['POST'])
def generate():
    #TODO: Check if generation is already in progress
    if gconfig["generating"] or gconfig["downloading"]:
        return jsonify(status='Image generation already in progress'), 400

    gconfig["generating"] = True
    if gconfig["reset_on_new_request"]:
        gconfig["image_cache"] = {}
    gconfig["status"] = "Starting Image Generation..."

    #TODO: Get parameters from the request
    model_name = request.form.get('model', 'https://huggingface.co/cagliostrolab/animagine-xl-3.1/blob/main/animagine-xl-3.1.safetensors')
    model_type = request.form.get('model_type', 'SDXL')
    scheduler_name = request.form.get('scheduler', 'Euler a')
    prompts = request.form.get('prompt', '').strip()
    style_prompt = request.form.get('style', '').strip()
    negative_prompt = request.form.get('negative_prompt', 'default_negative_prompt')
    width = int(request.form.get('width', 832))
    height = int(request.form.get('height', 1216))
    strength = float(request.form.get('strength', 0.5))
    img_input_link = request.form.get('img_input_link', None)
    img_input_img = request.files.get('img_input_img', None)
    use_orig_img = request.form.get('use_orig_img', "false")
    generation_type = request.form.get('generation_type', 'txt2img')
    image_size = request.form.get('image_size', 'original')
    cfg_scale = float(request.form.get('cfg_scale', 7))
    image_count = int(request.form.get('image_count', 4))
    custom_seed = int(request.form.get('custom_seed', gconfig["custom_seed"]))
    samplingSteps = int(request.form.get('sampling_steps', 28))
    clip_skip = int(request.form.get('clip_skip', 0))

    if custom_seed != gconfig["custom_seed"]:
        image_count = 1

    #save the temp image if provided and if not txt2img
    if img_input_img and generation_type != "txt2img":
        temp_image = Image.open(img_input_img).convert("RGB")
        png_info = PngImagePlugin.PngInfo()
        for k, v in temp_image.info.items():
            png_info.add_text(k, str(v))
        temp_image.save(f"{gconfig['generated_dir']}temp_image.png", pnginfo=png_info)

    if img_input_link and generation_type != "txt2img":
        img_input_link = Image.open(requests.get(img_input_link, stream=True).raw)
        png_info = PngImagePlugin.PngInfo()
        for k, v in img_input_link.info.items():
            png_info.add_text(k, str(v))
        img_input_link.save(f"{gconfig['generated_dir']}temp_image.png", pnginfo=png_info)

    img_input = f"{gconfig['generated_dir']}temp_image.png" if img_input_img else img_input_link

    if prompts == "":
        gconfig["status"] = "No Prompt Provided"
        gconfig["generating"] = False
        return jsonify(status='No prompt provided'), 400

    #TODO: Function to generate images
    def generate_images():
        global _pipeline_cache
        try:
            user_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_name.lstrip("./")).replace("\\", "/")
            cache_key = (user_model_path, model_type, generation_type, scheduler_name, clip_skip)
            if _pipeline_cache["pipe"] is not None and _pipeline_cache["key"] == cache_key:
                print("Reusing cached pipeline...")
                gconfig["status"] = "Reusing cached pipeline..."
                pipe = _pipeline_cache["pipe"]
            else:
                # Unload old pipeline before loading new one
                if _pipeline_cache["pipe"] is not None:
                    del _pipeline_cache["pipe"]
                    _pipeline_cache["pipe"] = None
                    gc.collect()
                    torch.cuda.empty_cache()
                pipe = load_pipeline(user_model_path, model_type, generation_type, scheduler_name, clip_skip)
                _pipeline_cache["pipe"] = pipe
                _pipeline_cache["key"] = cache_key
        except Exception:
            traceback_details = traceback.format_exc()
            gconfig["generating"] = False
            gconfig["status"] = f"Error Loading Model...<br>{traceback_details}"
            print(f"Error Loading Model...\n{traceback_details}")
            gconfig["progress"] = 0
            return

        try:
            if gconfig["use_multi_prompt"]:
                if gconfig.get("multi_prompt_separator"):
                    prompt_list = [p.strip() for p in prompts.split(gconfig["multi_prompt_separator"])]
                else:
                    # Handle the case when the separator is empty or missing
                    prompt_list = [prompts.strip()]

            extension_loader.hooks.fire(
                "before_generate",
                prompts=prompts,
                image_count=image_count,
                width=width,
                height=height,
                model_name=model_name,
            )

            for prompt_index, prompt in enumerate(prompt_list):

                if prompt == "":
                    raise Exception("No Prompt Provided")

                for i in range(image_count):
                    if gconfig["generation_stopped"]:
                        gconfig["progress"] = 0
                        gconfig["status"] = "Generation Stopped"
                        gconfig["generating"] = False
                        gconfig["generation_stopped"] = False
                        raise Exception("Generation Stopped")

                    #TODO: Update the progress message
                    gconfig["remainingImages"] = (image_count * len(prompt_list)) - (prompt_index * image_count + i)
                    gconfig["status"] = f"Generating {gconfig['remainingImages'] * len(prompt_list)} Images..."
                    gconfig["progress"] = 0

                    #TODO: Generate a new seed for each image
                    if custom_seed == gconfig["custom_seed"]:
                        seed = random.randint(0, 100000000000)
                    else:
                        seed = custom_seed

                    image_path = generateImage(pipe, model_name, prompt, prompts, style_prompt, negative_prompt, seed, width, height, img_input, use_orig_img, strength, model_type, generation_type, image_size, cfg_scale, samplingSteps, scheduler_name, image_count, len(prompt_list))

                    #TODO: Store the generated image path
                    if image_path:
                        gconfig["image_cache"][seed] = [image_path, None]
                        extension_loader.hooks.fire(
                            "after_generate",
                            image_path=image_path,
                            seed=seed,
                            prompt=prompt,
                        )

            gconfig["status"] = "Generation Complete"
        except Exception as exc:
            if "Generation Stopped" not in str(exc):
                traceback_details = traceback.format_exc()
                gconfig["status"] = f"Error Generating Images...<br>{traceback_details}"
                print(f"Error Generating Images...\n{traceback_details}")
            gconfig["generating"] = False
            gconfig["generation_stopped"] = False
            gconfig["progress"] = 0

        finally:
            # Don't delete pipe here — it's kept in cache for reuse
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gconfig["progress"] = 0
            gconfig["generating"] = False
            gconfig["generation_stopped"] = False

    #TODO: Start image generation in a separate thread to avoid blocking
    threading.Thread(target=generate_images).start()
    return jsonify(status='Image generation started', count=image_count)

@app.route('/extensions', methods=['GET'])
def list_extensions():
    return jsonify(extension_loader.get_info())

@app.route('/save_prompt', methods=['POST'])
def save_prompt():
    file_path = './static/json/saved_prompts.json'
    if isFile(file_path):
        try:
            with open(file_path, 'r') as f:
                prompts = json.load(f)
        except json.JSONDecodeError:
            prompts = {}
    else:
        prompts = {}

    prompts[time.time()] = {'prompt': request.form['prompt'], 'negative_prompt': request.form['negative_prompt']}

    with open(file_path, 'w') as f:
        json.dump(prompts, f, indent=4)

    return jsonify(status='Prompt saved')

@app.route('/addmodel', methods=['POST'])
def addmodel():
    #TODO: Download the model
    model_id = int(request.form['model_id']) if request.form['model_id'].isdigit() else 0
    version_id = int(request.form['version_id']) if request.form['version_id'].isdigit() else 0

    if model_id == 0 or version_id == 0:
        return jsonify(status='Invalid Model ID or Version ID')

    gconfig["status"] = "Downloading Model..."

    if gconfig["generating"]:
        return jsonify(status='Image generation in progress. Please wait'), 400

    #! civitai.com
    try:
        gconfig["downloading"] = True
        downloadModelFromCivitai(model_id, version_id)
        gconfig["downloading"] = False
        return jsonify(status='Model Downloaded')
    except:
        gconfig["downloading"] = False
        return jsonify(status='Error Downloading Model')

@app.route('/serve_controlnet', methods=['POST'])
def serve_controlnet():
    file = request.files.get('imageUpload', "")
    controlnet_type = request.form.get('type_select', "")
    if not file:
        return 'No file provided', 400

    # Convert the uploaded file to a NumPy array
    img = Image.open(file)
    width, height = img.width, img.height

    # Convert the result to a PIL Image
    if "canny" in controlnet_type:
        edges_image = controlNets.get_canny_image(img, width, height)
    elif "depth_pro" in controlnet_type:
        edges_image = controlNets.get_depth_map(img, width, height, colored=True)
    elif "depth_beit" in controlnet_type:
        edges_image = controlNets.get_depth_map_beit(img, width, height)
    elif "normal" in controlnet_type:
        edges_image = controlNets.get_normal_map(img, width, height)
    else:
        edges_image = controlNets.get_canny_image(img, width, height)

    # Save the result to a byte buffer
    buf = BytesIO()
    edges_image.save(buf, format='PNG')
    buf.seek(0)

    return send_file(buf, mimetype='image/png')

@app.route('/controlnet')
def controlnet():
    return render_template('controlnet_preview.html')


@app.route('/status', methods=['GET'])
def status():
    # Convert the generated images to a list to send to the client
    images = []
    for seed, path in gconfig["image_cache"].items():
        img_path = path[0] if len(path) > 0 else None
        nsfw_score = path[1] if len(path) > 1 else None
        images.append({
            'img': img_path,
            'seed': seed,
            'nsfw_score': nsfw_score
        })
    return jsonify(
        images=images,
        gconfig=gconfig
    )

@app.route('/generated/<filename>', methods=['GET'])
def serve_image(filename):
    image_path = os.path.join(gconfig["generated_dir"], filename)
    resize_image = request.args.get('r') == '1' and gconfig.get("preview_size", "100") != "100"

    if not os.path.exists(image_path):
        with open("./static/json/placeholders.json", "r") as f:
            placeholders = json.load(f)
        placeholder_text = random.choice(placeholders)
        return jsonify(status='Image not found', image=placeholder_text)

    if resize_image:
        image_size_percentage = int(gconfig.get("preview_size", "100"))
        image = Image.open(image_path)

        # Calculate new dimensions while keeping the aspect ratio
        width, height = image.size
        new_width = int(width * (image_size_percentage / 100))
        new_height = int(height * (image_size_percentage / 100))
        image = image.resize((new_width, new_height))

        img_io = BytesIO()
        image.save(img_io, 'PNG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png')
    else:
        return send_file(image_path, mimetype='image/png')

@app.route('/image/<filename>', methods=['GET'])
def image(filename):
    return render_template('image.html', image=filename)

@app.route('/stop', methods=['POST'])
def stop_generation():
    gconfig["generation_stopped"] = True
    return jsonify(status='Image generation stopped')

@app.route('/clear', methods=['POST'])
def clear_images():
    gconfig["image_cache"] = {}
    files = glob.glob(os.path.join(gconfig["generated_dir"], '*'))
    
    for file in files:
        for _ in range(10):
            try:
                os.remove(file)
                break
            except PermissionError:
                time.sleep(0.2)
            except Exception:
                break

    return jsonify(status='Images cleared')

@app.route('/restart', methods=['POST'])
def restart_app():
    global _pipeline_cache
    if _pipeline_cache["pipe"] is not None:
        del _pipeline_cache["pipe"]
        _pipeline_cache = {"pipe": None, "key": None}
        gc.collect()
        torch.cuda.empty_cache()
    gconfig["progress"] = 0
    subprocess.Popen([sys.executable] + sys.argv)
    os._exit(0)

@app.route('/')
def index():
    return render_template('index.html')

def get_clip_token_info(text):
    clip_tokenizer = CLIPTokenizer.from_pretrained(gconfig["default_clip_model"] if not gconfig["use_long_clip"] else gconfig["long_clip_model"])
    # Get tokens and their IDs
    clip_tokens = clip_tokenizer.encode(text, add_special_tokens=True)
    
    # Decode individual tokens for display
    clip_decoded = []
    for token in clip_tokens:
        decoded = clip_tokenizer.decode([token], skip_special_tokens=False)
        if decoded.isspace():
            decoded = "␣"
        elif decoded == "":
            decoded = "∅"
        clip_decoded.append(decoded)
    
    # Return the dictionary
    return {
        "CLIPTokenCount": len(clip_tokens),
        "MaxTokens": clip_tokenizer.model_max_length,
        "Tokens": clip_decoded
    }

@app.route('/clip_token_count', methods=['POST'])
def clip_token_count():
    text = request.form['text']
    result = get_clip_token_info(text)
    return jsonify(result)

@app.route('/save_form_data', methods=['POST'])
def save_form_data():
    form_data = request.get_json()
    with open('./static/json/form_data.json', 'w', encoding='utf-8') as f:
        json.dump(form_data, f, indent=4)
    return jsonify(status='Form data saved')

@app.route('/load_form_data', methods=['GET'])
def load_form_data():
    if gconfig["load_previous_data"]:
        if not isFile('./static/json/form_data.json'):
            with open('./static/json/form_data.json', 'w', encoding='utf-8') as f:
                json.dump({}, f, indent=4)
            return jsonify({})
        with open('./static/json/form_data.json', 'r', encoding='utf-8') as f:
            form_data = json.load(f)
        return jsonify(form_data)
    else:
        return jsonify({})

@app.route('/reset_form_data', methods=['GET'])
def reset_form_data():
    with open('./static/json/form_data.json', 'w', encoding='utf-8') as f:
        json.dump({}, f, indent=4)
    return jsonify(status='Form data reset')

@app.route('/clip_token')
def clip_token():
    return render_template('clip_token_count.html')

@app.route('/models')
def models():
    return render_template('models.html')

@app.route('/settings')
def settings():
    return render_template('settings.html')

@app.route('/save_settings', methods=['POST'])
def save_settings():
    settings = request.get_json()
    with open('./static/json/settings.json', 'w', encoding='utf-8') as f:
        json.dump(settings, f, indent=4)
    gconfig.update(settings)
    return jsonify(status='Settings saved')

@app.route('/load_settings', methods=['GET'])
def load_settings():
    if not isFile('./static/json/settings.json'):
        with open('./static/json/settings.json', 'w', encoding='utf-8') as f:
            json.dump({}, f, indent=4)
    with open('./static/json/settings.json', 'r', encoding='utf-8') as f:
        settings = json.load(f)
    gconfig.update(settings)
    # Return full gconfig so all defaults are present, minus internal/sensitive keys
    excluded = {
        "HF_TOKEN", "image_cache", "status", "progress", "generating",
        "generation_stopped", "generation_done", "downloading", "remainingImages",
        "image_count", "custom_seed", "SDXL", "SD 1.5", "generated_dir"
    }
    return jsonify({k: v for k, v in gconfig.items() if k not in excluded})

@app.route('/metadata')
def metadata():
    return render_template('metadata.html')

@app.route('/nsfw_check/<path:filename>', methods=['GET'])
def nsfw_check(filename):
    # Prevent path traversal — only allow bare filenames
    filename = os.path.basename(filename)
    image_path = os.path.join(gconfig["generated_dir"], filename)

    if not os.path.exists(image_path):
        return jsonify(error='Image not found'), 404

    # Return cached score if already computed
    try:
        img = Image.open(image_path)
        if "NSFWScoreWD" in img.info:
            try:
                return jsonify(score=float(img.info["NSFWScoreWD"]))
            except (ValueError, TypeError):
                pass
    except Exception:
        return jsonify(score=0.0)

    # Run classifier (lazy-loads model on first call) — stores raw score, no threshold
    from tools.NSFWClassifier import score as nsfw_score
    nsfw = nsfw_score(img)

    # Write score into PNG metadata
    try:
        meta = PngImagePlugin.PngInfo()
        for k, v in img.info.items():
            if k in ("NSFWScoreWD", "NSFWScore", "NSFWRating", "NSFWThreshold"):
                continue
            if isinstance(v, str):
                try:
                    meta.add_text(k, v)
                except Exception:
                    pass
        meta.add_text("NSFWScoreWD", str(nsfw))
        img.save(image_path, "PNG", pnginfo=meta)
    except Exception:
        pass

    return jsonify(score=nsfw)

def get_model_configs():
    """Scans the models directory and returns a list of model configurations."""
    models_path = "./models/"
    merged_config = []

    if not os.path.exists(models_path):
        return {"error": "Model directory not found"}

    for model_name in os.listdir(models_path):
        model_dir = os.path.join(models_path, model_name)
        if os.path.isdir(model_dir):
            json_files = [f for f in os.listdir(model_dir) if f.endswith(".json")]
            if json_files:
                json_path = os.path.join(model_dir, json_files[0])  # Pick the first JSON file
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        config_data = json.load(f)
                    merged_config.append(config_data)
                except json.JSONDecodeError:
                    return {"error": f"Invalid JSON in {json_path}"}

    return merged_config  # Returns a list of JSON data

@app.route('/scan_model_configs')
def scan_for_model_configs():
    """Flask route to get model configurations as a JSON response."""
    result = get_model_configs()

    if "error" in result:
        return jsonify(result), 404 if result["error"] == "Model directory not found" else 400

    return jsonify(result)

@app.route('/save_model_configs', methods=['POST'])
def save_model_configs():
    json_data = request.get_json()
    import json
    for item in json_data:
        model_config_path = item["path"]+".json"
        with open(model_config_path, "w", encoding="utf-8") as f:
            json.dump(item, f, indent=4)
    return jsonify({"message": "JSON saved successfully!"}), 200

@app.route('/delete_model', methods=['POST'])
def delete_model():
    model_name = request.form['model_name']
    if not model_name.endswith(".safetensors"):
        model_name += ".safetensors"

    models_data = get_model_configs()
    for item in models_data:
        if item["name"] == model_name:
            model_path = os.path.dirname(item["path"])
            try:
                for root, dirs, files in os.walk(model_path, topdown=False):
                    for name in files:
                        os.remove(os.path.join(root, name))
                    for name in dirs:
                        try:
                            os.rmdir(os.path.join(root, name))
                        except OSError as e:
                            print(f"Error removing directory {name}: {e}")

                try:
                    os.rmdir(model_path)
                except OSError as e:
                    print(f"Error removing top-level directory {model_path}: {e}")
                    return jsonify({"error": "Directory not empty or permissions issue."}), 500

                return jsonify({"message": "Model deleted successfully!"}), 200
            except Exception as e:
                return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Model not found"}), 404

def _cleanup():
    """Deep cleanup of GPU/CPU memory on server shutdown."""
    global _pipeline_cache
    print("[cleanup] Shutting down — freeing memory...")

    # Stop any in-progress generation
    gconfig["generation_stopped"] = True
    gconfig["generating"] = False

    # Unload cached pipeline and all sub-models
    pipe = _pipeline_cache.get("pipe")
    if pipe is not None:
        try:
            for attr in ("text_encoder", "text_encoder_2", "unet", "vae",
                         "tokenizer", "tokenizer_2", "clip_model", "clip_processor",
                         "controlnet", "scheduler", "feature_extractor"):
                if hasattr(pipe, attr):
                    delattr(pipe, attr)
        except Exception:
            pass
        del pipe
        _pipeline_cache["pipe"] = None
        _pipeline_cache["key"] = None

    # Full Python GC passes
    for _ in range(3):
        gc.collect()

    # CUDA deep clean
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
        # Reset all CUDA memory stats
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)
            torch.cuda.reset_accumulated_memory_stats(i)
        # Another GC pass after CUDA clean
        gc.collect()
        torch.cuda.empty_cache()

    print("[cleanup] Memory freed.")

atexit.register(_cleanup)

# Also catch Ctrl+C and SIGTERM (e.g. from task manager / systemd)
def _signal_handler(sig, frame):
    _cleanup()
    sys.exit(0)

signal.signal(signal.SIGTERM, _signal_handler)
try:
    signal.signal(signal.SIGINT, _signal_handler)
except OSError:
    pass  # SIGINT may not be settable in some environments

if __name__ == '__main__':
    _api = AppAPI(
        device=device,
        dtype=torchdtype,
        pipeline_cache=_pipeline_cache,
        load_pipeline=load_pipeline,
        load_scheduler=load_scheduler,
        generateImage=generateImage,
        resize_image=resize_image,
        image_to_base64=image_to_base64,
        latents_to_img=latents_to_img,
        get_model_configs=get_model_configs,
        controlNets=controlNets,
    )
    extension_loader.load_all(app, gconfig, _api)
    extension_loader.hooks.fire("on_app_start", app=app, gconfig=gconfig)
    app.run(host=gconfig["host"], port=int(gconfig["port"]), debug=False)
    gconfig["status"] = "Server Started"
