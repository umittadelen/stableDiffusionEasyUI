# import the required libraries
from auto_installer import install_requirements
install_requirements()

from flask import Flask, render_template, request, send_file, jsonify
import torch, random, os, math, time, threading, sys, subprocess, glob, gc, logging, cv2, json
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
from transformers import pipeline as ppln
from compel import Compel, ReturnedEmbeddingsType
from diffusers.utils import load_image
from downloadModelFromCivitai import downloadModelFromCivitai
import base64

app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

def isDirectory(a):
    return os.path.isdir(a)

def isFile(a):
    return os.path.isfile(a)

def resize_image(image, width, height):
    return image.resize((width, height), resample=Image.BICUBIC)

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

    "theme": {"tone_1": "240, 240, 240","tone_2": "240, 218, 218","tone_3": "240, 163, 163"},
    "enable_attention_slicing": True,
    "enable_xformers_memory_efficient_attention": False,
    "enable_model_cpu_offload": True,
    "enable_sequential_cpu_offload": False,
    "use_long_clip": True,
    "show_latents": False,
    "load_previous_data": True,
    "use_multi_prompt": True,
    "multi_prompt_separator": "§",

    "SDXL":[
        "SDXL",
        "SDXL Lightning",
        "SDXL Hyper",
        "Illustrious",
        "Pony"
    ],
    "SD 1.5":[
        "SD 1.5"
    ]
}

if isFile("./static/json/settings.json"):
    gconfig.update(json.load(open('./static/json/settings.json', 'r', encoding='utf-8')))

gconfig["HF_TOKEN"] = (
    open(f'C:/Users/{os.getlogin()}/.cache/huggingface/token', 'r').read().strip() 
    if os.path.exists(f'C:/Users/{os.getlogin()}/.cache/huggingface/token') else
    json.load(open('./static/json/settings.json', 'r', encoding='utf-8'))["HF_TOKEN"]
    if isFile("./static/json/settings.json") else
    ""
)

#login to huggingface_cli using the lib
def login_to_huggingface():
    from huggingface_hub import login
    if gconfig["HF_TOKEN"]:
        login(token=gconfig["HF_TOKEN"])
    else:
        login()

login_to_huggingface()

if not isDirectory(gconfig["generated_dir"]):
    os.mkdir(gconfig["generated_dir"])

class controlNets:
    def get_depth_map(image, width, height):
        depth_estimator = ppln('depth-estimation')
        image = Image.fromarray(image)
        image = depth_estimator(image, use_fast=True, device="cuda")['depth']
        image = np.array(image)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image).resize((width, height), resample=Image.BICUBIC)

        return image
    def get_canny_image(image, width, height):
        image = np.array(image)

        canny_edges = cv2.Canny(image, 100, 200)
        canny_edges = canny_edges[:, :, None]
        canny_edges = np.concatenate([canny_edges, canny_edges, canny_edges], axis=2)
        return Image.fromarray(canny_edges).resize((width, height), resample=Image.BICUBIC)

#TODO:  function to load the selected scheduler from name
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

#TODO:  function to load pipeline from given huggingface repo and scheduler
def load_pipeline(model_name, model_type, generation_type, scheduler_name):
    gconfig["status"] = "Loading New Pipeline... (loading Pipeline)"
    #TODO: Set the pipeline

    kwargs = {}

    if "controlnet" in generation_type:
        if "canny" in generation_type and model_type in gconfig["SDXL"]:
            controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16)
        if "canny" in generation_type and model_type in gconfig["SD 1.5"]:
            controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)

        if "depth" in generation_type and model_type in gconfig["SDXL"]:
            controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-depth-sdxl-1.0", torch_dtype=torch.float16)
        if "depth" in generation_type and model_type in gconfig["SD 1.5"]:
            controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16)

        kwargs["controlnet"] = controlnet

    if model_type in gconfig["SD 1.5"] and "txt2img" in generation_type:
        kwargs["custom_pipeline"] = "lpw_stable_diffusion"
    elif model_type in gconfig["SDXL"] and "txt2img" in generation_type:
        kwargs["custom_pipeline"] = "lpw_stable_diffusion_xl"
        kwargs["clip_skip"] = 2
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        kwargs["tokenizer"] = tokenizer

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

    gconfig["status"] = "Loading New Pipeline... (Pipeline loaded)"

    gconfig["status"] = "Loading New Pipeline... (pipe)"
    #TODO: Load the pipeline

    pipe = pipeline(
        model_name,
        torch_dtype=torch.float16,
        use_safetensors=True,
        add_watermarker=False,
        use_auth_token=gconfig["HF_TOKEN"],
        safety_checker=None,
        **kwargs
    )

    gconfig["status"] = "Loading New Pipeline... (loading VAE)"
    if gconfig["use_long_clip"]:
        clip_model = CLIPModel.from_pretrained("zer0int/LongCLIP-GmP-ViT-L-14", torch_dtype=torch.float16)
        clip_processor = CLIPProcessor.from_pretrained("zer0int/LongCLIP-GmP-ViT-L-14")
    else:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16", torch_dtype=torch.float16)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    pipe.clip_model = clip_model
    pipe.clip_processor = clip_processor

    #TODO: Load the VAE model
    if not hasattr(pipe, "vae") or pipe.vae is None:
        gconfig["status"] = "Model does not include a VAE. Loading external VAE..."
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16,
        )
        pipe.vae = vae
        gconfig["status"] = "External VAE loaded."

    gconfig["status"] = "Loading New Pipeline... (VAE loaded)"

    if scheduler_name != "None":
        pipe = load_scheduler(pipe, scheduler_name)
    gconfig["status"] = "Loading New Pipeline... (pipe loaded)"

    if torch.cuda.is_available():
        pipe.to('cuda')
    else:
        pipe.to('cpu')
        gconfig["status"] = "Using CPU..."
    
    if gconfig["enable_attention_slicing"]:
        pipe.enable_attention_slicing()
    if gconfig["enable_xformers_memory_efficient_attention"]:
        pipe.enable_xformers_memory_efficient_attention()
    if gconfig["enable_model_cpu_offload"]:
        pipe.enable_model_cpu_offload()
    if gconfig["enable_sequential_cpu_offload"]:
        pipe.enable_sequential_cpu_offload()

    gconfig["status"] = "Pipeline Loaded..."
    return pipe

def latents_to_rgb(latents):
    weights = (
        (60, -60, 25, -70),
        (60,  -5, 15, -50),
        (60,  10, -5, -35),
    )

    weights_tensor = torch.t(torch.tensor(weights, dtype=latents.dtype).to(latents.device))
    biases_tensor = torch.tensor((150, 140, 130), dtype=latents.dtype).to(latents.device)
    rgb_tensor = torch.einsum("...lxy,lr -> ...rxy", latents, weights_tensor) + biases_tensor.unsqueeze(-1).unsqueeze(-1)
    image_array = rgb_tensor.clamp(0, 255).byte().cpu().numpy().transpose(1, 2, 0)

    return Image.fromarray(image_array)

def image_to_base64(img, temp_file=f"{gconfig["generated_dir"]}temp_base64_image.png"):
    img = img.convert("RGB")
    img.save(temp_file, format="PNG")
    with open(temp_file, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode()
    os.remove(temp_file)
    return "data:image/png;base64," + img_base64

def generateImage(pipe, model, prompt, original_prompt, negative_prompt, seed, width, height, img_input, use_orig_img, strength, model_type, generation_type, image_size, cfg_scale, samplingSteps, scheduler_name, image_count, prompt_count):
    #TODO: Generate image with progress tracking
    current_time = time.time()

    if not isDirectory(gconfig["generated_dir"]):
        os.makedirs(gconfig["generated_dir"])

    def progress(pipe, step_index, timestep, callback_kwargs):
        gconfig["status"] = int(math.floor(step_index / samplingSteps * 100))
        gconfig["progress"] = int(math.floor(((image_count - gconfig["remainingImages"]) + (step_index / samplingSteps)) / (image_count * prompt_count) * 100))

        if gconfig["show_latents"]:
            image_path = os.path.join(gconfig["generated_dir"], f'image{current_time}_{seed}.png')
            image = latents_to_rgb(callback_kwargs["latents"][0])
            image.save(image_path, 'PNG')

            gconfig["image_cache"][seed] = [image_path]

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
                    if "depth" in generation_type:
                        new_image = controlNets.get_depth_map(image, width, height)
                    else:
                        new_image = image
                else:
                    new_image = image

                #! Pass the image to pipeline - (kwargs for controlnet)
                kwargs["image"] = new_image
                kwargs["strength"] = strength
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
                kwargs["strength"] = strength
        else:
            #! Pass the parameters to the pipeline - (kwargs for txt2img)
            kwargs["width"] = width
            kwargs["height"] = height
        
        if not gconfig["enable_sequential_cpu_offload"]:
            if hasattr(pipe, "tokenizer_2"):
                compel = Compel(
                    tokenizer=[pipe.tokenizer, pipe.tokenizer_2] ,
                    text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
                    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                    requires_pooled=[False, True]
                )
                kwargs["prompt_embeds"], kwargs["pooled_prompt_embeds"] = compel(prompt)
                kwargs["negative_prompt_embeds"], kwargs["negative_pooled_prompt_embeds"] = compel(negative_prompt)
            else:
                compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
                kwargs["prompt_embeds"] = compel(prompt)
                kwargs["negative_prompt_embeds"] = compel(negative_prompt)
        else:
            kwargs["prompt"] = prompt
            kwargs["negative_prompt"] = negative_prompt

        with torch.no_grad():
            image = pipe(
                **kwargs
            ).images[0]

        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("Prompt", prompt)
        metadata.add_text("OriginalPrompt", original_prompt)
        metadata.add_text("NegativePrompt", negative_prompt)
        metadata.add_text("Width", str(width))
        metadata.add_text("Height", str(height))
        metadata.add_text("CFGScale", str(cfg_scale))
        metadata.add_text("ImgInput", str(image_to_base64(load_image(img_input).convert("RGB"))) if img_input else "N/A")
        metadata.add_text("Strength", str(strength) if "img2img" in model_type else "N/A")
        metadata.add_text("Seed", str(seed))
        metadata.add_text("SamplingSteps", str(samplingSteps))
        metadata.add_text("Model", str(model))
        metadata.add_text("Scheduler", str(scheduler_name))

        #TODO: Save the image to the temporary directory
        image_path = os.path.join(gconfig["generated_dir"], f'image{current_time}_{seed}.png')
        image.save(image_path, 'PNG', pnginfo=metadata)

        gconfig["status"] = "DONE"
        gconfig["progress"] = 0

        return image_path

    except Exception:
        traceback_details = traceback.format_exc()
        gconfig["status"] = f"Generation Stopped with reason:<br>{traceback_details}"
        gconfig["generation_stopped"] = True
        gconfig["generating"] = False
        print(f"Generation Stopped with reason:\n{traceback_details}")
        return False

@app.route('/generate', methods=['POST'])
def generate():
    #TODO: Check if generation is already in progress
    if gconfig["generating"] or gconfig["downloading"]:
        return jsonify(status='Image generation already in progress'), 400

    gconfig["generating"] = True
    gconfig["image_cache"] = {}
    gconfig["status"] = "Starting Image Generation..."

    #TODO: Get parameters from the request
    model_name = request.form.get('model', 'https://huggingface.co/cagliostrolab/animagine-xl-3.1/blob/main/animagine-xl-3.1.safetensors')
    model_type = request.form.get('model_type', 'SDXL')
    scheduler_name = request.form.get('scheduler', 'Euler a')
    prompts = request.form.get('prompt', '1girl, cute, kawaii, full body')
    negative_prompt = request.form.get('negative_prompt', 'default_negative_prompt')
    width = int(request.form.get('width', 832))
    height = int(request.form.get('height', 1216))
    strength = float(request.form.get('strength', 0.5))
    img_input_link = request.form.get('img_input_link', "")
    img_input_img = request.files.get('img_input_img', "")
    use_orig_img = request.form.get('use_orig_img', "false")
    generation_type = request.form.get('generation_type', 'txt2img')
    image_size = request.form.get('image_size', 'original')
    cfg_scale = float(request.form.get('cfg_scale', 7))
    image_count = int(request.form.get('image_count', 4))
    custom_seed = int(request.form.get('custom_seed', -1))
    samplingSteps = int(request.form.get('sampling_steps', 28))

    if custom_seed != -1:
        image_count = 1

    #save the temp image if provided and if not txt2img
    if img_input_img and generation_type != "txt2img":
        temp_image = Image.open(img_input_img).convert("RGB")
        temp_image.save(f"{gconfig["generated_dir"]}temp_image.png")

    img_input = f"{gconfig["generated_dir"]}temp_image.png" if img_input_img else img_input_link

    #TODO: Function to generate images
    def generate_images():
        try:
            user_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_name.lstrip("./")).replace("\\", "/")
            pipe = load_pipeline(user_model_path, model_type, generation_type, scheduler_name)
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

            for prompt in prompt_list:
                for i in range(image_count):
                    if gconfig["generation_stopped"]:
                        gconfig["progress"] = 0
                        gconfig["status"] = "Generation Stopped"
                        gconfig["generating"] = False
                        gconfig["generation_stopped"] = False
                        raise Exception("Generation Stopped")

                    #TODO: Update the progress message
                    gconfig["remainingImages"] = (image_count * len(prompt_list)) - i
                    gconfig["status"] = f"Generating {gconfig["remainingImages"] * len(prompt_list)} Images..."
                    gconfig["progress"] = 0

                    #TODO: Generate a new seed for each image
                    if gconfig["custom_seed"] == -1:
                        seed = random.randint(0, 100000000000)
                    else:
                        seed = gconfig["custom_seed"]

                    image_path = generateImage(pipe, model_name, prompt, prompts, negative_prompt, seed, width, height, img_input, use_orig_img, strength, model_type, generation_type, image_size, cfg_scale, samplingSteps, scheduler_name, image_count, len(prompt_list))

                    #TODO: Store the generated image path
                    if image_path:
                        gconfig["image_cache"][seed] = [image_path]
        except Exception:
            traceback_details = traceback.format_exc()
            gconfig["generating"] = False
            gconfig["generation_stopped"] = False
            gconfig["status"] = f"Error Generating Images...<br>{traceback_details}"
            print(f"Error Generating Images...\n{traceback_details}")
            gconfig["progress"] = 0

        finally:
            del pipe
            torch.cuda.ipc_collect()
            gc.collect()
            torch.cuda.empty_cache()
            gconfig["status"] = "Generation Complete"
            gconfig["progress"] = 0
            gconfig["generating"] = False
            gconfig["generation_stopped"] = False

    #TODO: Start image generation in a separate thread to avoid blocking
    threading.Thread(target=generate_images).start()
    return jsonify(status='Image generation started', count=image_count)

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
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Convert the result to a PIL Image
    if "canny" in controlnet_type:
        edges_image = controlNets.get_canny_image(img, width, height)
    if "depth" in controlnet_type:
        edges_image = controlNets.get_depth_map(img, width, height)
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
    #TODO: Convert the generated images to a list to send to the client
    images =[{
            'img': path[0],
            'seed': seed
        } for seed, path in gconfig["image_cache"].items()]

    return jsonify(
        images=images,
        imgprogress=gconfig["status"],
        allpercentage=gconfig["progress"],
        remainingimages=gconfig["remainingImages"]-1 if gconfig["remainingImages"] > 0 else gconfig["remainingImages"]
    )

@app.route('/generated/<filename>', methods=['GET'])
def serve_temp_image(filename):
    image_path = os.path.join(gconfig["generated_dir"], filename)
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
        try:
            os.remove(file)
        except Exception:
            traceback_details = traceback.format_exc()
            gconfig["progress"] = 0
            gconfig["status"] = f"Error Deleteing File... {traceback_details}"
            print(f"Error Deleteing File... {traceback_details}")
    return jsonify(status='Images cleared')

@app.route('/restart', methods=['POST'])
def restart_app():
    gconfig["progress"] = 0
    subprocess.Popen([sys.executable] + sys.argv)
    os._exit(0)

@app.route('/')
def index():
    return render_template('index.html')

def get_clip_token_info(text):
    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
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
        "CLIP Token Count": len(clip_tokens),
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
    return jsonify(settings)

@app.route('/metadata')
def metadata():
    return render_template('metadata.html')

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
    gconfig["status"] = "Server Started"
