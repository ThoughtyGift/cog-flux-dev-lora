# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import os, re, time, subprocess, torch
import numpy as np
from PIL import Image
from typing import List

from diffusers import (
    FluxKontextPipeline,
    FlowMatchEulerDiscreteScheduler,
    FluxTransformer2DModel,
    AutoencoderKL
)
from transformers import (
    CLIPTextModel, CLIPTokenizer,
    T5EncoderModel, T5TokenizerFast,
    CLIPImageProcessor
)
from flux.content_filters import PixtralContentFilter
from weights import WeightsDownloadCache
from lora_loading_patch import load_lora_into_transformer

# ---- WEIGHT PATHS ----
MODEL_CACHE = "./checkpoints"
MODEL_URL = "https://weights.replicate.delivery/default/black-forest-labs/kontext/huggingface/main.tar"
SCHEDULER_PATH = "./checkpoints/scheduler"
SCHEDULER_URL = "https://weights.replicate.delivery/default/black-forest-labs/kontext/huggingface/scheduler.tar"
TE_PATH = "./checkpoints/text_encoder"
TE_URL = "https://weights.replicate.delivery/default/black-forest-labs/kontext/huggingface/te.tar"
TE2_PATH = "./checkpoints/text_encoder2"
TE2_URL = "https://weights.replicate.delivery/default/black-forest-labs/kontext/huggingface/te2.tar"
TOK_PATH = "./checkpoints/tokenizer"
TOK_URL = "https://weights.replicate.delivery/default/black-forest-labs/kontext/huggingface/tok.tar"
TOK2_PATH = "./checkpoints/tokenizer_2"
TOK2_URL = "https://weights.replicate.delivery/default/black-forest-labs/kontext/huggingface/tok2.tar"
TRANSFORMER_PATH = "./checkpoints/transformer"
TRANSFORMER_URL = "https://weights.replicate.delivery/default/black-forest-labs/kontext/huggingface/transformer.tar"
VAE_PATH = "./checkpoints/vae"
VAE_URL = "https://weights.replicate.delivery/default/black-forest-labs/kontext/huggingface/vae.tar"

ASPECT_RATIOS = {
    "1:1": (1024, 1024),
    "16:9": (1344, 768),
    "21:9": (1536, 640),
    "3:2": (1216, 832),
    "2:3": (832, 1216),
    "4:5": (944, 1104),
    "5:4": (1104, 944),
    "3:4": (896, 1152),
    "4:3": (1152, 896),
    "9:16": (768, 1344),
    "9:21": (640, 1536),
    "match_input_image": (None, None),
}

def download_weights(url, dest, file=False):
    start = time.time()
    print("downloading url:", url)
    if not file:
        subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    else:
        subprocess.check_call(["pget", url, dest], close_fds=False)
    print("downloading took:", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load kontext model and prepare for LoRA"""
        print("Downloading weights...")
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)
        if not os.path.exists(SCHEDULER_PATH):
            download_weights(SCHEDULER_URL, SCHEDULER_PATH)
        if not os.path.exists(TE_PATH):
            download_weights(TE_URL, TE_PATH)
        if not os.path.exists(TE2_PATH):
            download_weights(TE2_URL, TE2_PATH)
        if not os.path.exists(TOK_PATH):
            download_weights(TOK_URL, TOK_PATH)
        if not os.path.exists(TOK2_PATH):
            download_weights(TOK2_URL, TOK2_PATH)
        if not os.path.exists(TRANSFORMER_PATH):
            download_weights(TRANSFORMER_URL, TRANSFORMER_PATH)
        if not os.path.exists(VAE_PATH):
            download_weights(VAE_URL, VAE_PATH)

        # Load components
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(SCHEDULER_PATH, local_files_only=True)
        text_encoder = CLIPTextModel.from_pretrained(TE_PATH, torch_dtype=torch.bfloat16, local_files_only=True)
        text_encoder_2 = T5EncoderModel.from_pretrained(TE2_PATH, torch_dtype=torch.bfloat16, local_files_only=True)
        tokenizer = CLIPTokenizer.from_pretrained(TOK_PATH, local_files_only=True)
        tokenizer_2 = T5TokenizerFast.from_pretrained(TOK2_PATH, local_files_only=True)
        transformer = FluxTransformer2DModel.from_pretrained(
            TRANSFORMER_PATH, torch_dtype=torch.bfloat16, local_files_only=True
        )
        vae = AutoencoderKL.from_pretrained(VAE_PATH, torch_dtype=torch.bfloat16, local_files_only=True)

        # Build pipeline
        self.pipe = FluxKontextPipeline(
            scheduler=scheduler,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            vae=vae
        ).to("cuda")

        # enable LoRA loading
        self.pipe.__class__.load_lora_into_transformer = classmethod(load_lora_into_transformer)
        self.weights_cache = WeightsDownloadCache()
        self.last_loaded_lora = None

        # Content filter
        self.integrity_checker = PixtralContentFilter(torch.device("cuda"))
        print("Model + LoRA ready!")

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(description="Prompt for generation"),
        image: Path = Input(description="Optional input image for conditioning", default=None),
        aspect_ratio: str = Input(
            description="Aspect ratio of output",
            choices=list(ASPECT_RATIOS.keys()),
            default="match_input_image"
        ),
        guidance_scale: float = Input(description="Guidance scale", ge=1.0, le=10.0, default=2.5),
        num_inference_steps: int = Input(description="Steps", ge=1, le=50, default=28),
        seed: int = Input(description="Random seed (-1 = random)", default=-1),
        hf_lora: str = Input(description="HF/Replicate/CivitAI/URL to LoRA", default=None),
        lora_scale: float = Input(description="Scale for LoRA", ge=0, le=1, default=0.8),
    ) -> Path:
        # Random seed
        if seed == -1:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed {seed}")
        generator = torch.Generator("cuda").manual_seed(seed)

        # Handle LoRA
        if hf_lora:
            if hf_lora != self.last_loaded_lora:
                self.pipe.unload_lora_weights()
                print(f"Loading LoRA: {hf_lora}")
                if re.match(r"^[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+$", hf_lora):
                    self.pipe.load_lora_weights(hf_lora)
                elif hf_lora.endswith(".safetensors") or hf_lora.startswith("http"):
                    lora_path = self.weights_cache.ensure(hf_lora, file=True)
                    self.pipe.load_lora_weights(lora_path)
                else:
                    raise Exception(f"Unsupported LoRA format: {hf_lora}")
                self.last_loaded_lora = hf_lora
            self.pipe.fuse_lora(lora_scale=lora_scale)
        else:
            self.pipe.unload_lora_weights()
            self.last_loaded_lora = None

        # Load & resize image
        init_image = None
        if image:
            init_image = Image.open(image).convert("RGB")
            if aspect_ratio == "match_input_image":
                target_width, target_height = init_image.size
            else:
                target_width, target_height = ASPECT_RATIOS[aspect_ratio]
            init_image = init_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        else:
            if aspect_ratio == "match_input_image":
                target_width, target_height = ASPECT_RATIOS["1:1"]
            else:
                target_width, target_height = ASPECT_RATIOS[aspect_ratio]

        # Run model
        print(f"Running with prompt: {prompt}")
        result = self.pipe(
            image=init_image,
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=target_width,
            height=target_height,
            generator=generator,
        )
        output_image = result.images[0]

        # Run content filter
        arr = np.array(output_image) / 255.0
        arr = 2 * arr - 1
        tensor = torch.from_numpy(arr).to("cuda", dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)
        if self.integrity_checker.test_image(tensor):
            raise ValueError("Image flagged by content filter.")

        # Save output
        out_path = "/tmp/output.png"
        output_image.save(out_path)
        return Path(out_path)
