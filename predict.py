# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import os
import time
import torch
import subprocess
import numpy as np
from PIL import Image
from diffusers import (
    FluxKontextPipeline,
    FlowMatchEulerDiscreteScheduler,
    FluxTransformer2DModel,
    AutoencoderKL
)
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from flux.content_filters import PixtralContentFilter
from peft import PeftModel  # Thêm để hỗ trợ LoRA

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
    print("downloading url: ", url)
    print("downloading to: ", dest)
    if not file:
        subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    else:
        subprocess.check_call(["pget", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
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

        # Load components individually
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(SCHEDULER_PATH, local_files_only=True)
        text_encoder = CLIPTextModel.from_pretrained(TE_PATH, torch_dtype=torch.bfloat16, local_files_only=True)
        text_encoder_2 = T5EncoderModel.from_pretrained(TE2_PATH, torch_dtype=torch.bfloat16, local_files_only=True)
        tokenizer = CLIPTokenizer.from_pretrained(TOK_PATH, local_files_only=True)
        tokenizer_2 = T5TokenizerFast.from_pretrained(TOK2_PATH, local_files_only=True)
        transformer = FluxTransformer2DModel.from_pretrained(TRANSFORMER_PATH, torch_dtype=torch.bfloat16, local_files_only=True)
        vae = AutoencoderKL.from_pretrained(VAE_PATH, torch_dtype=torch.bfloat16, local_files_only=True)
        
        # Construct pipeline manually
        self.pipe = FluxKontextPipeline(
            scheduler=scheduler,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            vae=vae
        )
        
        self.pipe.to("cuda")
        self.integrity_checker = PixtralContentFilter(torch.device("cuda"))
        print("Model loaded successfully!")

    def predict(
        self,
        prompt: str = Input(description="Text prompt"),
        image: Path = Input(description="Input image to edit"),
        lora_url: str = Input(
            description="Optional: URL to a LoRA .safetensors file",
            default=None
        ),
        lora_scale: float = Input(
            description="LoRA strength",
            ge=0.0,
            le=1.0,
            default=1.0
        ),
        aspect_ratio: str = Input(
            description="Aspect ratio of the generated image. Use 'match_input_image' to match the input image.",
            choices=list(ASPECT_RATIOS.keys()),
            default="match_input_image",
        ),
        guidance_scale: float = Input(
            description="Guidance scale for generation",
            ge=1.0,
            le=10.0,
            default=2.5
        ),
        num_inference_steps: int = Input(
            description="Number of inference steps",
            ge=1,
            le=50,
            default=28
        ),
        seed: int = Input(
            description="Random seed. Set -1 to randomize",
            default=-1
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        # Set random seed
        if seed == -1:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        generator = torch.Generator(device="cuda").manual_seed(seed)

        # Load input image
        input_image = Image.open(image).convert("RGB")
        if aspect_ratio == "match_input_image":
            width, height = input_image.size
            input_aspect_ratio = width / height
            _, target_width, target_height = min((abs(input_aspect_ratio - w / h), w, h) for w, h in ASPECT_RATIOS.values() if w is not None)
        else:
            target_width, target_height = ASPECT_RATIOS[aspect_ratio]

        input_image = input_image.resize((target_width, target_height), Image.Resampling.LANCZOS)

        # --- Load LoRA if provided ---
        if lora_url:
            lora_path = "./lora.safetensors"
            if not os.path.exists(lora_path):
                download_weights(lora_url, lora_path, file=True)
            print(f"Applying LoRA from {lora_path} with scale {lora_scale}")
            self.pipe.transformer = PeftModel.from_pretrained(self.pipe.transformer, lora_path)
            self.pipe.fuse_lora(lora_scale)

        # Run the model
        print(f"Editing image with prompt: '{prompt}'")
        result = self.pipe(
            image=input_image,
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=target_width,
            height=target_height,
            generator=generator
        )
        output_image = result.images[0]

        # Run integrity checker
        image_array = np.array(output_image) / 255.0
        image_array = 2 * image_array - 1
        image_tensor = torch.from_numpy(image_array).to("cuda", dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)
        if self.integrity_checker.test_image(image_tensor):
            raise ValueError("Generated image flagged by content filter")

        output_path = "/tmp/output.png"
        output_image.save(output_path)
        return Path(output_path)
