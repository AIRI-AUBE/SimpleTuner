import argparse
import os
import torch
from diffusers import AutoencoderKL, StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from safetensors.torch import load_file, save_file, safe_open

def load_transformer_as_unet(transformer_path, dtype):
    """
    Load the transformer model (SD3.5's equivalent of UNet) from sharded safetensors.
    """
    index_file = os.path.join(transformer_path, "diffusion_pytorch_model.safetensors.index.json")
    if os.path.exists(index_file):
        # Use safe_open to handle the entire sharded model
        with safe_open(index_file, framework="pt", device="cpu") as f:
            transformer = f.load()
    else:
        raise FileNotFoundError(f"Transformer index file not found: {index_file}")
    
    # Convert to the desired dtype
    return {k: v.to(dtype) for k, v in transformer.items()}

def convert_diffusers_to_sd3(diffusers_model_path, sd3_checkpoint_path, dtype=torch.float32):
    """
    Convert a Diffusers model in SD3.5 format into a Stable Diffusion checkpoint.
    """
    # Load transformer (equivalent to UNet)
    transformer_path = os.path.join(diffusers_model_path, "transformer")
    transformer = load_transformer_as_unet(transformer_path, dtype)

    # Load VAE
    vae_path = os.path.join(diffusers_model_path, "vae")
    vae = AutoencoderKL.from_pretrained(vae_path)

    # Load text encoder and tokenizer
    text_encoder_path = os.path.join(diffusers_model_path, "text_encoder")
    text_encoder = CLIPTextModel.from_pretrained(text_encoder_path)

    tokenizer_path = os.path.join(diffusers_model_path, "tokenizer")
    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)

    # Create a manual pipeline
    pipeline = StableDiffusionPipeline(
        unet=None,  # Placeholder, we replace unet with the transformer below
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
    )
    pipeline.unet = transformer  # Set the transformer as unet

    # Prepare state_dict for saving
    state_dict = {
        **{f"model.diffusion_model.{k}": v for k, v in transformer.items()},
        **{f"first_stage_model.{k}": v for k, v in vae.state_dict().items()},
        **{f"cond_stage_model.transformer.{k}": v for k, v in text_encoder.state_dict().items()},
    }

    if dtype in [torch.float16, torch.bfloat16]:
        state_dict = {k: v.half() for k, v in state_dict.items()}

    # Save as safetensors
    save_file(state_dict, sd3_checkpoint_path)

    print(f"SD3.5 model successfully saved to {sd3_checkpoint_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Diffusers SD3.5 model to SD3 checkpoint.")
    parser.add_argument("--diffusers_model_path", type=str, required=True, help="Path to the Diffusers model directory.")
    parser.add_argument("--sd3_checkpoint_path", type=str, required=True, help="Path to save the SD3 checkpoint.")
    parser.add_argument("--dtype", type=str, default="fp32", help="Precision type: fp16, bf16, fp32 (default: fp32).")

    args = parser.parse_args()

    # Map dtype string to torch dtype
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    convert_diffusers_to_sd3(args.diffusers_model_path, args.sd3_checkpoint_path, dtype)
