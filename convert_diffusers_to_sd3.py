import argparse
import os
import torch
from diffusers import DiffusionPipeline
from safetensors.torch import save_file

def load_transformer_as_unet(root_folder, dtype):
    """
    Load the transformer model (SD3.5's equivalent of UNet) using DiffusionPipeline.
    """
    print(f"Loading pipeline from root folder: {root_folder}")
    
    # Load the pipeline from the root folder (which contains model_index.json)
    pipeline = DiffusionPipeline.from_pretrained(
        root_folder,
        torch_dtype=dtype
    )
    
    # Check the available attributes in the pipeline to see if we can access the transformer
    print(f"Pipeline components: {pipeline.__dict__.keys()}")
    
    # If the transformer is in a different attribute, we need to access it correctly
    transformer = pipeline.transformer.state_dict()  # Adjust this line to use the correct attribute
    
    # Move the model to the correct device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer = {k: v.to(device) for k, v in transformer.items()}
    
    return transformer

def convert_diffusers_to_sd3(diffusers_model_path, sd3_checkpoint_path, dtype=torch.float32):
    """
    Convert a Diffusers model in SD3.5 format into a Stable Diffusion checkpoint.
    """
    # Load transformer (equivalent to UNet in SD3.5)
    transformer = load_transformer_as_unet(diffusers_model_path, dtype)

    # Load VAE
    vae_path = os.path.join(diffusers_model_path, "vae")
    vae = DiffusionPipeline.from_pretrained(vae_path).vae.state_dict()

    # Load text encoder and tokenizer
    text_encoder_path = os.path.join(diffusers_model_path, "text_encoder")
    text_encoder = DiffusionPipeline.from_pretrained(text_encoder_path).text_encoder.state_dict()

    # Save the model weights as safetensors
    state_dict = {
        **{f"model.diffusion_model.{k}": v for k, v in transformer.items()},
        **{f"first_stage_model.{k}": v for k, v in vae.items()},
        **{f"cond_stage_model.transformer.{k}": v for k, v in text_encoder.items()},
    }

    # Handle dtype (fp16, fp32, etc.)
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
