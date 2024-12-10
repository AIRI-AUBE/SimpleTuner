import argparse
import os
import torch
from safetensors.torch import save_file, load_file

def reverse_scale_shift(weight, dim):
    scale, shift = weight.chunk(2, dim=0)
    return torch.cat([shift, scale], dim=0)

def convert_diffusers_to_sd3(diffusers_model_path, sd3_checkpoint_path, dtype=torch.float32):
    # Load Diffusers Transformer Model

    transformer_path = os.path.join(diffusers_model_path, "transformer/diffusion_pytorch_model.safetensors.index.json")
    if os.path.exists(transformer_path):
        # Load the sharded safetensors model
        transformer = load_file(os.path.join(diffusers_model_path, "transformer"))
    else:
        raise FileNotFoundError(f"Transformer model files not found in {os.path.join(diffusers_model_path, 'transformer')}")

    converted_state_dict = {}

    # Revert position embedding weights
    converted_state_dict["pos_embed"] = transformer["pos_embed.pos_embed"]
    converted_state_dict["x_embedder.proj.weight"] = transformer["pos_embed.proj.weight"]
    converted_state_dict["x_embedder.proj.bias"] = transformer["pos_embed.proj.bias"]

    # Revert context projection weights
    converted_state_dict["context_embedder.weight"] = transformer["context_embedder.weight"]
    converted_state_dict["context_embedder.bias"] = transformer["context_embedder.bias"]

    # Revert timestep embeddings
    converted_state_dict["t_embedder.mlp.0.weight"] = transformer["time_text_embed.timestep_embedder.linear_1.weight"]
    converted_state_dict["t_embedder.mlp.0.bias"] = transformer["time_text_embed.timestep_embedder.linear_1.bias"]
    converted_state_dict["t_embedder.mlp.2.weight"] = transformer["time_text_embed.timestep_embedder.linear_2.weight"]
    converted_state_dict["t_embedder.mlp.2.bias"] = transformer["time_text_embed.timestep_embedder.linear_2.bias"]

    # Revert pooled context projection
    converted_state_dict["y_embedder.mlp.0.weight"] = transformer["time_text_embed.text_embedder.linear_1.weight"]
    converted_state_dict["y_embedder.mlp.0.bias"] = transformer["time_text_embed.text_embedder.linear_1.bias"]
    converted_state_dict["y_embedder.mlp.2.weight"] = transformer["time_text_embed.text_embedder.linear_2.weight"]
    converted_state_dict["y_embedder.mlp.2.bias"] = transformer["time_text_embed.text_embedder.linear_2.bias"]

    # Revert Transformer blocks
    for key in transformer.keys():
        if "transformer_blocks" in key:
            new_key = key.replace("transformer_blocks", "joint_blocks").replace(
                ".attn.", ".x_block.attn.").replace(".attn2.", ".x_block.attn2.")
            converted_state_dict[new_key] = transformer[key]

    # Handle qk norm
    for i in range(16):  # Assuming 16 layers
        if f"transformer_blocks.{i}.attn.norm_q.weight" in transformer:
            converted_state_dict[f"joint_blocks.{i}.x_block.attn.ln_q.weight"] = transformer[f"transformer_blocks.{i}.attn.norm_q.weight"]
            converted_state_dict[f"joint_blocks.{i}.x_block.attn.ln_k.weight"] = transformer[f"transformer_blocks.{i}.attn.norm_k.weight"]

    # Handle VAE if present
    vae_path = os.path.join(diffusers_model_path, "vae/pytorch_model.bin")
    if os.path.exists(vae_path):
        vae = torch.load(vae_path)
        converted_state_dict.update(vae)

    # Save to SD3 checkpoint
    save_file(converted_state_dict, sd3_checkpoint_path, metadata={"format": "Stable Diffusion 3"})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Diffusers SD3 model to SD3 checkpoint.")
    parser.add_argument("--diffusers_model_path", type=str, required=True, help="Path to the Diffusers model directory.")
    parser.add_argument("--sd3_checkpoint_path", type=str, required=True, help="Path to save the SD3 checkpoint.")
    parser.add_argument("--dtype", type=str, default="fp32", help="Precision type: fp16, bf16, fp32 (default: fp32).")

    args = parser.parse_args()

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    convert_diffusers_to_sd3(args.diffusers_model_path, args.sd3_checkpoint_path, dtype)
