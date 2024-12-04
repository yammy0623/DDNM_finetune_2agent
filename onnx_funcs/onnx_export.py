from diffusers import DDPMPipeline, DDIMPipeline, DiffusionPipeline

import torch
import argparse
import onnxruntime
import numpy as np

parser = argparse.ArgumentParser(description="ONNX Export Script")
parser.add_argument("--model", type=str, required=True, help="Model ID to use for the pipeline")
parser.add_argument("--type", type=str, required=True, help="Type of model to use")
parser.add_argument("--output", "-o", type=str, required=True, help="Output file name for the ONNX model")

args = parser.parse_args()
model_id = args.model
model_type = args.type
output = args.output

# if model_type == "DDPM":
#     pipeline = DDPMPipeline.from_pretrained(model_id)
# elif model_type == "DDIM":
#     pipeline = DDIMPipeline.from_pretrained(model_id)
# else:
#     print("Unrecognized model type!")
#     exit()

pipeline = DiffusionPipeline.from_pretrained(model_id)

# Define a dummy input matching the model's expected input size
dummy_input = torch.randn(1, pipeline.unet.in_channels, pipeline.unet.sample_size, pipeline.unet.sample_size)
dummy_timestep = torch.tensor(0, dtype=torch.long)

# Export the UNet model to ONNX
torch.onnx.export(
    pipeline.unet,                      # The model to export
    (dummy_input, dummy_timestep),      # The model's input
    f"{output}.onnx",                   # The output file
    opset_version=18,                   # ONNX opset version
    input_names=["input", "timesteps"],              # Input tensor name
    output_names=["output"],            # Output tensor name
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},  # Enable dynamic batching
)