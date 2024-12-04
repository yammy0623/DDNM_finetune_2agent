from diffusers import DDPMPipeline, DDIMPipeline, DiffusionPipeline

import onnxruntime
import numpy as np
import torch
import argparse

parser = argparse.ArgumentParser(description="ONNX test Script")
parser.add_argument("--model", type=str, required=True, help="Model ID to use for the pipeline")
parser.add_argument("--type", type=str, required=False, help="Type of model to use")
parser.add_argument("--dir", "-d", type=str, required=True, help="File directory for the ONNX model")

args = parser.parse_args()
model_id = args.model
model_type = args.type
onnx_model_path = args.dir

# Load the pipeline based on the model type
# if model_type == "DDPM":
#     pipeline = DDPMPipeline.from_pretrained(model_id)
# elif model_type == "DDIM":
#     pipeline = DDIMPipeline.from_pretrained(model_id)
# else:
#     pipeline = DiffusionPipeline.from_pretrained(model_id)

pipeline = DiffusionPipeline.from_pretrained(model_id)

# Move the model to the desired device
device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline.to(device)
pipeline.unet.eval()

# Load the ONNX model
session = onnxruntime.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider'])

# Set a fixed seed for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Define the input shape based on the model's expected dimensions
batch_size = 1
in_channels = pipeline.unet.in_channels
sample_size = pipeline.unet.sample_size

# Create a dummy input tensor for PyTorch
dummy_input_torch = torch.randn(
    batch_size,
    in_channels,
    sample_size,
    sample_size,
    dtype=torch.float32,
    device=device,
)

# Create a dummy timesteps tensor for PyTorch
dummy_timesteps_torch = torch.tensor(
    [0],  # You can choose any valid timestep value
    dtype=torch.long,
    device=device,
)

# Convert the PyTorch tensors to NumPy arrays for ONNX Runtime
dummy_input_onnx = dummy_input_torch.cpu().numpy()
dummy_timesteps_onnx = dummy_timesteps_torch.cpu().numpy()

# Run inference with the PyTorch model
with torch.no_grad():
    torch_output = pipeline.unet(dummy_input_torch,dummy_timesteps_torch).sample.cpu().numpy()

# Run inference with the ONNX model
# Get the input and output names for the ONNX model
onnx_input_names = [input.name for input in session.get_inputs()]
onnx_output_name = session.get_outputs()[0].name

# Prepare the inputs for the ONNX model
onnx_inputs = {
    onnx_input_names[0]: dummy_input_onnx,
    onnx_input_names[1]: dummy_timesteps_onnx,
}

# Run inference with the ONNX model
onnx_outputs = session.run([onnx_output_name], onnx_inputs)
onnx_output = onnx_outputs[0]

# Calculate the absolute difference
difference = np.abs(torch_output - onnx_output)

# Calculate metrics
max_diff = difference.max()
mean_diff = difference.mean()
std_diff = difference.std()

print(f"Max difference: {max_diff}")
print(f"Mean difference: {mean_diff}")
print(f"Standard deviation of difference: {std_diff}")