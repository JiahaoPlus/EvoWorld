from safetensors.torch import load_file, save_file
import argparse
import os
import glob

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Convert 32-bit model to 16-bit model')
parser.add_argument('--root', type=str, required=True, help='Path to the original model directory')
# root can be "MODELS/trained_deepspeed_o1_genex_on_UE_Curve"
args = parser.parse_args()

pattern = os.path.join(args.root, '**', '*.safetensors')

# glob with recursive=True
safetensor_files = glob.glob(pattern, recursive=True)
for path in safetensor_files:
    if ".fp16" in path:
        print(f"Skipping {path} as it is already in fp16 format.")
        continue
    # print(path)
    output_path = path.replace(".safetensors", ".fp16.safetensors")
    weights = load_file(path)
    # Convert each tensor to float16
    fp16_weights = {key: tensor.half() for key, tensor in weights.items()}
    # Save the converted weights
    save_file(fp16_weights, output_path)
    print(f"Converted {path} to fp16 and saved as {output_path}")
