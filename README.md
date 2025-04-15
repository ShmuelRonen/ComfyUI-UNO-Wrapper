# ComfyUI-UNO-Wrapper

This extension integrates ByteDance's UNO-FLUX model into ComfyUI, allowing you to use UNO's powerful text-to-image generation with reference capabilities.

![view](https://github.com/user-attachments/assets/d69881e8-36f9-44ac-b2b6-673536ece186)

## Features

- Generate images with text prompts and up to 4 reference images
- Full control over guidance scale, steps, and dimensions
- Batch generation support
- Leverages your existing ComfyUI text encoders

## Installation

### 1. Install ComfyUI-UNO-Wrapper

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ShmuelRonen/ComfyUI-UNO-Wrapper
```

### 2. Download the LoRA Model (REQUIRED)

Most models will be installed automatically, but you must manually download the ByteDance LoRA model:

1. Create a folder named `uno_lora` in your `ComfyUI/models/loras` directory
2. Download `dit_lora.safetensors` and place it in this folder

#### Download Link: 
- [bytedance-research/UNO](https://huggingface.co/bytedance-research/UNO)

### 3. Important Note: Flux Model Access

The UNO-FLUX implementation automatically attempts to download model files from Hugging Face repositories. If you encounter authorization errors, please:

1. Visit the Flux model page on Hugging Face:
   https://huggingface.co/black-forest-labs/FLUX.1-dev

2. Sign in to your Hugging Face account

3. Accept any terms of use for the model (if required)

4. Once authorized, return to ComfyUI and try again - the download should now work properly

This one-time authorization will allow the automatic downloads to work correctly.

## Usage

### Basic Workflow

1. Add a "UNO Model Loader" node
   - Configure device settings if needed (most settings work with defaults)

2. Connect it to a "UNO Image Generator" node
   - Enter your text prompt
   - Set dimensions, guidance scale, and steps as desired
   - Connect up to 4 reference images
   - Set batch size and seed

3. Run the workflow to generate images

### Working with Multiple Reference Images

UNO-FLUX's standout feature is its ability to incorporate elements from multiple reference images:

1. Connect different reference images to `image_ref1`, `image_ref2`, etc.
2. Each reference image will influence different aspects of the generated output
3. Use a descriptive prompt to guide how the references are combined

![teaser](https://github.com/user-attachments/assets/c1a4d514-35ed-4208-bc81-ec26298dd8c5)


## Troubleshooting

- **Generation Fails**: Make sure you've downloaded the LoRA model to `models/loras/uno_lora/dit_lora.safetensors`
- **CUDA Errors**: Try enabling offload in the UNO Model Loader
- **Poor Results**: Try adjusting the guidance scale (higher for more prompt adherence, lower for more creativity)

## License

### Based on UNO ByteDance project: https://github.com/bytedance/UNO

This project follows the same license as UNO-FLUX (Apache License 2.0)
