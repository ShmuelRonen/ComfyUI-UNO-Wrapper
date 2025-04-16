# ComfyUI-UNO-Wrapper

This extension integrates ByteDance's UNO-FLUX model into ComfyUI, allowing you to use UNO's powerful text-to-image generation with reference capabilities.

![view](https://github.com/user-attachments/assets/d69881e8-36f9-44ac-b2b6-673536ece186)


## Features

- Generate images with text prompts and up to 4 reference images
- Full control over guidance scale, steps, and dimensions
- Batch generation support
- Leverages your existing ComfyUI text encoders
- **NEW: Support for applying custom LoRAs from your models/loras directory**

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

### 3. Hugging Face Authentication Setup

The UNO-FLUX implementation automatically attempts to download model files from Hugging Face repositories. For private or gated models, you'll need to set up authentication:

#### Option 1: Using config.json (Recommended)

1. Create a file named `config.json` in the `ComfyUI/custom_nodes/ComfyUI-UNO-Wrapper` directory
2. Add the following content, replacing with your actual Hugging Face token:

```json
{
    "hf_token": "your_huggingface_token_here"
}
```

3. Restart ComfyUI for the changes to take effect


#### Getting a Hugging Face Token

1. Visit [Hugging Face](https://huggingface.co/) and create an account or log in
2. Go to your profile settings
3. Navigate to "Access Tokens"
4. Create a new token with "read" permissions
5. Copy the token and paste it into your `config.json` file

## Usage

### Basic Workflow

1. Add a "UNO Model Loader" node
   - Configure device settings if needed (most settings work with defaults)
   - **NEW: Select a custom LoRA from the dropdown to influence the generation style**

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
- **Download Errors**: Check your Hugging Face token in the config.json file is correct and has proper permissions
- **LoRA Not Working**: Ensure your LoRA files are in the correct format and located in the `models/loras` directory

## License

### Based on UNO ByteDance project: https://github.com/bytedance/UNO

This project follows the same license as UNO-FLUX (Apache License 2.0)
