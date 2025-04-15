import os
import sys
import torch
import numpy as np
from PIL import Image
import folder_paths
import traceback

# Define model paths using ComfyUI's standard structure
MODELS_DIR = folder_paths.models_dir
UNET_MODELS_DIR = os.path.join(MODELS_DIR, "unet")  # Flux models directly in unet directory
LORA_MODELS_DIR = os.path.join(MODELS_DIR, "loras", "uno_lora")  # UNO LoRA in loras/uno_lora
TEXT_ENCODERS_DIR = os.path.join(MODELS_DIR, "text_encoders")  # Text encoders directory

# Create the LoRA directory if it doesn't exist
os.makedirs(LORA_MODELS_DIR, exist_ok=True)

# Flag to track if UNO-FLUX is properly imported
UNO_IMPORTED = False
UNOPipeline = None

# Attempt to import UNO package
try:
    # Try to import from different possible locations
    try:
        from uno.flux.pipeline import UNOPipeline
        UNO_IMPORTED = True
        print("Successfully imported UNO-FLUX pipeline from uno.flux.pipeline")
    except ImportError:
        # Try alternative import paths
        try:
            # Check if 'uno' module is in current directory
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from uno.flux.pipeline import UNOPipeline
            UNO_IMPORTED = True
            print("Successfully imported UNO-FLUX pipeline from local uno module")
        except ImportError:
            # Try importing from parent directory
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            sys.path.append(parent_dir)
            try:
                from uno.flux.pipeline import UNOPipeline
                UNO_IMPORTED = True
                print("Successfully imported UNO-FLUX pipeline from parent directory")
            except ImportError:
                print("Warning: UNO-FLUX package not found in any location. Please install it or add it to your PYTHONPATH.")
                print("UNO-FLUX should be accessible as 'uno.flux.pipeline'")
                UNO_IMPORTED = False
except Exception as e:
    print(f"Error importing UNO-FLUX: {e}")
    UNO_IMPORTED = False


class UNOModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (["flux-dev-fp8"], {"default": "flux-dev-fp8"}),
                "device": (["cuda", "cpu"], {"default": "cuda" if torch.cuda.is_available() else "cpu"}),
                "offload": ("BOOLEAN", {"default": True}),
                "lora_rank": ("INT", {"default": 512, "min": 1, "max": 1024}),
                # Removed the use_existing_text_encoder option from here
            }
        }
    
    RETURN_TYPES = ("UNO_PIPELINE",)
    FUNCTION = "load_model"
    CATEGORY = "UNO"

    def load_model(self, model_name, device, offload, lora_rank):  # Removed the parameter
        # Check if UNO-FLUX is properly imported
        if not UNO_IMPORTED:
            raise RuntimeError(
                "UNO-FLUX package is not properly imported. Please make sure it's installed and in your PYTHONPATH.\n"
                "Installation steps:\n"
                "1. Clone the UNO-FLUX repository\n"
                "2. Add it to your PYTHONPATH or install it\n"
                "3. Restart ComfyUI"
            )
        
        # Set model paths according to ComfyUI structure
        flux_model_path = UNET_MODELS_DIR  # Flux models directly in unet directory
        lora_model_path = LORA_MODELS_DIR
        
        # Set environment variables for the UNO pipeline
        os.environ["UNO_MODEL_PATH"] = flux_model_path
        os.environ["UNO_LORA_PATH"] = lora_model_path
        
        # Always use existing text encoders
        use_existing_text_encoder = True  # Hardcoded to True
        os.environ["UNO_TEXT_ENCODER_PATH"] = TEXT_ENCODERS_DIR
        print(f"Using existing text encoders from: {TEXT_ENCODERS_DIR}")
        
        # Always use LoRA
        only_lora = True
        
        # Initialize UNO pipeline
        try:
            pipeline = UNOPipeline(
                model_type=model_name,
                device=device,
                offload=offload,
                only_lora=only_lora,
                lora_rank=lora_rank
            )
            
            # If the pipeline exposes a method to set paths, use them
            if hasattr(pipeline, "set_text_encoder_path"):
                pipeline.set_text_encoder_path(TEXT_ENCODERS_DIR)
            
            if hasattr(pipeline, "set_lora_path"):
                pipeline.set_lora_path(lora_model_path)
            
            return (pipeline,)
        except Exception as e:
            error_msg = f"Error loading UNO model: {e}\n"
            
            # Check model directories
            if not os.path.exists(os.path.join(flux_model_path, model_name)) and not os.path.exists(os.path.join(flux_model_path, f"{model_name}.safetensors")):
                error_msg += f"Flux model not found: {model_name}\n"
                error_msg += f"Please download the model to: {flux_model_path}\n"
            
            if not os.path.exists(lora_model_path) or not any(f.endswith('.safetensors') for f in os.listdir(lora_model_path) if os.path.isfile(os.path.join(lora_model_path, f))):
                error_msg += f"UNO LoRA models not found in: {lora_model_path}\n"
                error_msg += f"Please download the dit_lora.safetensors file to: {lora_model_path}\n"
            
            # Check for text encoder models
            if os.path.exists(TEXT_ENCODERS_DIR):
                t5_models = [f for f in os.listdir(TEXT_ENCODERS_DIR) if "t5" in f.lower()]
                clip_models = [f for f in os.listdir(TEXT_ENCODERS_DIR) if "clip" in f.lower()]
                
                if not (t5_models or clip_models):
                    error_msg += "No suitable text encoder models found.\n"
                    error_msg += "UNO requires T5 or CLIP text encoder models.\n"
            
            error_msg += "See the README.md for download instructions."
            raise RuntimeError(error_msg)


class UNOImageGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("UNO_PIPELINE",),
                "prompt": ("STRING", {"default": "handsome woman in the city", "multiline": True}),
                "width": ("INT", {"default": 512, "min": 512, "max": 2048, "step": 16}),
                "height": ("INT", {"default": 512, "min": 512, "max": 2048, "step": 16}),
                "guidance_scale": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 5.0, "step": 0.1}),
                "num_inference_steps": ("INT", {"default": 25, "min": 1, "max": 50}),
                "seed": ("INT", {"default": -1}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 10}),
            },
            "optional": {
                "image_ref1": ("IMAGE",),
                "image_ref2": ("IMAGE",),
                "image_ref3": ("IMAGE",),
                "image_ref4": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "UNO"

    def generate(self, pipeline, prompt, width, height, guidance_scale, num_inference_steps, 
                 seed, batch_size, image_ref1=None, image_ref2=None, image_ref3=None, image_ref4=None): # Added batch_size here
        
        # --- Input Reference Image Processing (Same as before) ---
        ref_images = []
        for ref_img_input in [image_ref1, image_ref2, image_ref3, image_ref4]:
            # (Keep the corrected input processing logic from the previous response here)
            if ref_img_input is not None:
                if isinstance(ref_img_input, torch.Tensor):
                    if ref_img_input.dim() == 4 and ref_img_input.shape[0] == 1:
                        img_tensor_hwc = ref_img_input[0]
                        img_tensor_hwc = torch.clamp(img_tensor_hwc, 0.0, 1.0)
                        img_tensor_uint8 = (img_tensor_hwc * 255.0).byte()
                        img_np = img_tensor_uint8.cpu().numpy()
                        try:
                           ref_images.append(Image.fromarray(img_np))
                           # print(f"Successfully converted input tensor {ref_img_input.shape} to PIL Image.") # Optional print
                        except Exception as e:
                           print(f"Error converting input NumPy array (shape: {img_np.shape}, dtype: {img_np.dtype}) to PIL Image: {e}")
                           ref_images.append(None)
                    else:
                        # print(f"Warning: Received unexpected tensor shape {ref_img_input.shape} for reference image. Skipping conversion.") # Optional print
                        ref_images.append(None) 
                elif isinstance(ref_img_input, Image.Image):
                     ref_images.append(ref_img_input)
                     # print("Received input reference image as PIL Image.") # Optional print
                else:
                     # print(f"Warning: Received unexpected type {type(ref_img_input)} for reference image. Skipping.") # Optional print
                     ref_images.append(None)
            else:
                ref_images.append(None)
        valid_ref_images = ref_images # Assuming pipeline handles None inputs

        # --- Start: Batch Loop Logic ---
        output_images = [] # List to collect generated image tensors
        
        # Ensure batch_size is at least 1
        batch_size = max(1, int(batch_size)) 
        print(f"Generating batch of {batch_size} images...")

        for i in range(batch_size):
            current_seed = seed if seed == -1 else seed + i
            print(f"  Generating image {i+1}/{batch_size} with seed {current_seed}")

            # Generate single image using the pipeline
            try:
                image, file_path = pipeline.gradio_generate(
                    prompt=prompt,
                    width=width,
                    height=height,
                    guidance=guidance_scale,
                    num_steps=num_inference_steps,
                    seed=current_seed, # Use the incrementing seed
                    image_prompt1=valid_ref_images[0], 
                    image_prompt2=valid_ref_images[1],
                    image_prompt3=valid_ref_images[2],
                    image_prompt4=valid_ref_images[3]
                )
                
                # print(f"  Image type received from pipeline: {type(image)}") # Optional print
                
                # --- Output Image Processing (applied to each image in loop) ---
                if isinstance(image, Image.Image):
                    img_array = np.array(image).astype(np.uint8)
                    
                    if img_array.ndim == 3 and img_array.shape[-1] == 4:
                        img_array = img_array[:, :, :3]
                    elif img_array.ndim == 2:
                        img_array = np.stack((img_array,)*3, axis=-1)
                    
                    if img_array.shape[0] != height or img_array.shape[1] != width:
                         pil_img_resized = image.resize((width, height), Image.Resampling.LANCZOS)
                         img_array = np.array(pil_img_resized).astype(np.uint8)
                         if img_array.ndim == 3 and img_array.shape[-1] == 4:
                             img_array = img_array[:, :, :3]
                         elif img_array.ndim == 2:
                             img_array = np.stack((img_array,)*3, axis=-1)

                    img_tensor_uint8 = torch.from_numpy(img_array) # HWC uint8
                    img_tensor_float32 = img_tensor_uint8.float() / 255.0 # HWC float32
                    
                    # Add the processed tensor to our list (without batch dim yet)
                    output_images.append(img_tensor_float32) 
                    
                else:
                    print(f"  Error: pipeline did not return a PIL Image object for batch item {i+1}. Adding blank image.")
                    # Add a blank image tensor (HWC float32)
                    output_images.append(torch.zeros((height, width, 3), dtype=torch.float32))

            except Exception as e:
                print(f"  Error during image generation for batch item {i+1}: {e}")
                traceback.print_exc()
                # Add a blank image tensor (HWC float32) on error
                output_images.append(torch.zeros((height, width, 3), dtype=torch.float32))
        # --- End: Batch Loop Logic ---

        # Combine the list of image tensors into a single batch tensor
        if not output_images:
             print("Error: No images were generated.")
             # Return a single blank image tensor if the list is empty
             return (torch.zeros((1, height, width, 3), dtype=torch.float32),)
             
        # Stack along a new dimension (dimension 0) to create the batch
        final_batch_tensor = torch.stack(output_images, dim=0) # Shape: (batch_size, H, W, C)
        
        print(f"Finished generating batch. Returning tensor with shape: {final_batch_tensor.shape} and dtype: {final_batch_tensor.dtype}")
        return (final_batch_tensor,)


# Define node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "UNOModelLoader": UNOModelLoader,
    "UNOImageGenerator": UNOImageGenerator,
}

# Define display names for the ComfyUI interface
NODE_DISPLAY_NAME_MAPPINGS = {
    "UNOModelLoader": "UNO Model Loader",
    "UNOImageGenerator": "UNO Image Generator",
}