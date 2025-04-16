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
USER_LORA_DIR = os.path.join(MODELS_DIR, "loras")  # User LoRAs in loras directory
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
                # Add the user_lora option
                "user_lora": (cls.get_available_loras(), {"default": "None"}),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }
    
    RETURN_TYPES = ("UNO_PIPELINE",)
    FUNCTION = "load_model"
    CATEGORY = "UNO"

    @classmethod
    def get_available_loras(cls):
        """Get list of available LoRA files in the loras directory"""
        loras = ["None"]  # Start with "None" as the first option
        
        try:
            # Use ComfyUI's built-in folder_paths to find loras
            if hasattr(folder_paths, "get_filename_list"):
                lora_filenames = folder_paths.get_filename_list("loras")
                for filename in lora_filenames:
                    # Only include files with standard extensions
                    if filename.endswith(('.safetensors', '.pt', '.pth', '.ckpt')):
                        # Get filename without extension
                        name = os.path.splitext(filename)[0]
                        if name not in loras:  # Avoid duplicates with different extensions
                            loras.append(name)
            else:
                # Fallback method if folder_paths doesn't have the method
                if os.path.exists(USER_LORA_DIR):
                    for file in os.listdir(USER_LORA_DIR):
                        if file.endswith('.safetensors') or file.endswith('.pt') or file.endswith('.pth') or file.endswith('.ckpt'):
                            # Add the file without extension
                            name = os.path.splitext(file)[0]
                            if name not in loras:  # Avoid duplicates with different extensions
                                loras.append(name)
        except Exception as e:
            print(f"Error loading LoRA list: {e}")
        
        return loras

    def load_model(self, model_name, device, offload, lora_rank, user_lora, lora_strength):
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
            
            # NEW: Apply user LoRA if selected and not "None"
            if user_lora != "None":
                user_lora_path = None
                
                # Try to find the user LoRA file
                if hasattr(folder_paths, "get_full_path"):
                    for ext in ['.safetensors', '.pt', '.pth', '.ckpt']:
                        try:
                            potential_path = folder_paths.get_full_path("loras", f"{user_lora}{ext}")
                            if potential_path and os.path.exists(potential_path):
                                user_lora_path = potential_path
                                print(f"Found user LoRA: {user_lora_path}")
                                break
                        except:
                            continue
                
                # Fallback to manual search
                if user_lora_path is None:
                    for ext in ['.safetensors', '.pt', '.pth', '.ckpt']:
                        potential_path = os.path.join(USER_LORA_DIR, f"{user_lora}{ext}")
                        if os.path.exists(potential_path):
                            user_lora_path = potential_path
                            print(f"Found user LoRA via manual search: {user_lora_path}")
                            break
                
                # If we found a LoRA file, try to apply it
                if user_lora_path:
                    try:
                        # Try different methods that might be available
                        if hasattr(pipeline, "load_additional_lora"):
                            pipeline.load_additional_lora(user_lora_path, weight=lora_strength)
                            print(f"Applied user LoRA with load_additional_lora()")
                        elif hasattr(pipeline, "apply_lora"):
                            pipeline.apply_lora(user_lora_path, scale=lora_strength)
                            print(f"Applied user LoRA with apply_lora()")
                        elif hasattr(pipeline, "load_lora"):
                            pipeline.load_lora(user_lora_path, alpha=lora_strength)
                            print(f"Applied user LoRA with load_lora()")
                        else:
                            print(f"Warning: Unable to apply user LoRA - no suitable method found")
                    except Exception as e:
                        print(f"Error applying user LoRA: {e}")
                else:
                    print(f"Warning: Selected user LoRA '{user_lora}' not found")
            
            # Store user lora info for reference
            pipeline.user_lora_info = {
                "name": user_lora,
                "strength": lora_strength
            }
            
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
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 30}),
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
                 seed, batch_size, image_ref1=None, image_ref2=None, image_ref3=None, image_ref4=None):
        
        # --- Input Reference Image Processing (Same as before) ---
        ref_images = []
        for ref_img_input in [image_ref1, image_ref2, image_ref3, image_ref4]:
            if ref_img_input is not None:
                if isinstance(ref_img_input, torch.Tensor):
                    if ref_img_input.dim() == 4 and ref_img_input.shape[0] == 1:
                        img_tensor_hwc = ref_img_input[0]
                        img_tensor_hwc = torch.clamp(img_tensor_hwc, 0.0, 1.0)
                        img_tensor_uint8 = (img_tensor_hwc * 255.0).byte()
                        img_np = img_tensor_uint8.cpu().numpy()
                        try:
                           ref_images.append(Image.fromarray(img_np))
                        except Exception as e:
                           print(f"Error converting input NumPy array (shape: {img_np.shape}, dtype: {img_np.dtype}) to PIL Image: {e}")
                           ref_images.append(None)
                    else:
                        ref_images.append(None) 
                elif isinstance(ref_img_input, Image.Image):
                     ref_images.append(ref_img_input)
                else:
                     ref_images.append(None)
            else:
                ref_images.append(None)
        valid_ref_images = ref_images

        # NEW: Log if user LoRA is being used
        if hasattr(pipeline, "user_lora_info") and pipeline.user_lora_info["name"] != "None":
            lora_info = pipeline.user_lora_info
            print(f"Generating with user LoRA: {lora_info['name']} (strength: {lora_info['strength']})")

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
                
                # --- Output Image Processing (applied to each image in loop) ---
                if isinstance(image, Image.Image):
                    img_array = np.array(image).astype(np.uint8)
                    
                    if img_array.ndim == 3 and img_array.shape[-1] == 4:
                        img_array = img_array[:, :, :3]
                    elif img_array.ndim == 2:
                        img_array = np.stack((img_array,)*3, axis=-1)
                    
                    if img_array.shape[0] != height or img_array.shape[1] != width:
                         pil_img_resized = image.resize((width, height), Image.LANCZOS)
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