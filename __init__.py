"""
ComfyUI-UNO-Wrapper
A ComfyUI extension for integrating the UNO-FLUX model
"""

# Import the HF authentication setup
from .hf_auth import setup_hf_auth

# Try to authenticate if config.json exists
setup_hf_auth()

from .uno_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]