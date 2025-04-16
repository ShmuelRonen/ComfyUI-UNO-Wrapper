import os
import json
from huggingface_hub import login

# Path to the config file
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

# Function to authenticate with HuggingFace
def setup_hf_auth():
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            token = config.get("hf_token", "")
            if token and token.strip():
                # Set the token as environment variable
                os.environ["HF_TOKEN"] = token.strip()
                # Log in to Hugging Face
                login(token=token.strip())
                print("Successfully authenticated with Hugging Face using token from config.json")
                return True
        except Exception as e:
            print(f"Error reading config.json or authenticating: {e}")
    
    return False

# Execute authentication when imported
setup_hf_auth()