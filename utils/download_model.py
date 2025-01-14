from __future__ import annotations

import os
from pathlib import Path
from urllib.request import urlretrieve
from transformers import AutoModelForCausalLM

HOME_DIR = Path.home()
# Nebari shared directory
CACHE_DIR = HOME_DIR / "shared" / "scipy" / "gen-ai-copilot-with-rag"
if not CACHE_DIR.exists():
    CACHE_DIR = HOME_DIR / ".cache"

DEFAULT_TUTORIAL_CACHE = CACHE_DIR / "ssec_tutorials"
TUTORIAL_CACHE = Path(os.environ.get("SSEC_TUTORIALS_CACHE", DEFAULT_TUTORIAL_CACHE)).expanduser()
TUTORIAL_CACHE.mkdir(parents=True, exist_ok=True)

# Environment variables
OLMO_MODEL_FILE = os.environ.get("OLMO_MODEL_FILE", "OLMo-7B-Instruct-Q4_K_M.gguf")
OLMO_2_MODEL_FILE = os.environ.get("OLMO_2_MODEL_FILE", "olmo-2-1124-7B-instruct-Q4_K_M.gguf")

# Configure Hugging Face to use the tutorial cache
os.environ["HF_HOME"] = str(TUTORIAL_CACHE)

def download_model(
    model_name: str,
    model_file: str,
    source: str,
    force: bool = False
) -> Path:
    """
    Generalized function to download a model and store it in the cache directory.

    Parameters
    ----------
    model_name : str
        Name of the model for display purposes.
    model_file : str
        File name or identifier for the model.
    source : str
        URL or identifier (Hugging Face model ID).
    force : bool, optional
        Whether to force download even if the model already exists, by default False.

    Returns
    -------
    pathlib.Path
        Path to the downloaded model.
    """
    model_path = TUTORIAL_CACHE / model_file

    if model_path.exists() and not force:
        print(f"{model_name} model already exists at {model_path}")
        return model_path

    print(f"Downloading {model_name} model...")

    if source.startswith("http"):
        # Download from URL
        urlretrieve(source, model_path)
    else:
        # Download using Hugging Face's transformers
        AutoModelForCausalLM.from_pretrained(source, cache_dir=TUTORIAL_CACHE)

    print(f"{model_name} model cached at {model_path}")
    return model_path


def download_olmo_model(force: bool = False) -> Path:
    """
    Wrapper for downloading the OLMO model from a URL.
    """
    url = f"https://huggingface.co/ssec-uw/OLMo-7B-Instruct-GGUF/resolve/main/{OLMO_MODEL_FILE}"
    return download_model("OLMO", OLMO_MODEL_FILE, url, force)


def download_olmo_2_model(force: bool = False) -> Path:
    """
    Wrapper for downloading the OLMO 2 model from Hugging Face.
    """
    url = f"https://huggingface.co/allenai/OLMo-2-1124-7B-Instruct-GGUF/resolve/main/{OLMO_2_MODEL_FILE}"
    return download_model("OLMO 2", OLMO_2_MODEL_FILE, url,force)

