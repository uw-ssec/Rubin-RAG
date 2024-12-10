from __future__ import annotations

import os
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import requests
from langchain.schema import Document
from transformers import AutoModelForCausalLM

HOME_DIR = Path.home()
# Nebari shared directory
# default to this
CACHE_DIR = HOME_DIR / "shared" / "scipy" / "gen-ai-copilot-with-rag"
if not CACHE_DIR.exists():
    # This is for local development
    CACHE_DIR = HOME_DIR / ".cache"

DEFAULT_TUTORIAL_CACHE = CACHE_DIR / "ssec_tutorials"
TUTORIAL_CACHE = Path(os.environ.get("SSEC_TUTORIALS_CACHE", DEFAULT_TUTORIAL_CACHE)).expanduser()
TUTORIAL_CACHE.mkdir(parents=True, exist_ok=True)

OLMO_MODEL_FILE = os.environ.get("OLMO_MODEL_FILE", "OLMo-7B-Instruct-Q4_K_M.gguf")
OLMO_MODEL = TUTORIAL_CACHE / OLMO_MODEL_FILE

OLMO_2_MODEL_FILE = os.environ.get("OLMO_2_MODEL_FILE", "OLMo-2-1124-7B-Instruct")
OLMO_2_MODEL = TUTORIAL_CACHE / OLMO_2_MODEL_FILE

# Configure Hugging Face to use the tutorial cache
os.environ["HF_HOME"] = str(TUTORIAL_CACHE)

# Set the URL for tutorials data assets
TUTORIALS_DATA_URL = "https://github.com/uw-ssec/tutorials-data/releases/download/scipy-2024/"


def download_olmo_model(model_file: str | None = None, force=False) -> Path:
    """Download the OLMO model from the Hugging Face model hub.

    Parameters
    ----------
    model_file : str | None, optional
        The name of the model file to download, by default None
    force : bool, optional
        Whether to force the download even if the file already exists, by default False

    Returns
    -------
    pathlib.Path
        The path to the downloaded model file
    """

    if not OLMO_MODEL.exists() or force:
        if model_file is None:
            model_file = OLMO_MODEL_FILE
            olmo_model = OLMO_MODEL
        else:
            olmo_model = TUTORIAL_CACHE / model_file
        olmo_model_url = (
            f"https://huggingface.co/ssec-uw/OLMo-7B-Instruct-GGUF/resolve/main/{model_file}"
        )
        urlretrieve(olmo_model_url, olmo_model)
        return olmo_model

    print(f"Model already exists at {OLMO_MODEL}")
    return OLMO_MODEL

def download_olmo_2_model(force=False) -> Path:
    """Download the OLMO 2 model using Hugging Face Transformers and cache locally.

    Parameters
    ----------
    force : bool, optional
        Whether to force the download even if the file already exists, by default False

    Returns
    -------
    pathlib.Path
        The path to the downloaded model directory
    """

    if OLMO_2_MODEL.exists() and not force:
        print(f"Model already exists at {OLMO_2_MODEL}")
        return OLMO_2_MODEL

    # Download model using Hugging Face's AutoModelForCausalLM
    print(f"Downloading OLMO 2 model: {OLMO_2_MODEL_FILE}...")
    model = AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-1124-7B-Instruct")

    print(f"Model cached at: {TUTORIAL_CACHE}")

    return OLMO_2_MODEL