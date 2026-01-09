import os
# Set the Hugging Face endpoint to a mirror site, uncomment the following line if needed
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="opengait/OpenGait",
    local_dir="output",
    resume_download=True,
    local_dir_use_symlinks=False
)