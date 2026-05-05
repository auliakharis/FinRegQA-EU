"""
Download Llama model to cache directory.
Run this on the LOGIN NODE (has internet access) before submitting sbatch.

Usage:
    python download_model.py
    python download_model.py --model meta-llama/Llama-3.1-8B-Instruct
    python download_model.py --cache_dir /scratch/$USER/hf_cache
"""

import argparse
import os
from huggingface_hub import snapshot_download

def main():
    parser = argparse.ArgumentParser(description="Download HF model to cache")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--cache_dir", default="/cluster/home/arakhmasari/hf_cache")
    parser.add_argument("--hf_token", default=None)
    args = parser.parse_args()

    token = args.hf_token or os.environ.get("HF_TOKEN")
    cache_dir = os.path.expandvars(args.cache_dir)

    if not token:
        print("ERROR: No HF token. Set HF_TOKEN env variable or use --hf_token")
        exit(1)

    print(f"Model:     {args.model}")
    print(f"Cache dir: {cache_dir}")
    print(f"Token:     {token[:8]}...")
    print()

    print("Downloading model files (this may take a while)...")
    snapshot_download(repo_id=args.model, token=token, cache_dir=cache_dir)
    print("Download done.")

    print(f"\nDownload complete. You can now run: sbatch batch.sh")

if __name__ == "__main__":
    main()

