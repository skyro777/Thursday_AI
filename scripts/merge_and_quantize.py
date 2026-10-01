#!/usr/bin/env python3
"""
merge_and_quantize.py — Post-training script.

After QLoRA fine-tuning produces a LoRA adapter, this script:
  1. Loads the base model + adapter
  2. Merges the adapter into the base weights
  3. Saves the merged model
  4. (Optionally) converts to GGUF Q4_K_M using llama.cpp's convert script
  5. (Optionally) pushes to HuggingFace Hub

Typical usage on Kaggle (after training cell completes):

    python scripts/merge_and_quantize.py \\
        --base Qwen/Qwen2.5-3B-Instruct \\
        --adapter ./thursday-ai-lora \\
        --out ./thursday-ai-merged \\
        --gguf ./thursday-ai-v0.1-Q4_K_M.gguf \\
        --quant Q4_K_M

If running on Kaggle without a GPU, skip --gguf (do that locally with llama.cpp).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def merge(base_model: str, adapter_path: str, out_dir: str) -> None:
    """Merge LoRA adapter into base model and save to out_dir."""
    print(f"[merge] loading base {base_model} + adapter {adapter_path}", file=sys.stderr)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        import torch
    except ImportError as e:
        print(f"ERROR: missing dependency: {e}", file=sys.stderr)
        print("Install with: pip install transformers peft torch", file=sys.stderr)
        sys.exit(2)

    # Load base in FP16 (CPU if no GPU, but merge is fast either way)
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    print(f"[merge] device_map={device_map}", file=sys.stderr)
    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float16, device_map=device_map,
    )
    tok = AutoTokenizer.from_pretrained(base_model)

    print("[merge] attaching adapter", file=sys.stderr)
    model = PeftModel.from_pretrained(base, adapter_path)

    print("[merge] merging adapter weights into base", file=sys.stderr)
    model = model.merge_and_unload()

    print(f"[merge] saving merged model to {out_dir}", file=sys.stderr)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    print(f"[merge] ✅ saved to {out_dir}", file=sys.stderr)


def convert_to_gguf(merged_dir: str, gguf_path: str, quant: str = "Q4_K_M") -> None:
    """Convert merged HF model to GGUF using llama.cpp's convert_hf_to_gguf.py.

    Requires llama.cpp cloned at LLAMA_CPP_PATH (env var or ./llama.cpp).
    """
    llama_cpp_path = os.environ.get("LLAMA_CPP_PATH", "./llama.cpp")
    if not Path(llama_cpp_path).exists():
        print(f"ERROR: llama.cpp not found at {llama_cpp_path}.", file=sys.stderr)
        print("Clone it: git clone https://github.com/ggml-org/llama.cpp", file=sys.stderr)
        print("Then: LLAMA_CPP_PATH=/path/to/llama.cpp python scripts/merge_and_quantize.py ...", file=sys.stderr)
        sys.exit(2)

    convert_script = Path(llama_cpp_path) / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        print(f"ERROR: {convert_script} not found.", file=sys.stderr)
        sys.exit(2)

    print(f"[gguf] converting {merged_dir} → {gguf_path} (f16 first)", file=sys.stderr)
    f16_path = str(Path(gguf_path).with_suffix(".f16.gguf"))
    subprocess.run(
        ["python", str(convert_script), merged_dir, "--outtype", "f16", "--outfile", f16_path],
        check=True,
    )

    print(f"[gguf] quantizing to {quant}", file=sys.stderr)
    quantize_bin = Path(llama_cpp_path) / "build" / "bin" / "llama-quantize"
    if not quantize_bin.exists():
        # try alternate path
        quantize_bin = Path(llama_cpp_path) / "llama-quantize"
    if not quantize_bin.exists():
        print(f"ERROR: llama-quantize binary not found in {llama_cpp_path}.", file=sys.stderr)
        print("Build llama.cpp: cd llama.cpp && make", file=sys.stderr)
        sys.exit(2)
    subprocess.run([str(quantize_bin), f16_path, gguf_path, quant], check=True)

    # Remove f16 intermediate
    Path(f16_path).unlink(missing_ok=True)
    size_mb = Path(gguf_path).stat().st_size / (1024 * 1024)
    print(f"[gguf] ✅ wrote {gguf_path} ({size_mb:.1f} MB)", file=sys.stderr)


def push_to_hub(local_dir: str, repo_id: str, token: str | None = None) -> None:
    """Upload the merged model (or GGUF) to HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("ERROR: pip install huggingface_hub", file=sys.stderr)
        sys.exit(2)
    api = HfApi(token=token)
    create_repo(repo_id, exist_ok=True, token=token)
    api.upload_folder(folder_path=local_dir, repo_id=repo_id, token=token)
    print(f"[hub] ✅ uploaded {local_dir} → {repo_id}", file=sys.stderr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct", help="HF id of base model")
    p.add_argument("--adapter", required=True, help="Path to LoRA adapter dir")
    p.add_argument("--out", required=True, help="Dir to save merged HF model")
    p.add_argument("--gguf", default=None, help="Optional path for the final GGUF file")
    p.add_argument("--quant", default="Q4_K_M", help="GGUF quantization type")
    p.add_argument("--push-hub", default=None, help="HF Hub repo id to push the merged model to")
    p.add_argument("--push-gguf-hub", default=None, help="HF Hub repo id to push the GGUF to")
    p.add_argument("--token", default=None, help="HF token (or set HF_TOKEN env var)")
    args = p.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")

    merge(args.base, args.adapter, args.out)

    if args.gguf:
        convert_to_gguf(args.out, args.gguf, args.quant)

    if args.push_hub:
        push_to_hub(args.out, args.push_hub, token=token)

    if args.push_gguf_hub and args.gguf:
        # Upload the single GGUF file
        try:
            from huggingface_hub import HfApi, create_repo
        except ImportError:
            print("ERROR: pip install huggingface_hub", file=sys.stderr)
            sys.exit(2)
        api = HfApi(token=token)
        create_repo(args.push_gguf_hub, exist_ok=True, token=token, repo_type="model")
        api.upload_file(path_or_fileobj=args.gguf, path_in_repo=Path(args.gguf).name, repo_id=args.push_gguf_hub, token=token)
        print(f"[hub] ✅ uploaded {args.gguf} → {args.push_gguf_hub}", file=sys.stderr)


if __name__ == "__main__":
    main()
