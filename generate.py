#!/usr/bin/env python3
"""
Generate text from a trained DSTT-T model.

Loads a checkpoint, takes a prompt, and generates text using
autoregressive sampling with configurable temperature, top-k,
top-p, and repetition penalty.

Usage:
    # Interactive generation
    python generate.py --checkpoint checkpoints/best.pt --prompt "ROMEO:"

    # Greedy decoding
    python generate.py --checkpoint checkpoints/best.pt --temperature 0

    # Creative sampling
    python generate.py --checkpoint checkpoints/best.pt \
        --temperature 1.0 --top_k 50 --top_p 0.95

    # Long generation
    python generate.py --checkpoint checkpoints/best.pt \
        --max_tokens 1000 --prompt "Once upon a time"
"""

import argparse
import os
import sys

import torch

from dstt import DSTTConfig, DSTTTransformer
from dstt.generate import generate_text
from dstt.tokenizer import CharTokenizer, get_tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Generate text from a trained DSTT-T model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default="\n",
                        help="Text prompt to start generation")
    parser.add_argument("--max_tokens", type=int, default=500,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (0 = greedy)")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling (0 = disabled)")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Nucleus sampling threshold")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                        help="Repetition penalty (1.0 = disabled)")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of samples to generate")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]

    # Build model
    model = DSTTTransformer(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Model loaded: {config.n_layers}L, d={config.d_model}, "
          f"vocab={config.vocab_size}")

    # Load tokenizer
    data_dir = os.path.dirname(args.checkpoint)
    tok_path = os.path.join("data", "tokenizer.json")
    if os.path.exists(tok_path):
        tokenizer = CharTokenizer.load(tok_path)
        print(f"Tokenizer: char-level (vocab={tokenizer.vocab_size})")
    else:
        try:
            tokenizer = get_tokenizer("gpt2")
            print("Tokenizer: GPT-2 BPE")
        except ImportError:
            print("ERROR: No tokenizer found. Put tokenizer.json in data/ "
                  "or install tiktoken.")
            sys.exit(1)

    # Generate
    print(f"\nPrompt: {repr(args.prompt)}")
    print(f"Temperature: {args.temperature}, Top-k: {args.top_k}, "
          f"Top-p: {args.top_p}")
    print("=" * 60)

    for i in range(args.num_samples):
        if args.num_samples > 1:
            print(f"\n--- Sample {i + 1} ---")

        text = generate_text(
            model,
            tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            device=device,
        )
        print(text)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
