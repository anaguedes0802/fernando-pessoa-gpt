import os
import argparse

import torch

from model import GPTLanguageModel
from train import CKPT_DIR, CharTokenizer, BPETokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", choices=["char", "bpe"], default="char")
    parser.add_argument("--prompt", default="", help="text to start from (empty = start from scratch)")
    parser.add_argument("--length", type=int, default=500, help="number of tokens to generate")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(os.path.join(CKPT_DIR, f"pessoa_{args.tokenizer}.pt"), map_location=device)
    cfg = ckpt["args"]

    if args.tokenizer == "char":
        tokenizer = CharTokenizer(ckpt["chars"])
    else:
        tokenizer = BPETokenizer.from_file(os.path.join(CKPT_DIR, "bpe_tokenizer.json"))

    model = GPTLanguageModel(ckpt["vocab_size"], cfg["n_embd"], cfg["n_head"], cfg["n_layer"],
                             cfg["block_size"], cfg["dropout"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    if args.prompt:
        context = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)  # index 0 is "\n" for the char vocab

    out = model.generate(context, max_new_tokens=args.length)
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
