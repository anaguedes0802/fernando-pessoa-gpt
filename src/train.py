import os
import argparse

import torch

from model import GPTLanguageModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(ROOT_DIR, "data", "fernando_pessoa.txt")
CKPT_DIR = os.path.join(ROOT_DIR, "checkpoints")


class CharTokenizer:
    def __init__(self, chars):
        self.chars = chars
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    @classmethod
    def from_text(cls, text):
        return cls(sorted(set(text)))

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


class BPETokenizer:
    def __init__(self, tok):
        self.tok = tok

    @classmethod
    def from_text(cls, text, vocab_size=50000):
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel()
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
        tok.train_from_iterator([text], trainer=trainer)
        return cls(tok)

    @classmethod
    def from_file(cls, path):
        from tokenizers import Tokenizer
        return cls(Tokenizer.from_file(path))

    @property
    def vocab_size(self):
        return self.tok.get_vocab_size()

    def encode(self, s):
        return self.tok.encode(s).ids

    def decode(self, ids):
        return self.tok.decode(ids)


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, data, args, device):
    model.eval()
    losses = torch.zeros(args.iters)
    for k in range(args.iters):
        x, y = get_batch(data, args.block_size, args.batch_size, device)
        _, loss = model(x, y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", choices=["char", "bpe"], default="char")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--iters", type=int, default=200, help="training steps per epoch (also used as number of eval batches)")
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-embd", type=int, default=512)
    parser.add_argument("--n-head", type=int, default=6)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    os.makedirs(CKPT_DIR, exist_ok=True)
    if args.tokenizer == "char":
        tokenizer = CharTokenizer.from_text(text)
    else:
        tokenizer = BPETokenizer.from_text(text)
        tokenizer.tok.save(os.path.join(CKPT_DIR, "bpe_tokenizer.json"))
    print("vocab size:", tokenizer.vocab_size)

    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.8 * len(data))  # 80/20 split, no shuffling so the val set is the end of the book
    train_data, val_data = data[:n], data[n:]
    print("tokens:", len(data))

    model = GPTLanguageModel(tokenizer.vocab_size, args.n_embd, args.n_head, args.n_layer,
                             args.block_size, args.dropout).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    ckpt_path = os.path.join(CKPT_DIR, f"pessoa_{args.tokenizer}.pt")
    best_val = float("inf")

    for epoch in range(args.epochs):
        train_losses = []
        for _ in range(args.iters):
            xb, yb = get_batch(train_data, args.block_size, args.batch_size, device)
            _, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = sum(train_losses) / len(train_losses)
        val_loss = estimate_loss(model, val_data, args, device)
        ppl = torch.exp(torch.tensor(val_loss)).item()
        print(f"epoch {epoch}: train loss {train_loss:.4f}, val loss {val_loss:.4f}, val perplexity {ppl:.4f}")

        # keep the checkpoint with the best validation loss
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "state_dict": model.state_dict(),
                "args": vars(args),
                "vocab_size": tokenizer.vocab_size,
                "chars": tokenizer.chars if args.tokenizer == "char" else None,
            }, ckpt_path)

    print(f"best val loss {best_val:.4f}, saved to {ckpt_path}")


if __name__ == "__main__":
    main()
