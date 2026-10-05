# Fernando Pessoa GPT

A small GPT-style language model, written from scratch in PyTorch and trained on Fernando Pessoa's English poems to generate new verse in his style.

> Originally built in March–April 2024 as a university project at FEUP (University of Porto). Cleaned up and reorganised in October 2026.

## Problem

Most people know Pessoa for his Portuguese work, but he also wrote poetry in English (*Antinous*, *Inscriptions*, *Epithalamium*).
The goal was to understand how a transformer language model works by building one without any pretrained weights, and to see what it can learn from a single author with a very small corpus (~48k characters).

## Approach

- **Data**: Pessoa's English poems, in public domain ([`data/fernando_pessoa.txt`](data/fernando_pessoa.txt)). First 80% for training, last 20% for validation.
- **Model**: decoder-only transformer (masked multi-head self-attention, feed-forward, residual connections + layer norm), 4 blocks, 512 embedding size, context of 64 tokens, ~12.7M parameters.
- **Tokenizers**: two versions were compared:
  1. character level (70 tokens), in [`notebooks/01_char_level_gpt.ipynb`](notebooks/01_char_level_gpt.ipynb)
  2. byte-level BPE trained on the same text, in [`notebooks/02_bpe_tokenizer.ipynb`](notebooks/02_bpe_tokenizer.ipynb)
- **Training**: AdamW, learning rate 3e-4, batch size 128, dropout 0.2, on a GPU.

## Results

| Tokenizer | Vocab size | Training steps | Best val loss | Val perplexity |
|---|---|---|---|---|
| Character | 70 | 1,200 | 0.945 | 2.57 |
| BPE | ~3.5k | 1,000 | 2.43 | 11.4 |

The two losses are per token, and a BPE token covers several characters, so they can't be compared directly.
What matters is how each one behaves:

- The **character-level** model learned spelling, line breaks, punctuation and the archaic vocabulary of the poems ("thou", "'tis", "thy"). It starts to overfit after ~1,000 steps.
- The **BPE** model overfits almost right away. The training loss goes close to 0 while the validation loss stays above 2.4, and the generated text is mostly copied from the training poems. With this little data there aren't enough examples of each word piece.

Sample from the character-level model (500 characters, no prompt):

```
And mee a shall did compel
Thoughts of the night's awaiting, dreamed
Untimed are of thy glad burn.

«Ay, 'tis open my dight heath and paunches,
And whose course, trailing through the brow
Ganymede again pour at his feast
Would see our dual soul from death released
And recreated unto joy, fear, playing
With his eyes and feign the saw
```

## How to run

```bash
pip install -r requirements.txt

# train the character-level model (default settings, as in the notebook)
python src/train.py

# generate text from the best checkpoint
python src/generate.py --length 500
python src/generate.py --prompt "The rain" --length 300

# BPE version, with the settings used in the notebook
python src/train.py --tokenizer bpe --n-head 4 --epochs 10 --iters 100
python src/generate.py --tokenizer bpe
```

A GPU is recommended. Training on CPU works but takes a couple of hours.
The notebooks in `notebooks/` keep the outputs of the original 2024 runs.

## Project structure

```
data/         Pessoa's English poems (training text)
notebooks/    original experiments with outputs
src/          model, training and generation scripts
```

## Acknowledgements

The model code follows freeCodeCamp's [Create a Large Language Model from Scratch with Python](https://www.youtube.com/watch?v=UU1WVnMk4E8) course by Elliot Arledge, adapted to train on Pessoa's poems, with a BPE tokenizer comparison added.
