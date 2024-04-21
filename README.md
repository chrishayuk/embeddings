# Introduction
These scripts are really about exploring embeddings, specifically input embeddings.

## Print Layers
If we need to print out the layers of an LLM, just call (by default it'll choose Gemma 2B)

```bash
python print_layers.py
```

and for other models such as mistral

```bash
python print_layers.py --model "mistralai/Mistral-7B-v0.1"
```

or llama 7b

```bash
python print_layers.py --model "meta-llama/Llama-2-7b-hf"
```

larger models such as llama-2-70b chat

```bash
python print_layers.py --model "meta-llama/Llama-2-70b-chat-hf" 
```

## Print Tokens
If we need to print out the tokens of an LLM, just call (by default it'll choose Gemma 2B) and the phrase "Who is Ada Lovelace?"

```bash
python print_layers.py
```

and for other models such as mistral

```bash
python print_tokens.py --tokenizer "mistralai/Mistral-7B-v0.1" --prompt "Who is Kitty Purry?"
```

```bash
python extract_embeddings.py --tokenizer "meta-llama/Meta-Llama-3-8b-Instruct" --model "meta-llama/Meta-Llama-3-8b-Instruct" --embeddings_file "./output/llama3_8b_embeddings_layer.pth" --dimensions 4096
```

```bash
python visualize_cosine_similarity.py --tokenizer "meta-llama/Meta-Llama-3-8b-Instruct" --model "meta-llama/Meta-Llama-3-8b-Instruct" --embeddings_file "./output/llama3_8b_embeddings_layer.pth" --dimensions 4096 --prompt "Sit Sat Mat Bat Hat Cat Nap Kit Kat Dog Fish Tree Math London Paris Rio Berlin Sydney Moscow Red Blue Green Black White for while print loop"
```