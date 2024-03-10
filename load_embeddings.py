import torch
from transformers import AutoTokenizer
from models.EmbeddingModel import EmbeddingModel
from utilities.args_parser import parse_args
from utilities.embeddings_load import load_embeddings_model
from utilities.embeddings_print import print_direct_embeddings

def main(tokenizer_name, embeddings_filename, dimensions, prompt):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Load the embedding model with pre-trained weights
    model = load_embeddings_model(embeddings_filename, tokenizer.vocab_size, dimensions)

    # Tokenize the input text
    tokens = tokenizer(prompt, return_tensors="pt")
    input_ids = tokens['input_ids']

    # make a forward pass
    outputs = model(input_ids)

    # Directly use the embeddings layer to get embeddings for the input_ids
    embeddings = outputs

    # Use the utility function to print direct embeddings
    print_direct_embeddings(tokenizer, embeddings, input_ids)

if __name__ == "__main__":
    # parse the arguments
    args = parse_args()

    # execute
    main(args.tokenizer, args.embeddings_file, args.dimensions, args.prompt)

