import torch
from transformers import AutoModel, AutoTokenizer
from utilities.args_parser import parse_args
from utilities.embeddings_print import print_direct_embeddings

def main(tokenizer_name, model_name, prompt):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Load the model configuration and model
    model = AutoModel.from_pretrained(model_name)

    # Tokenize the input text
    tokens = tokenizer(prompt, return_tensors="pt")
    input_ids = tokens['input_ids']

    # Directly use the embeddings layer to get embeddings for the input_ids
    with torch.no_grad():
        embeddings = model.get_input_embeddings()(input_ids)

    # Use the utility function to print direct embeddings
    print_direct_embeddings(tokenizer, embeddings, input_ids)

if __name__ == "__main__":
    # parse the arguments
    args = parse_args()

    # execute
    main(args.tokenizer, args.model, args.prompt)

