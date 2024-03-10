import torch
from transformers import AutoTokenizer, AutoModel
from utilities.args_parser import parse_args

def main(tokenizer_name, model_name, embeddings_filename):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Load the pre-trained model
    model = AutoModel.from_pretrained(model_name)

    # Extract the embeddings layer
    embeddings = model.get_input_embeddings()

    # Print out the embeddings
    print(f"Extracted Embeddings Layer for {model_name}: {embeddings}")

    # Save the embeddings layer
    torch.save(embeddings.state_dict(), embeddings_filename)

if __name__ == "__main__":
    # parse the arguments
    args = parse_args()

    # execute
    main(args.tokenizer, args.model, args.embeddings_file)
