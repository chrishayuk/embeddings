from transformers import AutoTokenizer
from utilities.args_parser import parse_args
from utilities.embeddings_load import load_embeddings_model
from utilities.embeddings_similarity import find_and_deduplicate_embeddings
from utilities.embeddings_visualization import plot_embeddings_2d


def main(tokenizer_name, embeddings_filename, dimensions, prompt, threshold=0.5, show_lines=False):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Load the embedding model with pre-trained weights
    model = load_embeddings_model(embeddings_filename, tokenizer.vocab_size, dimensions)

    # Tokenize the input text
    tokens = tokenizer(prompt, return_tensors="pt")
    input_ids = tokens['input_ids']

    # find and deduplicate tokens and embeddings
    embeddings = model(input_ids).detach().numpy()
    all_embeddings = model.embedding.weight.detach().numpy()
    combined_tokens, combined_embeddings, connections = find_and_deduplicate_embeddings(input_ids, embeddings, all_embeddings, tokenizer, threshold)
    
    # plot the embeddings
    plot_embeddings_2d(tokenizer, combined_tokens, combined_embeddings, all_embeddings, show_lines, threshold)

if __name__ == "__main__":
    # parse the arguments
    args = parse_args()

    # execute
    main(args.tokenizer, args.embeddings_file, args.dimensions, args.prompt, 0.75, True)

