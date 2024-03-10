from utilities.args_parser import parse_args
from transformers import AutoTokenizer, AutoModel

def main(tokenizer_name, model_name, prompt):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Load the pretrained model
    model = AutoModel.from_pretrained(model_name)

    # Tokenize the input text
    tokens = tokenizer(prompt, return_tensors="pt")
    input_ids = tokens['input_ids']

    # make a forward pass
    outputs = model(input_ids)

    # get the embeddings from the last hidden state
    embeddings = outputs.last_hidden_state

    # print the shape of the embeddings
    print("Embeddings tensor shape:", embeddings.shape)

if __name__ == "__main__":
    # parse the arguments
    args = parse_args()

    # execute
    main(args.tokenizer, args.model, args.prompt)
