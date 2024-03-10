from transformers import AutoTokenizer
from utilities.args_parser import parse_args
from utilities.tokens import print_tokens


def main(tokenizer_name, prompt):
    # Load the tokenizer and add pad token if necessary
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Tokenize the input text
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # Print tokens and their IDs
    print_tokens(tokenizer, input_ids)

if __name__ == "__main__":
    # parse the arguments
    args = parse_args()

    # execute
    main(args.tokenizer, args.prompt)

