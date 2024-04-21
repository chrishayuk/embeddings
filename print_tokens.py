import tiktoken
import torch
from transformers import AutoTokenizer
from utilities.args_parser import parse_args
from utilities.tokens import print_tokens

def main(tokenizer_name, prompt):

    if tokenizer_name.lower() == "gpt-4":
        # Set the tokenizer as GPT-4 using tiktoken
        tokenizer = tiktoken.encoding_for_model("gpt-4")
        input_ids = tokenizer.encode(prompt)
        input_ids_tensor = torch.tensor([input_ids])
    else:
        # Load the tokenizer using transformers and tokenize the input text
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids_tensor = inputs["input_ids"]

    # Print tokens and their IDs
    print_tokens(tokenizer, input_ids_tensor, tokenizer_name)

if __name__ == "__main__":
    # parse the arguments
    args = parse_args()

    # execute
    main(args.tokenizer, args.prompt)

