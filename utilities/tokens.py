def print_tokens(tokenizer, input_ids_tensor, last_column_width=50):
    # Convert input ids to tokens
    token_texts = [tokenizer.decode([token_id], skip_special_tokens=True) for token_id in input_ids_tensor[0]]

    # Prepare and print the header
    header = f"{'Token':<10} | {'ID':<8}"
    print(f"{'='*30}\n{'Tokens and their IDs':^30}\n{'='*30}")
    print(header)
    print(f"{'-'*10}-+-{'-'*8}")

    # Iterate over each token and its ID
    for idx, token_id in enumerate(input_ids_tensor[0]):
        token_text = token_texts[idx]
        print(f"{token_text:<10} | {token_id:<8}")

    # Print the table footer
    print(f"{'='*30}")