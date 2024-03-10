import torch

def print_direct_embeddings(tokenizer, embeddings, input_ids, num_start_elements=3, last_column_width=50):
    # Dynamically generate encoded tokens and their corresponding textual representation
    encoded_tokens = input_ids.squeeze().tolist()  # Assuming batch size of 1, squeeze if necessary
    token_texts = [tokenizer.decode([token_id], skip_special_tokens=True) for token_id in encoded_tokens]

    # Print the embeddings
    print_embeddings(token_texts, encoded_tokens, embeddings, num_start_elements, last_column_width)
    
def print_token_embeddings(tokenizer, model, text, num_start_elements=3, last_column_width=50):
    # Tokenize the input text
    tokens = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    input_ids = tokens['input_ids']

    # Generate embeddings using the model
    with torch.no_grad():
        model_output = model(input_ids)

    # Check if the model output includes 'last_hidden_state' or use the output directly
    if hasattr(model_output, 'last_hidden_state'):
        embeddings_tensor = model_output.last_hidden_state
    else:
        embeddings_tensor = model_output

    # Dynamically generate encoded tokens and their corresponding textual representation
    encoded_tokens = input_ids[0].tolist()  # Assuming batch size of 1
    token_texts = [tokenizer.decode([token_id], skip_special_tokens=True) for token_id in encoded_tokens]

    # Print the embeddings
    print_embeddings(token_texts, encoded_tokens, embeddings_tensor, num_start_elements, last_column_width)

def print_embeddings(token_texts, encoded_tokens, embeddings_tensor, num_start_elements, last_column_width):
    # Prepare and print the header
    header = f"{'Token':<10} | {'ID':<8} | {'Embedding (First ' + str(num_start_elements) + ', ..., Last)':<{last_column_width}}"
    print(f"{'='*90}\n{'Selected Truncated Embeddings for Specific Tokens':^90}\n{'='*90}")
    print(header)
    print(f"{'-'*10}-+-{'-'*8}-+{'-'*last_column_width}")

    # Iterate over each token and its embedding
    for idx, (token_id, token_text) in enumerate(zip(encoded_tokens, token_texts)):
        embedding = embeddings_tensor[0, idx, :].detach().numpy()  # Assuming batch size of 1
        start_elements = ', '.join([f"{value:.4f}" for value in embedding[:num_start_elements]])
        last_element = f"{embedding[-1]:.4f}"
        embedding_str = f"{start_elements}, ..., {last_element}"
        print(f"{token_text:<10} | {token_id:<8} | {embedding_str:<{last_column_width}}")

    # Print the table footer
    print(f"{'='*90}")
