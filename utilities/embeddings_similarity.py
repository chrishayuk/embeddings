import numpy as np

# embeddings_similarity.py

def find_and_deduplicate_embeddings(input_ids, embeddings, all_embeddings, tokenizer, threshold=0.5):
    # Find closest tokens for dimensionality reduction
    token_texts = [tokenizer.decode([token_id]) for token_id in input_ids[0].tolist()]

    additional_nodes = set()
    connections = []

    for idx, token_embedding in enumerate(embeddings[0]):
        # Only the closest 5 tokens
        closest_tokens = find_closest_tokens(token_embedding, all_embeddings, threshold, 5)  

        # ensure tokens are greater than the threshold
        for close_idx, similarity in closest_tokens:
            if similarity >= threshold:
                additional_nodes.add(tokenizer.decode([close_idx]))
                connections.append((idx, close_idx, similarity))

    # Deduplicate tokens and combine embeddings
    combined_tokens, combined_embeddings = deduplicate_embeddings(token_texts, additional_nodes, embeddings, all_embeddings, tokenizer)
    
    return combined_tokens, combined_embeddings, connections


def find_closest_tokens(embedding, all_embeddings, threshold=0.5, num_tokens=10):
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(embedding.reshape(1, -1), all_embeddings)
    closest_idxs = np.argsort(similarities[0])[::-1][1:]  # Skip the first one (itself)
    closest = [(idx, similarities[0][idx]) for idx in closest_idxs if similarities[0][idx] > threshold]
    return closest[:num_tokens]

def deduplicate_embeddings(token_texts, additional_nodes, embeddings, all_embeddings, tokenizer):
    combined_tokens = token_texts.copy()  # Start with the original tokens
    for token in additional_nodes:
        if token not in token_texts:
            combined_tokens.append(token)

    seen_tokens = set()
    unique_combined_tokens = [x for x in combined_tokens if not (x in seen_tokens or seen_tokens.add(x))]
    
    combined_embeddings = np.vstack([embeddings[0]] + [all_embeddings[tokenizer.encode(token, add_special_tokens=False)] for token in unique_combined_tokens if token in additional_nodes])

    return unique_combined_tokens, combined_embeddings
