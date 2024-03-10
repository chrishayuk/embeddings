import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utilities.embeddings_similarity import find_closest_tokens

def plot_embeddings_2d(tokenizer, combined_tokens, combined_embeddings, all_embeddings, show_lines, threshold):
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(combined_embeddings)

    # Visualization
    plt.figure(figsize=(12, 8))
    original_token_color = 'red'
    other_token_color = 'blue'
    original_indices = set(range(len(combined_tokens)))
    plotted_tokens = set()

    # Replace spaces with underscores for visualization purposes
    combined_tokens = [token.replace(' ', '_') for token in combined_tokens]

    # Map the indices from all_embeddings to the reduced_embeddings
    index_map = {idx: i for i, token in enumerate(combined_tokens) for idx in tokenizer.encode(token, add_special_tokens=False)}

    for i, token in enumerate(combined_tokens):
        if token not in plotted_tokens:
            color = original_token_color if i in original_indices else other_token_color
            plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1], color=color)
            plt.text(reduced_embeddings[i, 0], reduced_embeddings[i, 1], token, fontsize=9)
            plotted_tokens.add(token)

        if show_lines:
            closest_tokens = find_closest_tokens(combined_embeddings[i], all_embeddings, threshold, num_tokens=5)
            for sim_token_idx, sim_score in closest_tokens:
                mapped_idx = index_map.get(sim_token_idx, None)
                if mapped_idx is not None and mapped_idx != i:
                    line_alpha = min((sim_score - threshold) / (1 - threshold), 1) if sim_score > threshold else 0
                    if line_alpha > 0:
                        plt.plot([reduced_embeddings[i, 0], reduced_embeddings[mapped_idx, 0]],
                                 [reduced_embeddings[i, 1], reduced_embeddings[mapped_idx, 1]],
                                 'r-', linewidth=0.5, alpha=line_alpha)

    plt.xlabel('PCA Dimension 1')
    plt.ylabel('PCA Dimension 2')
    plt.title('2D Projection of Token Embeddings and Closest Tokens')
    plt.show()
