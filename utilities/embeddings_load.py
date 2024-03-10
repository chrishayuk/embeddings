import torch
from models.EmbeddingModel import EmbeddingModel

def load_embeddings_model(embeddings_filename, vocab_size, dimensions):
    # Initialize the custom embedding model
    model = EmbeddingModel(vocab_size, dimensions)

    # Load the saved embeddings from the file
    saved_embeddings = torch.load(embeddings_filename)
    
    # Ensure the 'weight' key exists in the saved embeddings dictionary
    if 'weight' not in saved_embeddings:
        raise KeyError("The saved embeddings file does not contain 'weight' key.")

    embeddings_tensor = saved_embeddings['weight']

    # Check if the dimensions match
    if embeddings_tensor.size() != (vocab_size, dimensions):
        raise ValueError(f"The dimensions of the loaded embeddings do not match the model's expected dimensions ({vocab_size}, {dimensions}).")

    # Assign the extracted embeddings tensor to the model's embedding layer
    model.embedding.weight.data = embeddings_tensor

    # put the model in eval mode
    model.eval()
    
    # return the model
    return model
