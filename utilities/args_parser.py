import argparse

def parse_args():
    # setup the parser
    parser = argparse.ArgumentParser(description='get model details')

    # parse model name and config path
    parser.add_argument("--tokenizer", required=False, default="google/gemma-2b", help='model tokenizer e.g. google/gemma-2b')
    parser.add_argument("--model", required=False, default="google/gemma-2b", help='model e.g. google/gemma-2b')
    parser.add_argument("--prompt", required=False, default="Who is Ada Lovelace?", help="Prompt to execute")
    parser.add_argument("--embeddings_file", required=False, default="./output/gemma_2b_embeddings_layer.pth", help="embeddings filename")
    parser.add_argument("--dimensions", required=False, default=2048, help="number of dimensions")

    # parse the arguments
    args = parser.parse_args()

    # return the args
    return args