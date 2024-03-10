from transformers import AutoModel
from utilities.args_parser import parse_args

def print_model_layers(model):
    # loop through the modules
    for name, _ in model.named_modules():
        # print
        print(name)
    
def main(model_name):
    # Load the model
    model = AutoModel.from_pretrained(model_name)

    # Print the model layers
    print_model_layers(model)

if __name__ == "__main__":
    # parse the arguments
    args = parse_args()

    # execute
    main(args.model)
