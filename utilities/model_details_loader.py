import json

def load_model_details(simple_model_name, path):
    # open the file
    with open(path, 'r') as details_file:
        # load the details
        model_details = json.load(details_file)
    
    # ensure we have the model name
    if simple_model_name not in model_details:
        raise ValueError(f"Model details for {simple_model_name} not found.")
    
    # return the config
    return model_details[simple_model_name]