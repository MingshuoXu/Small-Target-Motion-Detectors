from ..model import * # Import all models


def instancing_model(model_name, device = 'cpu', model_para=None):
    """
    Instantiate a model object based on the given model name.

    Parameters:
        model_name (str): Name of the model to instantiate. If None, a GUI for model selection will be opened.
        model_para: Parameters for model instantiation (optional).

    Returns:
        BaseModel: The instantiated model object.
    """
    
    # Instantiate the model
    _model_name =  globals().get(model_name)
    if _model_name:
        model = _model_name()
    else:
        print(f"Class {model_name} not found.")

    # Process additional parameters if provided
    if model_para is not None:
        # Handle model parameters
        pass

    model.setup()
    model.to(device=device)  # Move the model to the specified device

    return model




