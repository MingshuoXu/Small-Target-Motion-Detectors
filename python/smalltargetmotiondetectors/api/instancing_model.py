import os
import sys
from ..model import * # Import all models


def instancing_model(modelName, device = 'cpu', modelPara=None):
    """
    Instantiate a model object based on the given model name.

    Parameters:
        modelName (str): Name of the model to instantiate. If None, a GUI for model selection will be opened.
        modelPara: Parameters for model instantiation (optional).

    Returns:
        BaseModel: The instantiated model object.
    """
    
    # Instantiate the model
    modelN =  globals().get(modelName)
    if modelN:
        objModel = modelN(device=device)
    else:
        print(f"Class {modelName} not found.")

    # Process additional parameters if provided
    if modelPara is not None:
        # Handle model parameters
        pass

    return objModel




