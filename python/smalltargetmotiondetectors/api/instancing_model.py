import os
import sys

packagePath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(packagePath)
from smalltargetmotiondetectors.model import * #type: ignore
# from smalltargetmotiondetectors.util.datarecord import ModelNameMapping #type: ignore


def instancing_model(modelName, modelPara=None):
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
        objModel = modelN()
    else:
        print(f"Class {modelName} not found.")

    # Process additional parameters if provided
    if modelPara is not None:
        # Handle model parameters
        pass

    return objModel


if __name__ == "__main__":
    objModel = instancing_model('ESTMD')
    print(objModel)


