import os
import sys

# Get the full path of this file
filePath = os.path.realpath(__file__)
# Find the index of '/+smalltargetmotiondetectors/'
# in the file path
indexPath = filePath.rfind(os.path.sep + 'smalltargetmotiondetectors' + os.path.sep)
# Add the path to the package containing the models
sys.path.append(filePath[:indexPath])

# from smalltargetmotiondetectors import model
from smalltargetmotiondetectors.model import *
from smalltargetmotiondetectors.util.iostream import *

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


