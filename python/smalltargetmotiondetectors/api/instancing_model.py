import os
import tkinter as tk
from tkinter import filedialog

from ..model.backbone import *

def instancing_model(modelName=None, modelPara=None):
    """
    Instantiate a model object based on the given model name.

    Parameters:
        modelName (str): Name of the model to instantiate. If None, a GUI for model selection will be opened.
        modelPara: Parameters for model instantiation (optional).

    Returns:
        BaseModel: The instantiated model object.
    """

    # Determine if GUI should be opened for model selection
    isOpenUI = True if modelName is None else False

    '''
    # Open GUI for model selection if necessary
    if isOpenUI:
        # Open file dialog for model selection
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        modelName = filedialog.askopenfilename(
            initialdir='../',
            title="Pick a model from M-file or P-file",
            filetypes=(("M-file", "*.m"), ("P-file", "*.p")))
        root.destroy()  # Close the hidden Tkinter window

        # Check if BaseModel is selected (it's an abstract class)
        if os.path.basename(modelName) == 'BaseModel.m':
            raise ValueError("BaseModel is an Abstract Class! Please select another model.")

        # Check if a model is selected
        if not modelName:
            raise ValueError("Please re-run and select a model.")
        # Remove file extension from the model name
        modelName = os.path.splitext(os.path.basename(modelName))[0]
    '''
    
    # Instantiate the model
    model = eval(modelName + "()")

    # Process additional parameters if provided
    if modelPara is not None:
        # Handle model parameters
        pass

    return model

# Example usage:
# model = instancing_model('STMDv2')
