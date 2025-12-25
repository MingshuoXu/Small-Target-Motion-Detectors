def inference(model, image):
    """
    Performs inference using a given model on an input image.

    Parameters:
        model: The model object used for inference.
        image: The input image on which inference is performed.

    Returns:
        object: The result of the inference process.
    """
    return model.process(image)
