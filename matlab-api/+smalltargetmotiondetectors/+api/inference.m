function inferResult = inference(model, image)
    % INFERENCE performs inference using a given model on an input image.
    % 
    % Syntax:
    %   inferResult = inference(model, image)
    % 
    % Input:
    %   model - The model object used for inference.
    %   image - The input image on which inference is performed.
    % 
    % Output:
    %   inferResult - The result of the inference process.
    % 
    % Example:
    %   result = inference(myModel, myImage);
    
    inferResult = model.process(image);
end
