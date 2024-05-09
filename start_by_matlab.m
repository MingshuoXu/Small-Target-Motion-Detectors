% demo_gui
clc, clear, close all;

%%
% Get the full path of this file
filePath = mfilename('fullpath');
addpath(fullfile(filePath(1:end-15), 'matlab'));

import smalltargetmotiondetectors.util.iostream.*;
import smalltargetmotiondetectors.api.*;

%%
selector_gui = ModelAndInputSelectorGUI();
[modelName, inputName] = selector_gui.create_gui();

%%
objModel = instancing_model(modelName); 

if length(inputName) == 1
    hSteam = VidstreamReader(inputName{1});
elseif length(inputName) == 2
    hSteam = ImgstreamReader([], [], [], inputName{1}, inputName{2});
else
   error(''); 
end

%% Get visualization handle and initiate model
% Get visualization handle
hVisual = get_visualize_handle(class(objModel));

% Initialize the model
objModel.init_config();

%% Run
while hSteam.hasFrame && hVisual.hasFigHandle
    % Read the next frame from the image stream
    [grayImg, colorImg] = hSteam.get_next_frame();
    
    % Perform inference using the model
    result = inference(objModel, grayImg);
    
    % Display the result
    hVisual.show_result(colorImg, result);
end