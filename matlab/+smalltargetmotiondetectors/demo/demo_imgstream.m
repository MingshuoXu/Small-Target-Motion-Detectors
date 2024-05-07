% demo_imgstearm

clc, clear, close all;

% Get the full path of this file
filePath = mfilename('fullpath');
%   Find the index of '/+smalltargetmotiondetectors/'
% in the file path
indexPath = strfind(filePath, ...
    [filesep, '+smalltargetmotiondetectors', filesep]);
% Add the path to the package containing the models
addpath(filePath(1:indexPath));

% Import necessary modules
import smalltargetmotiondetectors.api.*;
import smalltargetmotiondetectors.util.iostream.*;

%% Model

% Instantiate the model
model = instancing_model(); 

%{
Please refer to +model/Readme or Readme for smalltargetmotiondetectors.

ESTMD           (2008, S.D. Wiederman, PLoS ONE)
DSTMD           (2020, H. Wang, IEEE T--Cybernetics)
STMDPlus        (2020, H. Wang, IEEE T-NNLS)
FeedbackSTMD    (2021, H. Wang, IEEE T-NNLS)
FSTMD           (2021, Ling J, Front. Neurorobot)
ApgSTMD         (2022, H. Wang, IEEE T--Cybernetics)
FracSTMD        (2023, Xu, M., Neurocomputing)
STMDv2          -indevelopment
%}

%% Input

% Create an image stream reader
hSteam = ImgstreamReader();

% Alternatively, uncomment the following options for different inputs:

% % Demo images
% hSteam = ImgstreamReader( ...
%     [filePath(1:indexPath(end)-1),'/demodata/imgstream/DemoFig*.jpg'], ...
%     10, 100 );

%% Get visualization handle and initiate model
% Get visualization handle
hVisual = get_visualize_handle(class(model));

% Initialize the model
model.init_config();

%% Run
while hSteam.hasFrame && hVisual.hasFigHandle
    % Read the next frame from the image stream
    [grayImg, colorImg] = hSteam.get_next_frame();
    
    % Perform inference using the model
    result = inference(model, grayImg);
    
    % Display the result
    hVisual.show_result(colorImg, result);
end
