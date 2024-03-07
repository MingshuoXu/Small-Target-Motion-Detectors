% demo_imgstearm

clc, clear, close all;

% Determine the file path of the current script
filePath = mfilename('fullpath');
indexPath = strfind(filePath, 'Small-Target-Motion-Detectors');
addpath(filePath(1:indexPath(end)+35));

% Import necessary modules
import smalltargetmotiondetectors.*;
import smalltargetmotiondetectors.api.*;
import smalltargetmotiondetectors.tool.*;

%% Model

% Instantiate the model
model = instancing_model(); 
% model = instancing_model('STMDv2');

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
hImgSteam = ImgstreamReader();

% Alternatively, uncomment one of the following options for different inputs:

% % Demo images
% hImgSteam = ImgstreamReader( ...
%     [filePath(1:indexPath(end)+28),'/demodata/DemoFig*.tif'], ...
%     10, 100 );

% % Real-world images
% hImgSteam = ImgstreamReader( ...
%     ['I:/Dataset/STMD_Dataset/Real-World-Scence-Material/RIST/', ...
%     'GX010290-1/Real-Image*.jpg'], ...
%     10, 1000 );

% % Simulated images
% hImgSteam = ImgstreamReader( ...
%     ['I:/Dataset/STMD_Dataset/Simulated-DataSet/White-Background/', ...
%     'BV-250-Leftward/SingleTarget-TW-5-TH-5-TV-250-TL-0-Rightward', ...
%     '-Amp-0-Theta-0-TemFre-2-SamFre-1000/', ...
%     'GeneratingDataSet*.tif'], ...
%     10, 1000 );

% Get visualization handle
hVisual = get_visualize_handle(class(model));
% Visualization(class(model));

% Initialize the model
model.init();

%% Run
while hImgSteam.hasFrame && hVisual.hasFigHandle
    % Read the next frame from the image stream
    [grayImg, colorImg] = hImgSteam.get_next_frame();
    
    % Perform inference using the model
    result = inference(model, grayImg);
    
    % Display the result
    hVisual.show_result(colorImg, result);
end
