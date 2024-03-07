% demo_vidstream

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

% Create a video stream reader
% Specify the path to the video file or uncomment one of the provided options

hSteam = VidstreamReader();

% Real-world video
% hSteam = VidstreamReader( ...
%     [filePath(1:indexPath(end)+28),'/demodata/GX010290-1.mp4']);

% Demo video
% hSteam = VidstreamReader( ...
%     [filePath(1:indexPath(end)+28),'/demodata/demo_video.avi'], ...
%      10, 1000 );

% Get visualization handle
hVisual = get_visualize_handle(class(model));
% Visualization(class(model));

% Initialize the model
model.init();

%% Run
while hSteam.hasFrame && hVisual.hasFigHandle
    % Read the next frame from the video stream
    [grayImg, colorImg] = hSteam.get_next_frame();
    
    % Perform inference using the model
    result = inference(model, grayImg);
    
    % Display the result
    hVisual.show_result(colorImg, result);
end
