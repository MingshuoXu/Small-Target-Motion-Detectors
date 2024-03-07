% Test script for FracSTMD model

% Clear command window, workspace, and close all figures
clc, clear, close all;

% Add necessary paths
filePath = mfilename('fullpath');
indexPath = strfind(filePath, 'Small-Target-Motion-Detectors');
addpath(filePath(1:indexPath(end)+35));

% Import necessary packages
import smalltargetmotiondetectors.*;
import smalltargetmotiondetectors.api.*;
import smalltargetmotiondetectors.tool.*;

%% Model instantiation

% Instantiate the FracSTMD model
model = instancing_model('FracSTMD');

%% Input source

% Define input source (e.g., image stream or video stream)

% Example 1: Image stream
% hSteam = ImgstreamReader();

% Example 2: Image stream from demo data
% hSteam = ImgstreamReader( ...
%     [filePath(1:indexPath(end)+28),'/demodata/DemoFig*.tif'], ...
%      10, 100 );

% Example 3: Image stream from real-world scene
% hSteam = ImgstreamReader( ...
%     ['I:/Dataset/STMD_Dataset/Real-World-Scence-Material/RIST/', ...
%     'GX010241-1/Real-Image*.jpg'], ...
%     100, 1000 );

% Example 4: Video stream from real-world scene
hSteam = VidstreamReader( ...
    ['D:/Dataset/STMD_Dataset/Real-World-Scence-Material/RIST/', ...
     'GX010290-1/GX010290-1.mp4']);

% Example 5: Image stream from simulated data
% hSteam = ImgstreamReader( ...
%     ['I:/Dataset/STMD_Dataset/Simulated-DataSet/White-Background/', ...
%     'BV-250-Leftward/SingleTarget-TW-5-TH-5-TV-250-TL-0-Rightward', ...
%     '-Amp-0-Theta-0-TemFre-2-SamFre-200/', ...
%     'GeneratingDataSet*.tif'], ...
%     100, 1000 );

% Example 6: Image stream with a single image
% hSteam = ImgstreamReader( ...
%     ['D:/Dataset/STMD_Dataset/Real-World-Scence-Material/RIST/', ...
%     'GX010290-1/Real-Image*.jpg'], ...
%     1, 4000 );

% Instantiate visualization handle
hVisual = get_visualize_handle(class(model));
% Visualization(class(model));

% Initialize the model
model.init();

%% Run inference
while hSteam.hasFrame && hVisual.hasFigHandle
    % Get the next frame from the input source
    [grayImg, colorImg] = hSteam.get_next_frame();
    
    % Perform inference using the model
    result = inference(model, grayImg);
    
    % Visualize the result
    hVisual.show_result(colorImg, result);
end
