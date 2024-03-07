% Test script for SpikingSTMD model

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
import smalltargetmotiondetectors.model.*;

%% Model instantiation
model = STMDv2();

%% Input source

% Define input source (e.g., image stream or video stream)

% Example 1: Image stream
% hSteam = ImgstreamReader();

% Example 2: Video stream
% hSteam = VidstreamReader();

% Example 3: Image stream from demo data
% hSteam = ImgstreamReader( ...
%     [filePath(1:indexPath(end)+28),'/demodata/DemoFig*.tif'], ...
%      10, 100 );

% Example 4: Image stream from real-world scene
% hSteam = ImgstreamReader( ...
%     ['D:/Dataset/STMD_Dataset/Real-World-Scence-Material/RIST/', ...
%     'GX010290-1/Real-Image*.jpg'], ...
%     1, 4000 );
hSteam = VidstreamReader( ...
    ['D:/Dataset/STMD_Dataset/Real-World-Scence-Material/RIST/', ...
     'GX010290-1/GX010290-1.mp4']);

% Example 5: Simulated image stream with FPS 1000
% hSteam = ImgstreamReader( ...
%     ['I:/Dataset/STMD_Dataset/Simulated-DataSet/White-Background/', ...
%     'BV-250-Leftward/SingleTarget-TW-5-TH-5-TV-250-TL-0-Rightward', ...
%     '-Amp-0-Theta-0-TemFre-2-SamFre-1000/', ...
%     'GeneratingDataSet*.tif'], ...
%     100, 1000 );

% Example 6: Simulated image stream with FPS 1000
% hSteam = ImgstreamReader( ...
%     ['I:/Dataset/STMD_Dataset/Simulated-DataSet/White-Background/', ...
%     'BV-250-Leftward/SingleTarget-TW-5-TH-5-TV-250-TL-0-Rightward', ...
%     '-Amp-0-Theta-0-TemFre-2-SamFre-250/', ...
%     'GeneratingDataSet*.tif'], ...
%     1, 550 );

% Instantiate visualization handle
hVisual = get_visualize_handle(class(model));

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
