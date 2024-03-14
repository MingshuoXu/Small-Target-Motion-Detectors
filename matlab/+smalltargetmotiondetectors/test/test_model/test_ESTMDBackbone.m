%test_ESTMDBackbone

clc, clear, close all;

% Get the full path of this file
filePath = mfilename('fullpath');
%   Find the index of 'Small-Target-Motion-Detectors'
% in the file path
indexPath = strfind(filePath, ...
    '/matlab/+smalltargetmotiondetectors/');
% Add the path to the package containing the models
addpath(filePath(1:indexPath(end)+7));

import smalltargetmotiondetectors.*;
import smalltargetmotiondetectors.api.*;
import smalltargetmotiondetectors.tool.*;

%% model
model = instancing_model('ESTMDBackbone');

%% input

% hSteam = ImgstreamReader();

% Demo images
% hSteam = ImgstreamReader( ...
%     [filePath(1:indexPath(end)-1),'/demodata/imgstream/DemoFig*.jpg'], ...
%     10, 100 );

% Demo video (RIST)
% hSteam = VidstreamReader( ...
%     [filePath(1:indexPath(end)-1),'/demodata/RIST_GX010290.mp4']);

% RIST
% hSteam = VidstreamReader( ...
%     ['D:/Dataset/STMD_Dataset/Real-World-Scence-Material/RIST/', ...
%      'GX010290-1/GX010290-1.mp4']);

% simulate
hSteam = ImgstreamReader( ...
    ['I:/Dataset/STMD_Dataset/Simulated-DataSet/White-Background/', ...
    'BV-250-Leftward/SingleTarget-TW-5-TH-5-TV-250-TL-0-Rightward', ...
    '-Amp-0-Theta-0-TemFre-2-SamFre-1000/', ...
    'GeneratingDataSet*.tif'], ...
    100, 1000 );

%% visualization and model init

% Get visualization handle
hVisual = get_visualize_handle(class(model));

% Initialize the model
model.init();


%% run
while hSteam.hasFrame && hVisual.hasFigHandle
    [grayImg, colorImg] = hSteam.get_next_frame();
    result = inference(model, grayImg);
    hVisual.show_result(colorImg, result);
end
