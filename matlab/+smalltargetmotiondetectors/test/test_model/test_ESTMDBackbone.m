%test_ESTMDBackbone

clc, clear, close all;

% Get the full path of this file
filePath = mfilename('fullpath');
%   Find the index of '/+smalltargetmotiondetectors/'
% in the file path
indexPath = strfind(filePath, ...
    [filesep, '+smalltargetmotiondetectors', filesep]);
% Add the path to the package containing the models
addpath(filePath(1:indexPath));

import smalltargetmotiondetectors.api.*;
import smalltargetmotiondetectors.util.iostream.*;

%% model
model = instancing_model('ESTMDBackbone');

%% input

% hSteam = ImgstreamReader();

% %%%%% Demo images
% hSteam = ImgstreamReader( ...
%     [filePath(1:indexPath(end)-8),'/demodata/imgstream/DemoFig*.jpg'], ...
%     10, 100 );

% %%%%% Demo video (RIST)
hSteam = VidstreamReader( ...
    [filePath(1:indexPath(end)-8),'/demodata/RIST_GX010290_orignal_240Hz.mp4']);
% %%%%% Demo video (simulate)
% hSteam = VidstreamReader( ...
%     [filePath(1:indexPath(end)-8),'/demodata/simulatedVideo0_orignal_1000Hz.mp4']);

% RIST
% hSteam = VidstreamReader( ...
%     ['D:/Dataset/STMD_Dataset/Real-World-Scence-Material/RIST/', ...
%      'GX010290-1/GX010290-1.mp4']);

% simulate
% hSteam = ImgstreamReader( ...
%     ['D:/Dataset/STMD_Dataset/Simulated-DataSet/White-Background/', ...
%     'BV-250-Leftward/SingleTarget-TW-5-TH-5-TV-250-TL-0-Rightward', ...
%     '-Amp-0-Theta-0-TemFre-2-SamFre-1000/', ...
%     'GeneratingDataSet*.tif'], ...
%     100, 1000 );

%% visualization and model init

% Get visualization handle
hVisual = get_visualize_handle(class(model));

% Initialize the model
model.init_config();


%% run
while hSteam.hasFrame && hVisual.hasFigHandle
    [grayImg, colorImg] = hSteam.get_next_frame();
    result = inference(model, grayImg);
    hVisual.show_result(colorImg, result);
end
