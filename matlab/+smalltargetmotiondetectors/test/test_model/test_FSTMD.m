%test_FSTMD

clc, clear, close all;

%%
filePath = mfilename('fullpath');
indexPath = strfind(filePath, 'Small-Target-Motion-Detectors');
addpath(filePath(1:indexPath(end)+35));

import smalltargetmotiondetectors.*;
import smalltargetmotiondetectors.api.*;
import smalltargetmotiondetectors.tool.*;

%% model

model = instancing_model('FSTMD');
model.init();

%% input

% hSteam = ImgstreamReader();

% demofig
% hSteam = ImgstreamReader( ...
%     [filePath(1:indexPath(end)+28),'/demodata/DemoFig*.tif'], ...
%      10, 100 );

% RIST
% hSteam = ImgstreamReader( ...
%     ['I:/Dataset/STMD_Dataset/Real-World-Scence-Material/RIST/', ...
%     'GX010241-1/Real-Image*.jpg'], ...
%     100, 1000 );
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

hVisual = get_visualize_handle(class(model));
% Visualization(class(model));


%% run
while hSteam.hasFrame && hVisual.hasFigHandle
    [grayImg, colorImg] = hSteam.get_next_frame();
    result = inference(model, grayImg);
    hVisual.show_result(colorImg, result);
end
