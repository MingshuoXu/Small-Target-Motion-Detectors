%test_ImgsteamReader

clc,clear,close all;

%%
filePath = mfilename('fullpath');
indexPath = strfind(filePath, 'Small-Target-Motion-Detectors');
addpath(filePath(1:indexPath(end)+35));
import smalltargetmotiondetectors.tool.*;

%%

testObj = ImgstreamReader();

% testObj.init();