%test_GaussianBlur

clc,clear,close all;
% Get the full path of this file
filePath = mfilename('fullpath');
%   Find the index of 'Small-Target-Motion-Detectors'
% in the file path
indexPath = strfind(filePath, ...
    '\matlab\+smalltargetmotiondetectors\');
% Add the path to the package containing the models
addpath(filePath(1:indexPath(end)+7));
import smalltargetmotiondetectors.core.*;

%%

testObj = smalltargetmotiondetectors.core.GaussianBlur();
