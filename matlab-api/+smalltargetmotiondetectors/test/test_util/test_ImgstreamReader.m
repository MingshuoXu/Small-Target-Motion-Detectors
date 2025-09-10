%test_ImgsteamReader

clc,clear,close all;

%%
% Get the full path of this file
filePath = mfilename('fullpath');
%   Find the index of '/matlab/+smalltargetmotiondetectors/'
% in the file path
indexPath = strfind(filePath, ...
    [filesep, 'matlab', filesep, '+smalltargetmotiondetectors', filesep]);
% Add the path to the package containing the models
addpath(filePath(1:indexPath(end)+7));

import smalltargetmotiondetectors.util.*;

%%

testObj = ImgstreamReader();

% testObj.init();