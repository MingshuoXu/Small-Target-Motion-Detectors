% test_gui

clc, clear, close all;

%%
% Get the full path of this file
filePath = mfilename('fullpath');
%   Find the index of '/+smalltargetmotiondetectors/'
% in the file path
indexPath = strfind(filePath, ...
    [filesep, '+smalltargetmotiondetectors', filesep]);
% Add the path to the package containing the models
addpath(filePath(1:indexPath));
import smalltargetmotiondetectors.util.iostream.*;

%%


selector_gui = ModelAndInputSelectorGUI();
selector_gui.create_gui();

% 
movegui(root, 'center');
set(root, 'Visible', 'on');