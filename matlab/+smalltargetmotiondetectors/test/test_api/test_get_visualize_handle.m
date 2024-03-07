% Test file for get_visualize_handle function

clc, clear, close all;

%%
filePath = mfilename('fullpath');
indexPath = strfind(filePath, 'Small-Target-Motion-Detectors');
addpath(filePath(1:indexPath(end)+35));

import smalltargetmotiondetectors.*;
import smalltargetmotiondetectors.api.*;

%%
% Test case 1: No input argument
disp('Test Case 1: No input argument');
hVis1 = get_visualize_handle();
disp('Visualization object created successfully.');

% Test case 2: With class name input argument
disp('Test Case 2: With class name input argument');
className = '.ESTMD';
hVis2 = get_visualize_handle(className);
disp('Visualization object created successfully.');

% Test case 3: Invalid class name input argument
disp('Test Case 3: Invalid class name input argument');
classNameInvalid = 'InvalidClassName';
try
    hVis3 = get_visualize_handle(classNameInvalid);
catch ME
    disp(['Error: ', ME.message]);
end
