%test_convKernel
clc, clear, close all;

%%
% Get the full path of this file
filePath = mfilename('fullpath');
%   Find the index of '/matlab/+smalltargetmotiondetectors/'
% in the file path
indexPath = strfind(filePath, ...
    [filesep, 'matlab', filesep, '+smalltargetmotiondetectors', filesep]);
% Add the path to the package containing the models
addpath(filePath(1:indexPath(end)+7));


import smalltargetmotiondetectors.tool.kernel.*;

%% 
predictionKernel = create_prediction_kernel();

kernel1 = predictionKernel{1};
kernel2 = kernel1;
kernel2(:,1:12) = 0;

%%
A = rand(250,500)*rand(1)*100;

times = 1000;
%%

for idT = 1:times

    B = conv2(A, kernel1, 'same');

    C = conv2(A, kernel2, 'same');

end

