%test_convKernel_in_logical
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


%% 
m = 8; n = 8;
kernel1 = [ones(m, ceil(n)); zeros(m, floor(n))];
kernel2 = [true(m, ceil(n)); false(m, floor(n))];


%%
A = rand(250,500)*rand(1)*100;

times = 1000;
%%

for idT = 1:times

    B = conv2(A, kernel1, 'same');

    C = conv2(A, kernel2, 'same');

    if max(abs(B-C), [], 'all')
        error('!=');
    end
end

