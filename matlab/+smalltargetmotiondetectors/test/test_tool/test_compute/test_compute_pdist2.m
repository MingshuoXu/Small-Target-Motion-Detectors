%test_ApgSTMD

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

%%
import smalltargetmotiondetectors.tool.compute.*;

%%
m = 300; n = 2;

times = 100;

%%
timeA = 0;
timeB = 0;
for tidx = 1:times
    A = rand(m, n);
    B = rand(m, n);
    
    ticA = tic;
    C = pdist2(A, B);
    timeA = timeA + toc(ticA);
    ticB = tic;
    D = compute_pdist2(A, B);
    timeB = timeB + toc(ticB);

    errormax = max(abs(C-D),[], 'all');
    if errormax
        error('C != D');
    end
end
averageTimeA = timeA / times;
averageTimeB = timeB / times;
fprintf('TimeA: %f,\t agerageTimeA: %f\n', timeA, averageTimeA);
fprintf('TimeB: %f,\t agerageTimeB: %f\n', timeB, averageTimeB);