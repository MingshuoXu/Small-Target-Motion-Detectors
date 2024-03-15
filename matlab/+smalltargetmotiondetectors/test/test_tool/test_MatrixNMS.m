clc,clear, close all;

%%
% Get the full path of this file
filePath = mfilename('fullpath');
%   Find the index of '/matlab/+smalltargetmotiondetectors/'
% in the file path
indexPath = strfind(filePath, ...
    [filesep, 'matlab', filesep, '+smalltargetmotiondetectors', filesep]);
% Add the path to the package containing the models
addpath(filePath(1:indexPath(end)+7));
import smalltargetmotiondetectors.tool.MatrixNMS

%%
intM = 250;
intN = 500;
listMaxRegionSize = 5:7;
[recordTime1, recordTime2, recordTime3, recordTime4, recordTime5] = ...
    deal( zeros(size(listMaxRegionSize)) );
iList = 0;
for maxRegionSize = listMaxRegionSize
    iList = iList + 1;
    fprintf('running... , maxRegionSize: %d\n', maxRegionSize);
    nTimes = 10;
    randInput = rand(intM, intN);
    %%
    obj1 = MatrixNMS(maxRegionSize, 'bubble');
    timeTic0 = tic;
    for k = 1:nTimes
        opt1 = obj1.nms(randInput);
    end
    recordTime1(iList) = toc(timeTic0);
    %%
    obj2 = MatrixNMS(maxRegionSize, 'conv2');
    timeTic0 = tic;
    for k = 1:nTimes
        opt2 = obj2.nms(randInput);
    end
    recordTime2(iList) = toc(timeTic0);
    %%
    obj3 = MatrixNMS(maxRegionSize, 'greedy');
    timeTic0 = tic;
    for k = 1:nTimes
        opt3 = obj3.nms(randInput);
    end
    recordTime3(iList) = toc(timeTic0);
    %%
    obj4 = MatrixNMS(maxRegionSize, 'sort');
    timeTic0 = tic;
    for k = 1:nTimes
        opt4 = obj4.nms(randInput);
    end
    recordTime4(iList) = toc(timeTic0);

    %%
    obj5 = MatrixNMS(maxRegionSize, 'auto');
    [~] = obj5.nms(randInput); % pre-train for auto
    timeTic0 = tic;
    for k = 1:nTimes
        opt5 = obj5.nms(randInput);
    end
    recordTime5(iList) = toc(timeTic0);
    %%
    diffOutput15 = opt1 - opt5;
    diffOutput14 = opt1 - opt4;
    diffOutput13 = opt1 - opt3;
    diffOutput12 = opt1 - opt2;
    if max(abs(diffOutput13(:))) || max(abs(diffOutput12(:)))...
            || max(abs(diffOutput14(:))) || max(abs(diffOutput15(:)))
        fprintf('error\t\t(%d,%d), maxRegionSize:%d\n',intM,intN,maxRegionSize);

    end

end

%{
%%
figure();
plot(listMaxRegionSize, recordTime1);
hold on;
plot(listMaxRegionSize, recordTime2);
plot(listMaxRegionSize, recordTime3);
plot(listMaxRegionSize, recordTime4);
plot(listMaxRegionSize, recordTime5);
legend('bubble', 'conv2', 'greedy', 'sort', 'auto')

%%
diary("test_result/result_for_testNMS.txt");
fprintf('\n Matrix size : (%d,%d)\n\n',intM,intN);
fprintf('maxRegionSize \t bubble \t conv2 \t\t greedy \t sort \t\t auto\n');
for ii = 1:length(listMaxRegionSize)
    fprintf('\t  %d \t\t %f \t %f \t %f \t %f \t %f \n', listMaxRegionSize(ii),...
        recordTime1(ii), recordTime2(ii), recordTime3(ii), recordTime4(ii), recordTime5(ii));
end
diary off;

%}




