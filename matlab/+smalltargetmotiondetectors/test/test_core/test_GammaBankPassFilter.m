%test_GammaBankPassFilter

clc,clear,close all;
filePath = mfilename('fullpath');
indexPath = strfind(filePath, 'Small-Target-Motion-Detectors');
addpath(filePath(1 : indexPath(end)+35));
import smalltargetmotiondetectors.core.estmd_core.*;

%%

testObj = GammaBankPassFilter();

testObj.init();