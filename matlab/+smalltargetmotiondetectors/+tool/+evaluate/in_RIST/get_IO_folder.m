% get_IO_folder

filePath = mfilename('fullpath');


%%
if startsWith(filePath, 'C:/Users/mings/MATLAB Drive/')
    folderRIST = 'C:/Users/mings/MATLAB Drive/RIST/';
    outputFolder = 'C:/Users/mings/MATLAB Drive/RIST_result/';
elseif startsWith(filePath, '/MATLAB Drive/')
    folderRIST = '/MATLAB Drive/RIST/';
    outputFolder = '/MATLAB Drive/RIST_result/';
end