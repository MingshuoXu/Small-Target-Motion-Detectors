% get_IO_folder

filePath = mfilename('fullpath');

if startsWith(filePath, [filesep, 'MATLAB Drive', filesep])
    % on matlab drive
    folderRIST = '/MATLAB Drive/RIST/';
    outputFolder = '/MATLAB Drive/RIST_result/';
else
    if exist(['C:', filesep, 'Users', filesep, 'mings', filesep],...
            "dir") == 7
        userName = 'mings';
    elseif exist(['C:', filesep, 'Users', filesep, 'mx60', filesep],...
            "dir") == 7
        userName = 'mx60';
    else
        userName = input('input the user name: ', 's');
    end

    if startsWith(filePath(3:end), [filesep, 'Onedrive_mings', filesep]) ...
            || startsWith(filePath, ['C:', filesep, 'Users', filesep, 'mings', filesep]) 

        % on mobile HDD
        folderRIST = ['C:', filesep, 'Users', filesep, userName, filesep, ...
            'MATLAB Drive', filesep, 'RIST', filesep];
        outputFolder = ['C:', filesep, 'Users', filesep, userName, filesep, ...
            'MATLAB Drive', filesep, 'RIST_result', filesep];

    end
end

