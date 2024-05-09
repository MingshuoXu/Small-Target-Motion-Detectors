classdef ImgstreamReader < handle
    properties
        hasFrame = false;
        fileList
        currIdx = 1;
        frameIdx
        hWaitbar
        isShowWaitbar
        hasDeleteWaitbar
        imgsteamFormat
        startFrame
        endFrame
        startImgFullName
        endImgFullName
    end
    
    methods
        function self = ImgstreamReader(imgsteamFormat, startFrame, endFrame, ...
                startImgFullName, endImgFullName)           
            
            if nargin > 0
                self.imgsteamFormat = imgsteamFormat;
            end
            
            if nargin > 1
                self.startFrame = startFrame;
            end
            
            if nargin > 2
                self.endFrame = endFrame;
            end
            
            if nargin == 5
                self.startImgFullName = startImgFullName;
                self.endImgFullName = endImgFullName;
                self.get_filelist_from_imgName();
            elseif nargin == 0
                self.get_filelist_from_ui()
            elseif nargin < 4
                self.get_filelist_from_imgsteamformat();              
            else
                error('Incorrect number of input arguments.');
            end
            
            self.get_idx();
        end
        
        function get_idx(self)
            % Find start and end frame indices
            shouldFoundStart = true;
            [~, startImgName, ext] = fileparts(self.startImgFullName);
            startImgName = [startImgName, ext];
            [~, endImgName, ~] = fileparts(self.endImgFullName);
            endImgName = [endImgName, ext];
            
            for idx = 1:length(self.fileList)
                if shouldFoundStart
                    if strcmp(self.fileList(idx).name, startImgName)
                        startIdx = idx;
                        shouldFoundStart = false;
                    end
                else
                    if strcmp(self.fileList(idx).name, endImgName)
                        endIdx = idx;
                        break;
                    end
                end
            end

            % Check if start and end frames are found
            if ~exist('startIdx', 'var')
                error('Cannot find the start frame.');
            end
            if ~exist('endIdx', 'var')
                error('Cannot find the end frame.');
            end

            % Set fileList to frames between startIdx and endIdx
            self.fileList = self.fileList(startIdx:endIdx);

            % Check if fileList is empty
            if isempty(self.fileList)
                self.hasFrame = false;
            end

            % Set frameIdx to startIdx
            self.frameIdx = startIdx;

            % Set endFrame to endIdx
            self.endFrame = endIdx;
            
            self.hasFrame = true;
        end
        
        function get_filelist_from_imgName(self)
            % Find the index of the dot in the start image name
            [startFolder, ~, startExt] = fileparts(self.startImgFullName);
            [endFolder, ~, endExt] = fileparts(self.endImgFullName);
            
            if ~strcmp(startExt, endExt)
                error('Start image has a different extension than end image.');
            end
            if ~strcmp(startFolder, endFolder)
                error('The image stream must be in the same folder!');
            end
            
            % Update fileList property with files matching the selected extension
            self.fileList = dir(fullfile(startFolder, ['*', startExt]));
        end
        
        function get_filelist_from_ui(self)
            % get_filelist_from_ui - Prompts user to select start and end frames via GUI.
            %   This method prompts the user to select the start and end frames of the
            %   image stream using a graphical user interface (GUI). It opens a file dialog
            %   for the user to choose the start frame, and then another file dialog for
            %   the end frame, ensuring that both frames have the same file extension.
            %
            %   Parameters:
            %       - self: Instance of the ImgstreamReader class.
            %
            %   Returns:
            %       - startImgName: Name of the selected start frame file.
            %       - endImgName: Name of the selected end frame file.

            % Get the full path of this file
            filePath = mfilename('fullpath');
            %   Find the index of 'Small-Target-Motion-Detectors'
            % in the file path
            indexPath = strfind(filePath, ...
                [filesep, '+smalltargetmotiondetectors', filesep]);
            % Add the path to the package containing the models
            addpath(filePath(1:indexPath));

            % Open file dialog for selecting the start frame
            [self.startImgFullName, pathName] = uigetfile(...
                {'*.*'}, 'Pick the start frame from image stream', ...
                [filePath(1:indexPath-7), 'demodata', filesep,'imgstream', filesep]);

            % Find the index of the dot in the start image name
            dotIndex = strfind(self.startImgFullName, '.');

            % Open file dialog for selecting the end frame
            [self.endImgFullName, ~] = uigetfile(...
                {self.startImgFullName(dotIndex:end)}, ...
                'Pick the end frame from image stream', ...
                pathName);

            % Get the file extensions of the start and end images
            [~, ~, ext1] = fileparts(self.startImgFullName);
            [~, ~, ext2] = fileparts(self.endImgFullName);

            % Check if the extensions of the start and end images are the same
            if ~strcmp(ext1, ext2)
                error('Start image has a different extension than end image.');
            end

            % Update fileList property with files matching the selected extension
            self.fileList = dir([pathName, '*', ext1]);
        end
        
        function get_filelist_from_imgsteamformat(self)
            % Retrieve the list of files matching the image stream format
            self.fileList = dir(self.imgsteamFormat);
            self.fileList = fullfile({self.fileList.folder}, {self.fileList.name});
            
            % Check if any files match the specified format
            if isempty(self.fileList)
                error('No files matching the format could be found.');
            else
                % Determine the end frame index
                if isempty(self.endFrame)
                    self.endFrame = length(self.fileList);
                else
                    self.endFrame = min(self.endFrame, length(self.fileList));
                end
                
                % Generate the start and end frame names
                [~, name, ext] = fileparts(self.imgsteamFormat);
                if length(name) == length(self.fileList(1))
                    self.startImgFullName = fullfile(fileparts(self.fileList(1)), [name num2str(self.startFrame, ['%0' num2str(length(self.fileList(1))) 'd']) ext]);
                    self.endImgFullName = fullfile(fileparts(self.fileList(1)), [name num2str(self.endFrame, ['%0' num2str(length(self.fileList(1))) 'd']) ext]);
                else
                    self.startImgFullName = fullfile(fileparts(self.fileList(1)), [name num2str(self.startFrame) ext]);
                    self.endImgFullName = fullfile(fileparts(self.fileList(1)), [name num2str(self.endFrame) ext]);
                end
            end
        end
        
        function [grayImg, colorImg] = get_next_frame(self)
            % Get information about the current frame
            fileInfo = self.fileList(self.currIdx);
            fullPath = fullfile(fileInfo.folder, fileInfo.name);
            
            % Try to read the image file
            try
                colorImg = imread(fullPath);
                self.hasFrame = true;
            catch
                % If an error occurs while reading the image, set hasFrame to false
                self.hasFrame = false;
                error('Could not read the image.');
            end
            
            % Convert the color image to grayscale
            grayImg = double(rgb2gray(colorImg));
            
            % Update internal state to point to the next frame
            if self.currIdx < length(self.fileList)
                self.hasFrame = true;
                self.currIdx = self.currIdx + 1;
                self.frameIdx = self.frameIdx + 1;
            else
                % If the end of the image stream is reached, set hasFrame to false
                self.hasFrame = false;
            end
        end
    end
end
