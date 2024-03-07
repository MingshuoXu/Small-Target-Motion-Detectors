classdef ImgstreamReader < handle
    % ImgstreamReader - Reads image streams for processing.
    %   This class reads image streams for processing or analysis. It can
    %   read image sequences from specified file paths or from user
    %   selection through a UI.

    properties
        hasFrame = true;    % Flag indicating if there are frames available
        fileList;           % List of files in the image stream
    end

    properties(Hidden)
        currIdx = 1;        % Index of the current frame being read
        frameIdx;           % Current frame index (visible)
        hWaitbar;           % Handle for waitbar UI
        endFrame;           % Index of the last frame
        isShowWaitbar = false;  % Flag indicating whether to show waitbar
        hasDeleteWaitbar = false;   % Flag indicating whether waitbar has been deleted
    end

    methods
        function self = ImgstreamReader(imgsteamFormat, startFrame, endFrame)
            % ImgstreamReader Constructor - Initializes the ImgstreamReader object.
            %   This constructor initializes the ImgstreamReader object. It takes optional
            %   arguments to specify the format of the image stream, the starting frame,
            %   and the ending frame.
            %
            %   Parameters:
            %       - imgsteamFormat: Format of the image stream.
            %       - startFrame: Starting frame index (optional).
            %       - endFrame: Ending frame index (optional).

            % Create waitbar handle
            self.create_waitbar_handle();

            % Set default values if optional arguments are not provided
            if nargin < 3
                endFrame = [];
            end
            if nargin < 2
                startFrame = 1;
            end

            % Initialize file list based on input arguments
            if nargin < 1
                [startImgName, endImgName] = self.get_filelist_from_ui();
            else
                [startImgName, endImgName] = self.get_filelist_from_imgsteamformat( ...
                    imgsteamFormat, startFrame, endFrame);
            end

            % Find start and end frame indices
            isFoundStart = true;
            for idx = 1:length(self.fileList)
                if isFoundStart
                    if strcmp(self.fileList(idx).name, startImgName)
                        startIdx = idx;
                        isFoundStart = false;
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
        end

        function [startImgName, endImgName] = get_filelist_from_ui(self)
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

            % Get the full path of the current file
            filePath = mfilename('fullpath');

            % Find the index of the directory containing "Small-Target-Motion-Detectors"
            indexPath = strfind(filePath, 'Small-Target-Motion-Detectors');

            % Open file dialog for selecting the start frame
            [startImgName, pathName] = uigetfile(...
                {'*.*'}, 'Pick the start frame from image stream', ...
                [filePath(1:indexPath(end)+29), '/demodata/']);

            % Find the index of the dot in the start image name
            dotIndex = strfind(startImgName, '.');

            % Open file dialog for selecting the end frame
            [endImgName, ~] = uigetfile(...
                {startImgName(dotIndex:end)}, ...
                'Pick the end frame from image stream', ...
                pathName);

            % Get the file extensions of the start and end images
            [~, ~, ext1] = fileparts(startImgName);
            [~, ~, ext2] = fileparts(endImgName);

            % Check if the extensions of the start and end images are the same
            if ~strcmp(ext1, ext2)
                error('Start image has a different extension than end image.');
            end

            % Update fileList property with files matching the selected extension
            self.fileList = dir([pathName, '*', ext1]);
        end

        function [startImgName, endImgName] = get_filelist_from_imgsteamformat(...
                self, imgsteamFormat, startFrame, endFrame)
            % get_filelist_from_imgsteamformat - Get start and end frame names from image stream format.
            %   This method generates the start and end frame names based on the specified image stream format.
            %
            %   Parameters:
            %       - self: Instance of the ImgstreamReader class.
            %       - imgsteamFormat: Format string specifying the image stream format.
            %       - startFrame: Index of the start frame.
            %       - endFrame: Index of the end frame.
            %
            %   Returns:
            %       - startImgName: Name of the start frame file.
            %       - endImgName: Name of the end frame file.

            % Retrieve the list of files matching the image stream format
            self.fileList = dir(imgsteamFormat);

            % Extract the basename and extension from the specified format
            [~, basename, ext1] = fileparts(imgsteamFormat);
            basename(end) = []; % Remove the trailing backslash if any

            % Check if any files match the specified format
            if isempty(self.fileList)
                error('No files matching the format could be found.');
            else
                % Determine the end frame index
                if isempty(endFrame)
                    endFrame = length(self.fileList);
                else
                    endFrame = min(endFrame, length(self.fileList));
                end

                % Extract the names of the first and last files in the list
                [~, name1, ~] = fileparts(self.fileList(1).name);
                [~, name0, ~] = fileparts(self.fileList(end).name);

                % Determine if the file names have the same length
                if length(name1) == length(name0)
                    % Extract the numeric part from the end file name
                    num1 = extractBetween(self.fileList(end).name, basename, ext1);
                    numDigits1 = numel(num1{:});

                    % Generate the start and end frame names with zero-padding
                    startImgName = sprintf([basename, '%0', num2str(numDigits1), 'd', ext1], startFrame);
                    endImgName = sprintf([basename, '%0', num2str(numDigits1), 'd', ext1], endFrame);
                else
                    % Generate the start and end frame names without zero-padding
                    startImgName = sprintf([basename, '%d', ext1], startFrame);
                    endImgName = sprintf([basename, '%d', ext1], endFrame);
                end
            end

        end



        function [garyImg, colorImg] = get_next_frame(self)
            % get_next_frame - Retrieves the next frame from the image stream.
            %   This method retrieves the next frame from the image stream and returns
            %   both grayscale and color versions of the frame. It updates the internal
            %   state to point to the next frame in the stream.
            %
            %   Parameters:
            %       - self: Instance of the ImgstreamReader class.
            %
            %   Returns:
            %       - garyImg: Grayscale version of the retrieved frame.
            %       - colorImg: Color version of the retrieved frame.

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
            garyImg = double(rgb2gray(colorImg));

            % Update internal state to point to the next frame
            if self.currIdx < length(self.fileList)
                self.hasFrame = true;
                self.currIdx = self.currIdx + 1;
                self.frameIdx = self.frameIdx + 1;
            else
                % If the end of the image stream is reached, set hasFrame to false
                self.hasFrame = false;
            end

            % Update the waitbar if necessary
            if self.isShowWaitbar
                self.call_waitbar();
            elseif ~self.hasDeleteWaitbar
                delete(self.hWaitbar);
                self.hasDeleteWaitbar = true;
            end
        end

        function create_waitbar_handle(self)
            % create_waitbar_handle - Creates a waitbar for displaying progress.
            %   This method creates a waitbar with an initial progress of 0% and a
            %   specific message, typically used to indicate the initialization phase
            %   of the ImgstreamReader.
            %
            %   Parameters:
            %       - self: Instance of the ImgstreamReader class.

            % Create a waitbar with specific properties
            self.hWaitbar = waitbar(...
                0, 'Initiating, please wait...', ...  % Initial progress and message
                'Name', 'ImgstreamReader', ...  % Title of the waitbar
                'Position', [450, 450, 270, 50] ...  % Position and size of the waitbar
                );
        end

        function call_waitbar(self)
            % call_waitbar - Updates the waitbar with current frame index.
            %   This method updates the progress of the waitbar based on the current
            %   frame index relative to the total number of frames and displays a
            %   message indicating the current frame index.
            %
            %   Parameters:
            %       - self: Instance of the ImgstreamReader class.

            % Calculate the progress as a floating-point number
            floatBar = self.frameIdx / self.endFrame;

            % Generate a message indicating the current frame index
            waitbarStr = sprintf('FrameIndex: %d', self.frameIdx);

            % Update the waitbar with the current progress and message
            self.hWaitbar = waitbar(floatBar, self.hWaitbar, waitbarStr);
        end


        function delete(self)
            % delete - Deletes the waitbar object.
            %   This method deletes the waitbar object associated with the
            %   ImgstreamReader instance when the instance is deleted.
            %
            %   Parameters:
            %       - self: Instance of the ImgstreamReader class.

            % Delete the waitbar object
            delete(self.hWaitbar);
        end


    end
end



