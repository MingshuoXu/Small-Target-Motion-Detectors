classdef InputSelectorGUI < handle
    properties
        root
        vidElement
        imgElement
        imgSelectFolder
        inputType
        startFrame
        endFrame
        videoFile
        check
        vidName
        startFolder
        endFolder
        startImgName
        endImgName
        startExt
        endExt
        selectedOption
        inputTypeLabel
        vidLabel
        imgLabel
        group
        IMG_DEFAULT_FOLDER
        VID_DEFAULT_FOLDER
    end
    
    
    
    methods
        function self = InputSelectorGUI(root)
            
            % Get the full path of this file
            filePath = mfilename('fullpath');
            %   Find the index of 'Small-Target-Motion-Detectors'
            % in the file path
            indexPath = strfind(filePath, ...
                [filesep, '+smalltargetmotiondetectors', filesep]);
            % Add the path to the package containing the models
            addpath(filePath(1:indexPath));
            
            self.root = root;
            self.vidElement = containers.Map;
            self.imgElement = containers.Map;
            self.imgSelectFolder = '';
            self.vidName = '';
            self.startImgName = '';
            self.endImgName = '';
            self.selectedOption = 0;
            self.VID_DEFAULT_FOLDER = ...
                fullfile(filePath(1:indexPath-7), 'demodata');
            self.IMG_DEFAULT_FOLDER = ...
                fullfile(self.VID_DEFAULT_FOLDER, 'imgstream');
        end
        
        function create_gui(self)
            self.group = uibuttongroup(self.root, ...
                'Position', [0 0.7 1 0.1], ...
                'BorderType', 'none');
            
            self.inputTypeLabel = uicontrol(self.group, ...
                'Style', 'text', ...
                'String', 'Select input from:', ...
                'Position', [10 10 120 20]);
            
            self.imgLabel = uicontrol(self.group, ...
                'Style', 'radiobutton', ...
                'String', 'Image stream', ...
                'Position', [150 10 100 20], ...
                'Tag', 'imginput', ...
                'Callback', @self.select_imgstream);
            self.vidLabel = uicontrol(self.group, ...
                'Style', 'radiobutton', ...
                'String', 'Video stream', ...
                'Position', [250 10 100 20], ...
                'Tag', 'vidinput', ...
                'Callback', @self.select_vidstream);
            set(self.group, 'SelectedObject', []);
        end
        
        function select_vidstream(self, ~, ~)
            self.imgSelectFolder = [];
            self.startImgName = [];
            self.endImgName = [];
            
            keys = self.imgElement.keys;
            for i = 1:numel(keys)
                key = keys{i};
                element = self.imgElement(key);
                delete(element);
            end
            
            
            self.vidElement('lblVidIndicate') = uicontrol(self.root, ...
                'Style', 'text', ...
                'String', 'Video''s path:', ...
                'Position', [10 180 120 50]);
            self.vidElement('lblVidPath') = uicontrol(self.root, ...
                'Style', 'text', ...
                'String', 'Waiting for the selection', ...
                'Position', [150 180 220 50]);
            self.vidElement('btn') = uicontrol(self.root, ...
                'Style', 'pushbutton', ...
                'String', 'Select a video', ...
                'Position', [250 140 100 30], ...
                'Callback', @self.clicked_vid);
        end
        
        function clicked_vid(self, ~, ~)
            [vidfileName, vidPathName] = uigetfile(...
                {'*.avi;*.mp4;*.mov'}, 'Select a video file', ...
                self.VID_DEFAULT_FOLDER);
            self.vidName = fullfile(vidPathName, vidfileName);
            set(self.vidElement('lblVidPath'), 'String', self.vidName);
        end
        
        function select_imgstream(self, ~, ~)
            self.vidName = [];
            
            keys = self.vidElement.keys;
            for i = 1:numel(keys)
                key = keys{i};  % 获取当前键
                element = self.vidElement(key);  % 获取当前键对应的值
                delete(element);  % 删除元素
            end
            
            
            self.imgElement('lblFolder') = uicontrol(self.root, ...
                'Style', 'text', ...
                'String', 'Image''s folder:', ...
                'Position', [10 180 120 50]);
            self.imgElement('lblFolderName') = uicontrol(self.root, ...
                'Style', 'text', ...
                'String', 'Waiting for the selection', ...
                'Position', [150 180 220 50]);
            self.imgElement('btnStart') = uicontrol(self.root, ...
                'Style', 'pushbutton', ...
                'String', 'Select start frame', ...
                'Position', [150 140 90 30], ...
                'Callback', @self.clicked_start_img);
            self.imgElement('btnEnd') = uicontrol(self.root, ...
                'Style', 'pushbutton', ...
                'String', 'Select end frame', ...
                'Position', [150 100 90 30], ...
                'Callback', @self.clicked_end_img);
        end
        
        function clicked_start_img(self, ~, ~)
            if isempty(self.imgSelectFolder)
                defaultPath = self.IMG_DEFAULT_FOLDER;
            else
                defaultPath = self.imgSelectFolder;
            end

            [self.startImgName, self.startFolder] = uigetfile(...
                    {'*.*'}, 'Select a start frame', ...
                    defaultPath);
            [~, ~, self.startExt] = fileparts(self.startImgName);
            
            if ~isempty(self.endFolder)
                % Check if folders are the same
                if strcmp(self.startFolder, self.endFolder)
                    % Check if file extensions are the same
                    if ~isempty(self.endImgName)
                        if strcmp(self.startExt, self.endExt)
                            self.check = true;
                        else
                            msgbox('Start image has a different extension than end image.',...
                                'Message title', 'warn');
                        end
                    end
                else
                    msgbox('The image stream must be in the same folder!',...
                        'Message title', 'warn');
                end
            end
            
            % Update imgSelectFolder
            self.imgSelectFolder = self.startFolder;
            
            % Update label text in GUI
            set(self.imgElement('lblFolderName'), 'String', self.imgSelectFolder);
            
            % Create and display label for image filename
            self.imgElement('lblStartImg') = uicontrol(self.root, ...
                'Style', 'text', ...
                'String', self.startImgName, ...
                'Position', [250 140 120 30]);
        end
        
        
        function clicked_end_img(self, ~, ~)
            if isempty(self.imgSelectFolder)
                defaultPath = self.IMG_DEFAULT_FOLDER;
            else
                defaultPath = self.imgSelectFolder;
            end
            
            [self.endImgName, self.endFolder] = uigetfile(...
                    {'*.*'}, 'Select a start frame', ...
                    defaultPath);
            [~, ~, self.endExt] = fileparts(self.endImgName);
            
            % Check if the selected folder matches with the start image folder
            if ~isempty(self.startFolder)
                % Check if folders are the same
                if strcmp(self.startFolder, self.endFolder)
                    % Check if file extensions are the same
                    if ~isempty(self.startImgName)
                        if strcmp(self.startExt, self.endExt)
                            self.check = true;
                        else
                            msgbox('Start image has a different extension than end image.',...
                                'Message title', 'warn');
                        end
                    end
                else
                    msgbox('The image stream must be in the same folder!',...
                        'Message title', 'warn');
                end
            end
            
            % Update imgSelectFolder with the selected folder
            self.imgSelectFolder = self.endFolder;
            
            % Update the text of lblFolderName in the GUI
            set(self.imgElement('lblFolderName'), ...
                'String', self.imgSelectFolder);
            
            % Create and display label for the end image filename
            self.imgElement('lblEndImg') = uicontrol(self.root, ...
                'Style', 'text', ...
                'String', self.endImgName, ...
                'Position', [250 100 120 30]);
        end
        
    end
end

