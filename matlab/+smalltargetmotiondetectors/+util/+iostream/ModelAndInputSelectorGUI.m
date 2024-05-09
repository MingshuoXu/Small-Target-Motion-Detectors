classdef ModelAndInputSelectorGUI < handle
    properties
        root
        objModelSelector
        objInputSelector
        btnRun
        modelName
        vidName
        startImgName
        endImgName
        modelList
        Tag
    end
    
    methods
        function self = ModelAndInputSelectorGUI()
            import smalltargetmotiondetectors.util.iostream.*;
            import smalltargetmotiondetectors.model.*;
            
            self.root = figure(...
                'Name', 'Small target motion detector - Run', ...
                'NumberTitle', 'off', ...
                'menubar', 'none', ...
                'toolbar', 'none');
            
            windowHeight = 350;
            windowWidth = 400;
            
            % 获取屏幕尺寸信息
            screenSize = get(0, 'ScreenSize');

            startHeight = (screenSize(4) - windowHeight) / 2;
            startWidth = (screenSize(3) - windowWidth) / 2;
            
            self.root.Position = [...
                startWidth, startHeight, ...
                windowWidth, windowHeight];
            self.root.Name = "Small target motion detector - Runner";
%             self.root.Icon = fullfile(filePath(1:indexPath-7), 'ico.ico');
            
            self.objModelSelector = ModelSelectorGUI(self.root);
            self.objInputSelector = InputSelectorGUI(self.root);
            self.modelList = BaseModel.get_model_list();
            
            self.btnRun = uicontrol(self.root, ...
                'Style', 'pushbutton', ...
                'String', 'Run', ...
                'Position', [20 30 60 30], ...
                'Callback', @self.callback_run);
        end
        
        function varargout = create_gui(self)
            self.objModelSelector.create_gui();
            self.objInputSelector.create_gui();
            
            uiwait(self.root);
            
            if strcmp(self.Tag, 'vidinput')
                varargout = {self.modelName, {self.vidName}};
            elseif strcmp(self.Tag, 'imginput')
                varargout = {self.modelName, {self.startImgName, self.endImgName}};
            end
        end
        
        function callback_run(self, ~, ~)
            modelOrder = get(self.objModelSelector.modelCombobox, 'Value');
            self.modelName = self.modelList{modelOrder};
            
            selectedTag = get(self.objInputSelector.group, 'SelectedObject');
            self.Tag = selectedTag.Tag;
            switch self.Tag
                case 'vidinput'
                    if ~isempty(self.objInputSelector.vidName)
                        self.vidName = self.objInputSelector.vidName;
                        delete(self.root);
                    else
                        msgbox("Please select a video", 'Message title', 'modal');
                    end
                case 'imginput'
                    if isempty(self.objInputSelector.startImgName)
                        msgbox("Please select start frame!", 'Message title', 'modal');
                        return;
                    end
                    
                    if isempty(self.objInputSelector.endImgName)
                        msgbox("Please select end frame!", 'Message title', 'modal');
                        return;
                    end
                    
                    if self.objInputSelector.check
                        self.startImgName = fullfile(...
                            self.objInputSelector.imgSelectFolder, ...
                            self.objInputSelector.startImgName);
                        self.endImgName = fullfile(...
                            self.objInputSelector.imgSelectFolder, ...
                            self.objInputSelector.endImgName);
                        delete(self.root);
                    else
                        msgbox("The image stream must be in the same folder!", 'Message title', 'modal');
                    end
                otherwise
                    msgbox("Please select input", 'Message title', 'modal');
            end
        end
    end
end
