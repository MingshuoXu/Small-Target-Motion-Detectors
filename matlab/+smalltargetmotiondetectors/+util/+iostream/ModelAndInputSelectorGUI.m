classdef ModelAndInputSelectorGUI
    properties
        root
        objModelSelector
        objInputSelector
        btnRun
        modelName
        vidName
        startImgName
        endImgName
    end
    
    methods
        function self = ModelAndInputSelectorGUI()
            import smalltargetmotiondetectors.util.iostream.*;
            
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
            
            self.btnRun = uicontrol(self.root, ...
                'Style', 'pushbutton', ...
                'String', 'Run', ...
                'Position', [20 30 60 30]);
        end
        
        function varargout = create_gui(self)
            self.objModelSelector.create_gui({'1', '2', '3'});
            self.objInputSelector.create_gui();
            
            uiwait(self.root);
            
            if self.objInputSelector.selectedOption == 1
                varargout = {self.modelName, self.vidName};
            elseif self.objInputSelector.selectedOption == 2
                varargout = {self.modelName, self.startImgName, self.endImgName};
            end
        end
        
        function custom_run(self, ~, ~)
            self.modelName = self.objModelSelector.modelCombobox.String{...
                self.objModelSelector.modelCombobox.Value};
            ALL_MODEL = {'Model1', 'Model2', 'Model3'};  % Define ALL_MODEL here
            
            if ~any(strcmp(self.modelName, ALL_MODEL))
                msgbox("Please select a STMD-based model!", 'Message title', 'modal');
                return;
            end
            
            switch self.objInputSelector.selectedOption
                case 1
                    if ~isempty(self.objInputSelector.vidName)
                        self.vidName = self.objInputSelector.vidName;
                        delete(self.root);
                    else
                        msgbox("Please select a video", 'Message title', 'modal');
                    end
                case 2
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
