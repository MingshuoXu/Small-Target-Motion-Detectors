classdef ModelSelectorGUI
    properties
        root;
        modelLabel;
        modelCombobox;
    end
    
    methods
        function self = ModelSelectorGUI(root)
            self.root = root;
        end
        
        function create_gui(self, modelList)
            group0 = uibuttongroup(self.root, 'Position', [0 0.8 1 0.1]);
            self.modelLabel = uicontrol(group0, ...
                'Style', 'text', ...
                'String', 'Select a model:', ...
                'Position', [10 10 120 20]);
            self.modelCombobox = uicontrol(group0, ...
                'Style', 'popupmenu', ...
                'String', modelList, ...
                'Position', [150 10 200 20], ...
                'Value', 1);
        end
    end
end
