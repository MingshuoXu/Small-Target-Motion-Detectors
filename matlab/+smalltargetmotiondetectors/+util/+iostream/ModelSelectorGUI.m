classdef ModelSelectorGUI < handle
    properties
        root;
        modelList;
        modelLabel;
        modelCombobox;
    end
    
    methods
        function self = ModelSelectorGUI(root)
            import smalltargetmotiondetectors.model.*;
            self.root = root;
            self.modelList = BaseModel.get_model_list();
        end
        
        function create_gui(self)
            group0 = uibuttongroup(self.root, ...
                'Position', [0 0.85 1 0.1], ...
                'BorderType', 'none');
            self.modelLabel = uicontrol(group0, ...
                'Style', 'text', ...
                'String', 'Select a model:', ...
                'Position', [10 10 120 20]);
            self.modelCombobox = uicontrol(group0, ...
                'Style', 'popupmenu', ...
                'String', self.modelList, ...
                'Position', [150 10 220 20], ...
                'Value', 4);
        end
    end
end
