classdef BaseCore < handle
    % BASECORE Abstract base class for core processing components

    properties
        Opt; % Output
    end

    methods
        function self = BaseCore()
            % Constructor
            % Get the full path of this file
            filePath = mfilename('fullpath');
            %   Find the index of 'Small-Target-Motion-Detectors' 
            % in the file path
            indexPath = strfind(filePath, ...
                '\matlab\+smalltargetmotiondetectors\');
            % Add the path to the package containing the models
            addpath(filePath(1:indexPath(end)+7));
            
            import smalltargetmotiondetectors.core.*;
        end
    end

    methods (Abstract)
        init(self);     % Abstract method for initialization
        process(self);  % Abstract method for processing
    end

end
