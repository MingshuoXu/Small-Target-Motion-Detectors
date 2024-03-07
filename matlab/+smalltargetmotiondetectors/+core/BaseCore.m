classdef BaseCore < handle
    % BASECORE Abstract base class for core processing components

    properties
        Opt; % Output
    end

    methods
        function self = BaseCore()
            % Constructor
            % Adds necessary paths and imports modules
            filePath = mfilename('fullpath');
            indexPath = strfind(filePath, 'Small-Target-Motion-Detectors');
            addpath(filePath(1:indexPath(end)+35));
            import smalltargetmotiondetectors.core.*;
        end
    end

    methods (Abstract)
        init(self);     % Abstract method for initialization
        process(self);  % Abstract method for processing
    end

end
