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
            %   Find the index of '/+smalltargetmotiondetectors/'
            % in the file path
            indexPath = strfind(filePath, ...
                [filesep, '+smalltargetmotiondetectors', filesep]);
            % Add the path to the package containing the models
            addpath(filePath(1:indexPath));

            import smalltargetmotiondetectors.core.*;
        end
    end

    methods (Abstract)
        init_config(self);  % Abstract method for initialing kernels etc.
        process(self);      % Abstract method for processing
    end

end
