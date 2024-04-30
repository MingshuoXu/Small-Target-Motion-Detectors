classdef Lobula < smalltargetmotiondetectors.core.feedbackstmd_core.Lobula
    % Lobula layer of the motion detection system
    %   This class represents the Lobula layer of the motion detection
    %   system. It performs several operations including temporal
    %   convolution, correlation, and surround inhibition.
    %
    %   Date: 2024-03-10

    properties
        hDireCell;
    end

    methods
        function self = Lobula()
            % Constructor method
            % Initializes the Lobula object
            
            self = self@smalltargetmotiondetectors.core.feedbackstmd_core.Lobula();

            import smalltargetmotiondetectors.core.backbonev2_core.*;

            self.hDireCell = ...
                smalltargetmotiondetectors.core.backbonev2_core.DirectionCell();
            % self.hGammaDelay.tau = 10;
        end
    end

    methods
        function init_config(self)
            % Initialization method
            % This method initializes the Lobula layer component
            init_config@smalltargetmotiondetectors.core.feedbackstmd_core.Lobula(self);

            self.hDireCell.init_config();

        end

        function [lobulaOpt, direciton] = process(self, varargin)
            % Processing 
            inhiOpt = ...
                process@smalltargetmotiondetectors.core.feedbackstmd_core.Lobula(...
                self, varargin{1}, varargin{2});

            [lobulaOpt, direciton] = self.hDireCell.process(inhiOpt);

            self.Opt = {lobulaOpt, direciton};
        end


    end

end
