classdef Lobula < smalltargetmotiondetectors.core.BaseCore
    %UNTITLED2 此处提供此类的摘要
    %   此处提供详细说明

    properties
        hSubInhi;
        hDireCell;
        % spikingThreshold = 6000;
    end

    properties(Hidden)
        
    end


    methods
        function self = Lobula()
            self = self@smalltargetmotiondetectors.core.BaseCore();
            import smalltargetmotiondetectors.core.math_operator.*;
            import smalltargetmotiondetectors.core.backbonev2_core.*;

            self.hSubInhi = SurroundInhibition();
            self.hDireCell = ...
                smalltargetmotiondetectors.core.backbonev2_core.DirectionCell();
        end
    end

    methods
        function init_config(self)
            self.hSubInhi.init_config();
            self.hDireCell.init_config();
        end

        %function [lobulaOpt, IsSpike] = process(self, varagein)
        function varargout = process(self, varagein)
            onSignal = varagein{1};
            offSignal = varagein{2};

            correlationOutput = onSignal.*offSignal;

%             IsSpike = correlationOutput > self.spikingThreshold;

%             lobulaOpt = ...
%                 self.hSubInhi.process(correlationOutput.*IsSpike);
            % lobulaOpt = self.hSubInhi.process(correlationOutput);

            inhiOpt = self.hSubInhi.process(correlationOutput);

            
            if nargout == 1
                varargout = {inhiOpt}; 
            elseif nargout == 2
                [lobulaOpt, direciton] = self.hDireCell.process(inhiOpt);
                varargout = {lobulaOpt, direciton};
            elseif nargout == 3
                [lobulaOpt, direciton] = self.hDireCell.process(inhiOpt);
                varargout = {lobulaOpt, direciton, correlationOutput};
            end
            
            self.Opt = varargout;
        end

    end

end
















