classdef Medulla < smalltargetmotiondetectors.core.BaseCore
    %Medulla
    %
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
    % Lamina   LMC1 (L1)            LMC2 (L2)   %
    %           |                    |          %
    % ----------|--------------------|--------- %
    % Medulla   |  {ON}    *         |  {OFF}   %
    %           |          *         v          %
    %           |    /-------<-<--- Tm2         %
    %           v    |     *         |          %
    %          Mi1 --|->->----\      |          %
    %           |    v     *  v      v          %
    %           |   Mi9    * TmOn    |          %
    %           |    |     *  |      |          %
    %           v    v     *  v      v          %
    %          Mi4 <-/     *  \->-> Tm9         %
    %           |          *         |          %
    % ----------|--------------------|--------- %
    % Lobula    v                    v          %
    %                                           %
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
    
    properties
        hMi4;
        hTm9;
    end
    
    methods
        function self = Medulla()
            self = self@smalltargetmotiondetectors.core.BaseCore();
            
            self.hMi4 = ...
                smalltargetmotiondetectors.core.backbonev2_core.MECumulativeCell();
            self.hTm9 = ...
                smalltargetmotiondetectors.core.backbonev2_core.MECumulativeCell();
        end

        function init_config(self)
            self.hMi4.init_config();
            self.hTm9.init_config();
        end
        
        function varargout = process(self, medullaIpt)
            %process Process the input through the Medulla layer.
            %
            % Args:
            %   - medullaIpt (array-like): Input to the Medulla layer.
            %
            % Returns:
            %	- onSignal (array-like): Output signal from Mi4.
            %	- offSignal (array-like): Output signal from Tm9.
            %
            % Process through hMi4 and hTm9
            
            onSignal = max(medullaIpt, 0); % ON
            offSignal = max(-medullaIpt, 0); % OFF
            
            mi4Opt = self.hMi4.process(onSignal, offSignal); % ON
            tm9Opt = self.hTm9.process(offSignal, onSignal); % OFF
            
            self.Opt = {mi4Opt, tm9Opt};
            
            if nargout == 2
                varargout = self.Opt;
            end
            
            
        end
        
        
    end
    
end
















