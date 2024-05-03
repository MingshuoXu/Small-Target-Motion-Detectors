classdef Medulla < smalltargetmotiondetectors.core.BaseCore
    %UNTITLED2 此处提供此类的摘要
    %   此处提供详细说明
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
        hTm1;
        hTm2;
        hTm3;
        hMi1;
    end


    methods
        function self = Medulla()
            self = self@smalltargetmotiondetectors.core.BaseCore();

%             self.hTm1 = indevelopment.core.spikingstmd_core.Tm1();
%             self.hMi1 = indevelopment.core.spikingstmd_core.Mi1();

            self.hTm2 = ...
                smalltargetmotiondetectors.core.backbonev2_core.Tm2();
            self.hTm3 = ...
                smalltargetmotiondetectors.core.backbonev2_core.Tm3();

        end
    end

    methods
        function init_config(self)
%             self.hTm1.init_config();
            self.hTm2.init_config();
            self.hTm3.init_config();
%             self.hMi1.init_config();
        end

        function varargout = process(self, medullaIpt)%, IsSpike)
            
%             if isempty(IsSpike)
%                 IsSpike = false(size(medullaIpt));
%             end

            tm2Signal = self.hTm2.process(medullaIpt);%, IsSpike); % OFF
            tm3Signal = self.hTm3.process(medullaIpt);%, IsSpike); % ON

%             tm1Signal = self.hTm1.process(tm2Signal);

            if nargout == 2
                varargout = {tm3Signal, tm2Signal};
            end
            
            self.Opt = {tm3Signal, tm2Signal};
        end


    end

end
















