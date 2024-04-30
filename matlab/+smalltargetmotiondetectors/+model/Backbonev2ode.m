classdef Backbonev2ode < smalltargetmotiondetectors.model.Backbonev2
    %


    properties
        mi4Signal;
        tm9Signal;
        mi9Signal;
        tmOnSignal;
    end
    properties(Hidden)

    end

    methods
        % Constructor function
        function self = Backbonev2ode()
            self@smalltargetmotiondetectors.model.Backbonev2();

            import smalltargetmotiondetectors.core.backbonev2_core.*;
            
        end

        function init_config(self)
            return;
        end

        function model_structure(self, modelIpt)

            %%
            self.retinaOpt = self.hRetina.process(modelIpt);

            %%
            self.laminaOpt = self.hLamina.process(self.retinaOpt);

            %%
            self.fun_medulla_process();
            self.medullaOpt = {self.mi4Signal, self.tm9Signal};


            %%
            [self.lobulaOpt, self.modelOpt.direction] = ...
                self.hLobula.process(self.medullaOpt);

            %%
            self.modelOpt.response = self.lobulaOpt;
        end % [EoF] end of function

        function fun_medulla_process(self)

            % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
            %                                           %
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




            mi1Signal = max(self.laminaOpt, 0); % On
            tm2Signal = max(-self.laminaOpt, 0);

            self.mi4Signal = (-self.mi4Signal + mi1Signal - self.mi9Signal);
            self.tm9Signal = (-self.tm9Signal + tm2Signal - self.tmOnSignal);
            

            %%

            self.mi9Signal = tm2Signal;
            self.tmOnSignal = mi1Signal;
            
        end

    end % [EoM] end of methods

end

