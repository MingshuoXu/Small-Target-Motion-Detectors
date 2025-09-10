classdef Tm2 < smalltargetmotiondetectors.core.BaseCore
    % Tm2 

    methods
        function self = Tm2()
            % Constructor method
            % Initializes the Tm2 object
            
            self = self@smalltargetmotiondetectors.core.BaseCore();
        end
    end

    methods
        function init_config(self)
            % Initialization method
            % This method initializes Tm2 (no initialization required)
        end

        function tm2Opt = process(self, tm2Ipt)
            % Processing method
            % Applies surround inhibition to the input to generate the
            % output
            
            tm2Opt = max(-tm2Ipt, 0);  % Apply surround inhibition
            self.Opt = tm2Opt;  % Store the output in Opt property
        end

    end

end
