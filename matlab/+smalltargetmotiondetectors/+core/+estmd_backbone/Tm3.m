classdef Tm3 < smalltargetmotiondetectors.core.BaseCore
    % Tm3 Tm3 is On signal with a Surround Inhibition
    %   This class implements the Tm3 component of the motion detection
    %   system. Tm3 receives an input and applies a surround inhibition to
    %   generate the output.
    %

    methods
        function self = Tm3()
            % Constructor method
            % Initializes the Tm3 object
            
            self = self@smalltargetmotiondetectors.core.BaseCore();
        end
    end

    methods
        function init_config(self)
            % Initialization method
            % This method initializes Tm3 (no initialization required)
            return;
        end

        function tm3Opt = process(self, tm3OptIpt)
            % Processing method
            % Applies a surround inhibition to the input to generate the
            % output
            
            tm3Opt = max(tm3OptIpt, 0);  % Apply surround inhibition
            self.Opt = tm3Opt;  % Store the output in Opt property
        end
    end

end
