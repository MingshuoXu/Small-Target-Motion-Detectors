classdef Tm2 < smalltargetmotiondetectors.core.BaseCore
    % Tm2 Medulla Layer Neurons in ESTMD
   
    properties
        hSubInhi;  % Handle to the SurroundInhibition object
    end

    methods
        function self = Tm2()
            % Constructor method
            % Initializes the Tm2 object
            
            self = self@smalltargetmotiondetectors.core.BaseCore();
            import smalltargetmotiondetectors.core.math_operator.*;
            
            % Initialize SurroundInhibition object
            self.hSubInhi = SurroundInhibition();  
        end
    end

    methods
        function init_config(self)
            % Initialization method
            % Initializes the SurroundInhibition object
            
            self.hSubInhi.init_config();
        end

        function tm2Opt = process(self, iptMatrix)
            % Processing method
            % Applies the Surround Inhibition mechanism to the input matrix iptMatrix
            
            offSignal = max(-iptMatrix, 0);  % Extracts the OFF signal from iptMatrix
            % Processes the OFF signal using SurroundInhibition
            tm2Opt = self.hSubInhi.process(offSignal);  
            self.Opt = tm2Opt;
        end
    end

end
