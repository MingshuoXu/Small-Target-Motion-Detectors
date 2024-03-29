classdef Tm3 < smalltargetmotiondetectors.core.BaseCore
    % Tm3 
    
    properties
        hSubInhi;  % Handle to the SurroundInhibition object
    end

    methods
        function self = Tm3()
            % Constructor method
            % Initializes the Tm3 object
            
            self = self@smalltargetmotiondetectors.core.BaseCore();
            import smalltargetmotiondetectors.core.SurroundInhibition;
            
            self.hSubInhi = SurroundInhibition();  % Initialize SurroundInhibition object
        end
    end

    methods
        function init(self)
            % Initialization method
            % Initializes the SurroundInhibition object
            
            self.hSubInhi.init();
        end

        function tm3Opt = process(self, iptMatrix)
            % Processing method
            % Applies Surround Inhibition to the On-signal matrix iptMatrix
            
            onSignal = max(iptMatrix, 0);  % Extracts the On-signal from iptMatrix
            tm3Opt = self.hSubInhi.process(onSignal);  % Processes the On-signal using SurroundInhibition
            self.Opt = tm3Opt;
        end
    end

end
