classdef Lobula < smalltargetmotiondetectors.core.BaseCore
    % Lobula 
    
    properties
        a = 0;
        b = 0;
        c = 1;
        %{
        The parameters a, b, and c are related to:
                Output = a*ON + b*D[OFF] + c*ON*D[OFF]
        
        Ref: Page 5 in ESTMD
            The final stage of processing is a recombination of the ON and
            OFF channels to form a single output corresponding to the
            ESTMD response. The simplest operation to achieve this would
            be a straightforward sum of the two output signals. However, we
            consider an operation that enhances selectivity for small, dark
            targets. A delay operator D[*], consisting of a low-pass filter
            (LPF5), is applied to the OFF channel prior to recombination 
            with the undelayed ON channel. For generality, we took a 
            phenomenological approach to this recombination allowing 
            second-order as well as linear interactions:
        
            Output = a*ON + b*D[OFF] + c*ON*D[OFF]      (4)
        %}
        
        hSubInhi; % Surround inhibition component
    end
    
    methods
        function self = Lobula()
            % Constructor method
            % Initializes the Lobula object
            
            self = self@smalltargetmotiondetectors.core.BaseCore();
            import smalltargetmotiondetectors.core.*;
            self.hSubInhi = SurroundInhibition();
        end
    end
    
    methods
        function init_config(self)
            % Initialization method
            % Initializes the SurroundInhibition object
            
            self.hSubInhi.init_config();
        end
        
        function [lobulaOpt, correlationOpt] = process(self, varagein)
            % Processing method
            % Processes the input signals to generate the Lobula output
            
            onSignal = varagein{1};
            offSignal = varagein{2};
            correlationOpt = ...
                self.a * onSignal + ...
                self.b * offSignal + ...
                self.c * onSignal .* offSignal;
            
            lobulaOpt = self.hSubInhi.process(correlationOpt);
            self.Opt = lobulaOpt;
        end
    end
    
end
