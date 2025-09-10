classdef Lobula < smalltargetmotiondetectors.core.BaseCore
    % Lobula Layer of DSTMD
    %   This class implements the Lobula layer of the ESTMD.
    %
    %   Author: [Your Name]
    %   Date: [Date]
    
    properties
        a = 0;  % Parameter a
        b = 0;  % Parameter b
        c = 1;  % Parameter c
        %{
        The parameters a, b, and c are related to the formula:
        Output = a*ON + b*D[OFF] + c*ON*D[OFF]
        
        Reference: Page 5 in ESTMD
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

        Output = a*ON + b*D[OFF] + c*ON*D[OFF]
        %}
    end

    methods
        function self = Lobula()
            % Constructor method
            % Initializes the Lobula object
            
            self = self@smalltargetmotiondetectors.core.BaseCore();
        end
    end

    methods
        function init_config(self)
            % Initialization method
            return;
        end

        function lobulaOpt = process(self, varagein)
            % Processing method
            % Processes the input ON and OFF signals
            
            onSignal = varagein{1};  % Extract ON signal
            offSignal = varagein{2};  % Extract OFF signal
            
            % Compute Lobula output using the provided formula
            lobulaOpt = ...
                self.a*onSignal + ...
                self.b*offSignal + ...
                self.c*onSignal.*offSignal;
            
            self.Opt = lobulaOpt;  % Update Opt property with output
        end
    end

end
