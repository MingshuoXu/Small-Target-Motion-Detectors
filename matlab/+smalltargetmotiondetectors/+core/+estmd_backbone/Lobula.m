classdef Lobula < smalltargetmotiondetectors.core.BaseCore
    % Lobula layer of the motion detection system
    %   This class represents the Lobula layer of the motion detection
    %   system. It performs a correlation operation on the ON and OFF
    %   channels and then applies surround inhibition.

    properties
        a = 0;
        b = 0;
        c = 1;
        %{
        the parameter a, b, and c are related to :
                Output = a*ON + b*D[OFF] + c*ON*D[OFF]

        ref: Page5 in ESTMD
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

        hSubInhi;  % SurroundInhibition component
    end


    methods
        function self = Lobula()
            % Constructor method
            % Initializes the Lobula object
            
            self = self@smalltargetmotiondetectors.core.BaseCore();
            import smalltargetmotiondetectors.core.SurroundInhibition;
            
            % Initialize the SurroundInhibition component
            self.hSubInhi = SurroundInhibition();
        end
    end

    methods
        function init(self)
            % Initialization method
            % This method initializes the Lobula layer component
            
            self.hSubInhi.init();
        end

        function lobulaOpt = process(self, varagein)
            % Processing method
            % Performs a correlation operation on the ON and OFF channels
            % and then applies surround inhibition
            
            % Extract ON and OFF channel signals from the input
            onSignal = varagein{1};
            offSignal = varagein{2};
            
            % Perform the correlation operation
            correlationOutput = ...
                self.a * onSignal + ...
                self.b * offSignal + ...
                self.c * onSignal .* offSignal;

            % Apply surround inhibition
            lobulaOpt = self.hSubInhi.process(correlationOutput);
            
            % Store the output in Opt property
            self.Opt = lobulaOpt;
        end
    end

end












