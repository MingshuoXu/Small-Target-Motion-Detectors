classdef AttentionModule < smalltargetmotiondetectors.core.BaseCore
    % AttentionModule class for attention mechanism
    %   This class implements the attention mechanism module in the ApgSTMD.
    
    properties
        kernalSize = 17;
        zetaList = [2, 2.5, 3, 3.5];
        thetaList = [0, pi/4, pi/2, pi*3/4];
        alpha = 1;
    end
    
    properties(Hidden)
        attentionKernal;
    end
    
    methods
        function self = AttentionModule()
            % Constructor method
            % Initializes the AttentionModule object
            
            self = self@smalltargetmotiondetectors.core.BaseCore();
        end
    end
    
    methods
        function init(self)
            % Initialization method
            % Initializes the attention kernel
            
            import smalltargetmotiondetectors.tool.kernel.*;
            self.attentionKernal = create_attention_kernal(...
                self.kernalSize, ...
                self.zetaList, ...
                self.thetaList);
        end
        
        function attentionOpt = process(self, retinaOpt, predictionMap)
            % Processing method
            % Processes the retinaOpt and predictionMap to generate the
            % attention-optimal output
            
            [r, s] = size(self.attentionKernal);
            
            if isempty(predictionMap)
                attentionOpt = retinaOpt;
            else
                mapRetinaOpt = retinaOpt .* predictionMap;
                
                for i = 1:r
                    A_j = conv2(mapRetinaOpt, self.attentionKernal{i, 1}, 'same');
                    for j = 2:s
                        A_j = min(A_j, ...
                            conv2(mapRetinaOpt, self.attentionKernal{i, j}, 'same'));
                    end
                    
                    if i == 1
                        attentionResponse = A_j;
                    else
                        attentionResponse = max(attentionResponse, A_j);
                    end
                end
                
                attentionOpt = retinaOpt + self.alpha * attentionResponse;
                
                self.Opt = attentionOpt;
            end
        end
    end
end
