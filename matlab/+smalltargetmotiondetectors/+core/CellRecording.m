classdef CellRecording < smalltargetmotiondetectors.core.BaseCore
    %CellRecording Gamma filter in lamina layer
    %
    %   Author: Mingshuo Xu
    %   Date: 2024-03-18
    %   LastEditTime: 2024-03-18

    properties
        lenCell;    % Length of cell
        isCircshift = true; % Flag to indicate circular shifting
        cellData;
    end
   

    methods
        function self = CellRecording(lenCell)
            % Constructor
            % Initializes the GammaDelay object with provided parameters
            self = self@smalltargetmotiondetectors.core.BaseCore();
            
            if nargin >= 1
                self.lenCell = lenCell;
            end
    
        end

        function init(self)
            % Initialization method

            if isempty(self.lenCell)
                self.lenCell = 2;
            end
            
            self.cellData = cell(self.lenCell, 1);
        end

        function circshift(self)
            self.cellData = circshift(self.cellData, -1);
        end
            
        function record(self, iptMatrix)
            self.cellData{end} = iptMatrix;
        end  
        
        function push(self, iptMatrix)
            if self.isCircshift
                self.circshift()
            end
            self.record(iptMatrix);
        end 

        function process(self, iptMatrix)
            self.push(iptMatrix);
        end


    end
end
