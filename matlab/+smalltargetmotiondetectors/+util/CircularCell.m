classdef CircularCell < handle
    %CellRecording Gamma filter in lamina layer
    %
    %   Author: Mingshuo Xu
    %   Date: 2024-03-18
    %   LastEditTime: 2024-04-28

    properties
        len;    % Length of cell
        data;   % data of cell
        point;  % point of cell
    end
   

    methods
        function self = CircularCell(lenCell)
            % Constructor
            % Initializes the GammaDelay object with provided parameters
            if nargin == 1
                self.len = lenCell;
                self.init_config();
            end
            
        end

        function init_config(self)
            if isempty(self.len)
                self.len = 2;
            end
            self.data = cell(self.len, 1);
            self.point = self.len;
        end
            
        function move_point(self)
            if self.point == 1
                self.point = self.len;
            else
                self.point = self.point - 1;
            end
        end

        function circrecord(self, iptMatrix)
            self.move_point();
            self.cover(iptMatrix);
        end  
        
        function cover(self, iptMatrix)
            self.data{self.point} = iptMatrix;
        end

    end
end
