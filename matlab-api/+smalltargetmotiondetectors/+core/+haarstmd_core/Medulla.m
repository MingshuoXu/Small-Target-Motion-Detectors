classdef Medulla < smalltargetmotiondetectors.core.BaseCore
    %Medulla Medulla layer for haarSTMD
    
    properties
        lenTemporalKernel = 15;
        sizeSpacialKernel = [8,16];
        cp = 15; % a parameter to adjust the spacialOn and spacialOff
        theta = pi;
        
    end
    
    
    properties(Hidden)
        spacialOnKernel;
        spacialOffKernel;
        temporalOnKernel;
        temporalOffKernel;
        
        cellSpatialOpt; % Parameter_Residual.DLSTMD_SpatialSum
        cellMedullaIpt; % Parameter_Residual.laminaL
    end
    
    
    methods
        function self = Medulla()
            self = self@smalltargetmotiondetectors.core.BaseCore();
            
            import smalltargetmotiondetectors.util.*;
            
            self.cellSpatialOpt = CircularCell();
            self.cellMedullaIpt = CircularCell();
            
        end % [EoF]
        
        function init_config(self)
            % init spacial kernel
            m1 = self.sizeSpacialKernel(1);
            n1 = ceil(self.sizeSpacialKernel(2)/2);
            n2 = self.sizeSpacialKernel(2) - n1;
            
            self.spacialOnKernel = [ones(m1, n1); zeros(m1, n2)];
            self.spacialOffKernel = [zeros(m1, n1); -ones(m1, n2)];
            
            % init temporal kernel
            k1 = ceil(self.lenTemporalKernel/2);
            k2 = self.lenTemporalKernel - k1;
            
            self.temporalOnKernel = ones(k1,1);
            self.temporalOffKernel = [zeros(k1,1); -ones(k2,1)];
            
            
            % Allocate memory
            self.cellSpatialOpt.len = self.lenTemporalKernel;
            self.cellSpatialOpt.init_config();
            
            self.cellMedullaIpt.len = self.lenTemporalKernel;
            self.cellMedullaIpt.init_config();
            
        end % [EoF]
        
        function varargout = process(self, medullaIpt)
            import smalltargetmotiondetectors.util.compute.*;
            
            %% compute spacial part
            % SP_ON
            spacialOnOpt = max(...
                conv2(medullaIpt, self.spacialOnKernel, 'same'), ...
                0);
            
            % SP_OFF
            spacialOffOpt = max(...
                conv2(medullaIpt, self.spacialOffKernel, 'same'), ...
                0);
            
            % SP
            nowSpacialOpt = compute_spacial_correlation(...
                spacialOnOpt, ...
                spacialOffOpt, ...
                self.cp, ...
                self.theta);
            nowSpacialOpt = nowSpacialOpt / max(nowSpacialOpt(:));
            % record Spatial output by cell
            %       (Parameter_Residual.DLSTMD_SpatialSum)
            self.cellSpatialOpt.circrecord(nowSpacialOpt);
            
            %% Compute temporal part
            
            % record Medulla input (Lamina output) by cell
            %       (Parameter_Residual.DLSTMD_SpatialSum)
            self.cellMedullaIpt.circrecord(medullaIpt);
            
            % TP_ON
            temporalOnOpt = max(...
                compute_circularcell_conv(self.cellMedullaIpt, self.temporalOnKernel), ...
                0);
            
            % TP_OFF
            temporalOffOpt = max(...
                compute_circularcell_conv(self.cellMedullaIpt, self.temporalOffKernel), ...
                0);
            
            % TP
            % There's no need for half-wave rectification here
            temporalOpt = temporalOnOpt .* temporalOffOpt;
            
            %% output
            
            if nargout == 2
                varargout = {self.cellSpatialOpt, temporalOpt};
            elseif nargout > 2 % used for test pattern
                varargout = {...
                    self.cellSpatialOpt, ...
                    temporalOpt, ...
                    spacialOnOpt, ...
                    spacialOffOpt, ...
                    temporalOnOpt, ...
                    temporalOffOpt ...
                    };
            end
            self.Opt = struct(...
                'cellSpatialOpt', self.cellSpatialOpt, ...
                'temporalOpt', temporalOpt);
        end % [EoF]
        
    end % [EoM]
    
end % [EoC]

%classmethod
function spacialOpt = compute_spacial_correlation(...
        spacialOnOpt, ...
        spacialOffOpt, ...
        alpha, ...
        theta ...
        )
    
    spacialOpt = zeros(size(spacialOnOpt));
    
    dColumn = round(alpha*cos(theta));
    if theta <= pi
        dLine = -abs(round(alpha*sin(theta)));
    else
        dLine = abs(round(alpha*sin(theta)));
    end
    
    bw = round(alpha); % bw = round(alpha*sin(pi/2));
    
    spacialOpt(1+bw:end-bw,1+bw:end-bw)...
        = spacialOnOpt(1+bw:end-bw, 1+bw:end-bw) ...
        .* spacialOffOpt(1+bw+dLine:end-bw+dLine, 1+bw+dColumn:end-bw+dColumn);
end
















