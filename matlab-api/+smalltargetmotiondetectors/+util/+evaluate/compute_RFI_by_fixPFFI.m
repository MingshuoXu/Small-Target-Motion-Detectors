function [RFI] = get_RFI_by_fixPFFI(...
        modelOnput, ...
        datasetIndex, ...
        aimPFFI, ...
        groundTurthErrorDistance, ...
        startFrame, ...
        endFrmae )
    if ~exist('groundTurthErrorDistance','var')
        r = 5; % 距离阈值
    else
        r = groundTurthErrorDistance;
    end
    if ~exist('aimPFFI','var')
        aimPFFI = 1;
    end
    if size(datasetIndex,2) ~= 3
        error('size(Index,2) ~= 3');
    end
    
    [m,n,T] = size(modelOnput);
    modelOnput = zeros(m,n,T);
   
    
    thresholdTop = 1;
    thresholdDown = 0;
    thresholdValue = 0.5;
    preThresholdValue = 0;
    
    %%
    while true
        %         fprintf('threshold_value = %f\n',threshold_value);%%%%%%%%%%%%%%%%%%
        
        threInput = modelOnput;
        threInput(threInput < thresholdValue) = 0;%阈值截断
        
        numTrueDetections = 0;
        number_of_actual_target = 0; % Single small moving target
        numFalseDetection = 0;
        number_of_image = T;
        
        s = 1;
        while datasetIndex(s,1) < startFrame
            s = s + 1;
        end
        
        for t = 1:T
            while t == datasetIndex(s,1) - startFrame + 1
                index_x = datasetIndex(s,2);
                index_y = datasetIndex(s,3);
                s = s + 1;
                
                threInputT = threInput(:,:,t);
                numDetection = sum(sum(threInputT > 0));
                number_of_actual_target = number_of_actual_target + 1;
                
                sliceT = threInputT( ...
                    max(index_x-r,1):min(index_x+r,m), ...
                    max(index_y-r,1):min(index_y+r,n) ...
                    );
                if ~any(sliceT(:)) % There is no response around Ground True
                    numFalseDetection = numFalseDetection + numDetection;
                else
                    numTrueDetections = numTrueDetections + 1;
                    numFalseDetection = numFalseDetection + numDetection - 1;
                end
                
            end % end while
        end % end for

        % RPI(Recall Per Image / True Positive)
        RFI = numTrueDetections / number_of_actual_target;
        % FPPI(False Positive Per Image)
        FPPI = numFalseDetection / number_of_image;
        
        if abs(FPPI - aimPFFI) < 0.01 ...
                || abs(thresholdValue - preThresholdValue) < 1e-5
            break;
        else
            if FPPI < aimPFFI
                thresholdTop = thresholdValue;
            else
                thresholdDown = thresholdValue;
            end
            preThresholdValue = thresholdValue;
            thresholdValue = (thresholdTop + thresholdDown)/2;
        end
    end
end