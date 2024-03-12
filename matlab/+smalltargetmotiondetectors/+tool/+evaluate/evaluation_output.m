function [listTP,listFN,listFP] = ...
        evaluation_output(result, bboxData, modelType)
    % evaluation_output: Evaluates the performance of the detection results.
    %
    %   INPUT:
    %       result: Cell array containing the detection results.
    %       bboxData: Ground truth bounding box data.
    %       modelType: Type of the detection model ('SOD' or 'STMD').
    %           SOD:    small target detection (target detection)
    %           STMD:   small target motion detection (motion detection)
    %
    %   OUTPUT:
    %       listTP: List of true positives.
    %       listFN: List of false negatives.
    %       listFP: List of false positives.
    %
    %   See also compute_metrics.

    % Import necessary modules
    import smalltargetmotiondetectors.tool.evaluate.*;

    % Initialization
    frame = 1;
    totalIdx = size(bboxData,1);
    [listTP,listFN,listFP] = deal(zeros(totalIdx,1));

    
    % Iterate over each frame
    for timeIdx = 1:size(result,1)
        GroundTruth = [];

        % Extract ground truth bounding boxes for the current frame
        while bboxData(frame,5) == timeIdx
            GroundTruth = [GroundTruth; bboxData(timeIdx,1:4)]; %#ok<AGROW>
            if frame < totalIdx
                frame = frame + 1;
            else
                break;
            end
        end

        % Compute metrics based on the detection model type
        if strcmp(modelType,'SOD')
            [TP, FN, FP] = compute_metrics(...
                result{timeIdx}, ...
                'bbox', ...
                GroundTruth, ...
                'bbox');
        elseif strcmp(modelType,'STMD')

            fullMatrix = full(result{timeIdx});
            
            [TP, FN, FP] = compute_metrics(...
                fullMatrix, ...
                'logic_matrix', ...
                GroundTruth, ...
                'bbox',...
                'SpatialDistance_threshold', ...
                1);
        end

        % Store the metrics for the current frame
        listTP(timeIdx) = TP;
        listFN(timeIdx) = FN;
        listFP(timeIdx) = FP;
    end
end
