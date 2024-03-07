function [TP,FN,FP] = compute_metrics( ...
        Prediction,...
        P_type,...
        GroundTruth,...
        GT_type,...
        varargin)
    %COMPUTE_METRICS Computes evaluation metrics.
    %   This function computes evaluation metrics based on predicted and
    %   ground truth data, considering different types of predictions and
    %   ground truth annotations.
    %
    %   Parameters:
    %   - Prediction: Predicted data.
    %   - P_type: Type of prediction ('bbox' or 'logic_matrix').
    %   - GroundTruth: Ground truth data.
    %   - GT_type: Type of ground truth ('bbox' or 'center_dots').
    %   - varargin: Additional optional parameters, including:
    %       - 'ROI_threshold': Threshold for region of interest.
    %       - 'SpatialDistance_threshold': Threshold for spatial distance.
    %
    %   Returns:
    %   - TP: True positives.
    %   - FN: False negatives.
    %   - FP: False positives.
    
    % Define default values
    ROI_threshold = 0.5;
    SpatialDistance_threshold = 5;
    
    % Check if there are additional input arguments
    if ~isempty(varargin)
        for i = 1:2:length(varargin) % Loop through input arguments
            eval([varargin{i},'= varargin{i+1};']);
        end
    end
    
    TP = 0;
    FN = 0;
    FP = 0;

    if strcmp(P_type,'bbox')
        isFP = true(size(Prediction, 1),1);
        for ll = 1:size(GroundTruth, 1)
            isFN = true;
            for kk = 1:size(Prediction, 1)
                if strcmp(GT_type,'bbox')
                    ROI_ = compute_ROI(Prediction(kk,:), GroundTruth(ll,:));
                    isTP =  ROI_ >= ROI_threshold;
                elseif strcmp(GT_type,'center_dots')
                    isTP = Prediction(kk,1)<=GroundTruth(ll,1)...
                        && Prediction(kk,1)+Prediction(kk,3)>=GroundTruth(ll,1)...
                        && Prediction(kk,2)<=GroundTruth(ll,2)...
                        && Prediction(kk,2)+Prediction(kk,3)>=GroundTruth(ll,2);
                end
                if isTP
                    TP = TP + 1;
                    isFN = false;
                    isFP(ll) = false;
                end
            end % end for
            if isFN
                FN = FN + 1;
            end
        end % end for
        FP = FP + sum(isFP);

    elseif strcmp(P_type,'logic_matrix')
        FP = sum(sum(Prediction>0));
        for ll = 1:size(GroundTruth, 1)
            isTP = compute_spatial_distance(...
                Prediction,...
                GroundTruth(ll,:),...
                SpatialDistance_threshold);
            if isTP
                TP = TP + 1;
                FP = FP - 1;
            else
                FN = FN + 1;
            end
        end % end for
    end
end

function ROI = compute_ROI(rect1, rect2)
    %COMPUTE_ROI Computes the region of interest (ROI) between two rectangles.
    %   rect1, rect2: A rectangle region in the format [x, y, w, h], where
    %   x and y are the coordinates of the top-left corner, and w and h are
    %   the width and height.
    
    % Calculate the intersection of two rectangles
    x1 = rect1(1);
    y1 = rect1(2);
    w1 = rect1(3);
    h1 = rect1(4);
    x2 = rect2(1);
    y2 = rect2(2);
    w2 = rect2(3);
    h2 = rect2(4);

    % Determine the coordinates of the top-left and bottom-right corners
    left = max(x1, x2);
    top = max(y1, y2);
    right = min(x1 + w1, x2 + w2);
    bottom = min(y1 + h1, y2 + h2);

    % Calculate the area of intersection
    if left < right && top < bottom
        area = (right - left) * (bottom - top);
    else
        area = 0;
    end

    % Calculate the ROI
    ROI = area / (w1 * h1 + w2 * h2 - area);
end

function isaTP = compute_spatial_distance(...
        Prediction_,...
        GroundTruth_,...
        Spa_Distance_threshold)
    %COMPUTE_SPATIAL_DISTANCE Computes spatial distance between prediction and ground truth.
    %   Prediction_: Predicted data.
    %   GroundTruth_: Ground truth data.
    %   Spa_Distance_threshold: Threshold for spatial distance.
    
    [H,W] = size(Prediction_);
    GroundTruth_ = round(GroundTruth_);
    if size(GroundTruth_,2) == 2
        x1 = max(1,GroundTruth_(1)-Spa_Distance_threshold);
        x2 = min(W,GroundTruth_(1)+Spa_Distance_threshold);
        y1 = max(1,GroundTruth_(2)-Spa_Distance_threshold);
        y2 = min(H,GroundTruth_(2)+Spa_Distance_threshold);
    elseif size(GroundTruth_,2) == 4
        x1 = max(1,GroundTruth_(1)-Spa_Distance_threshold);
        x2 = min(W,GroundTruth_(1)+GroundTruth_(3)+Spa_Distance_threshold);
        y1 = max(1,GroundTruth_(2)-Spa_Distance_threshold);
        y2 = min(H,GroundTruth_(2)+GroundTruth_(4)+Spa_Distance_threshold);
    end
    temp_slice = Prediction_(y1:y2,x1:x2);

    isaTP = any(temp_slice(:)); % There is no response around Ground True
end
