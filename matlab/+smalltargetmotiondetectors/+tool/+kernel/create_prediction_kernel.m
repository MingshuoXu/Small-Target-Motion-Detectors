function PredictionKernal = create_prediction_kernel(...
        Vel, Delta_t, filter_size, FilterNum, zeta, eta)
    % CREATE_PREDICTION_KERNAL Creates prediction kernels for motion detection.
    %   This function generates prediction kernels for motion detection based
    %   on the specified parameters.
    %
    %   Parameters:
    %   - Vel: Velocity of the moving object.
    %   - Delta_t: Time interval.
    %   - filter_size: Size of the filter.
    %   - FilterNum: Number of filters.
    %   - zeta: Zeta parameter.
    %   - eta: Eta parameter.
    %
    %   Returns:
    %   - PredictionKernal: Cell array containing prediction kernels.

    % Set default values if not provided
    if nargin < 1
        Vel = 0.25;
    end
    if nargin < 2
        Delta_t = 25;
    end
    if nargin < 3
        filter_size = 25;
    end
    if nargin < 4
        FilterNum = 8;
    end
    if nargin < 5
        zeta = 2;
    end
    if nargin < 6
        eta = 2.5;
    end

    % Initialize the cell array for prediction kernels
    PredictionKernal = cell(FilterNum, 1);
    [PredictionKernal{:}] = deal(zeros(filter_size, filter_size));
    Center = ceil(filter_size/2);

    % Create meshgrid
    [ShiftX, ShiftY] = ...
        meshgrid((1:filter_size) - Center, (filter_size:-1:1) - Center);

    % Compute angle
    fai = atan2(ShiftY, ShiftX);
    
    % Compute delta X and delta Y
    Delta_X = Vel * Delta_t * cos(fai);
    Delta_Y = Vel * Delta_t * sin(fai);

    % Generate prediction kernels
    for idx = 1:FilterNum
        theta = (idx - 1) * 2 * pi / FilterNum;

        PredictionKernalWithIdx = exp(...
            -((ShiftX - Delta_X).^2 + (ShiftY - Delta_Y).^2) / (2 * zeta^2) ...
            + eta * cos(fai - theta) ...
            );
     
        % normalization
        PredictionKernalWithIdx = ...
            PredictionKernalWithIdx / sum(PredictionKernalWithIdx, 'all');

        % To speed up the calculation
        PredictionKernalWithIdx(PredictionKernalWithIdx < 5e-4) = 0;
        
        % normalizating again
        PredictionKernal{idx} = ...
            PredictionKernalWithIdx / sum(PredictionKernalWithIdx, 'all');

    end
end
