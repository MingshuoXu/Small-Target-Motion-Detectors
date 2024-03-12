function frameRange = get_RIST_frame_range(datasetName)
    % get_RIST_frame_range: Returns the frame range for the specified dataset.
    %
    %   INPUT:
    %       datasetName: Name of the dataset.
    %
    %   OUTPUT:
    %       frameRange: Frame range for the specified dataset.
    %
    %   Example:
    %       frameRange = get_RIST_frame_range('GX010071-1');
    %
    %   See also <related functions or scripts>.

    switch datasetName
        case 'GX010071-1'
            frameRange = [1:1300]';
        case 'GX010220-1'
            frameRange = [1:1300]';
        case 'GX010228-1'
            frameRange = [1:1300]';
        case 'GX010230-1'
            frameRange = [1:2400]';
        case 'GX010231-1'
            frameRange = [1:2400]';
        case 'GX010241-1'
            frameRange = [1:3600]';
        case 'GX010250-1' % <2000
            frameRange = [1:2000]';%[550:1150]'
        case 'GX010266-1'
            frameRange = [1:2400]';
        case 'GX010290-1'
            frameRange = [1:1300]';
        case 'GX010291-1'
            frameRange = [1:1300]';
        case 'GX010303-1'
            frameRange = [1:2400]';
        case 'GX010307-1'
            frameRange = [1:1000]';
        case 'GX010315-1'
            frameRange = [1:1000]';
        case 'GX010321-1'
            frameRange = [1:1000]';
        case 'GX010322-1'
            frameRange = [1:1300]';
        case 'GX010327-1'
            frameRange = [1:900]';
        case 'GX010335-1'
            frameRange = [1:1300]';
        case 'GX010336-1'
            frameRange = [1:1000]';
        case 'GX010337-1'
            frameRange = [1:700]';
        otherwise
            error('Please input the correct dataset name.');
    end
end
