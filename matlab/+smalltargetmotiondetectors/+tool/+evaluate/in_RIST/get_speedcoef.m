function speedCoef = get_speedcoef(datasetName)
    % get_speedcoef: Returns the speed coefficient for the specified dataset.
    %
    %   INPUT:
    %       datasetName: Name of the dataset.
    %
    %   OUTPUT:
    %       speedCoef: Speed coefficient for the specified dataset.
    %
    %   Example:
    %       speedCoef = get_speedcoef('GX010071-1');
    %
    %   See also <related functions or scripts>.

    switch datasetName
        case 'GX010071-1'
            speedCoef = 15.221405;
        case 'GX010220-1'
            speedCoef = 10.826480;
        case 'GX010228-1'
            speedCoef = 8.387578;
        case 'GX010230-1'
            speedCoef = 7.232824;
        case 'GX010231-1'
            speedCoef = 10.158651;
        case 'GX010241-1'
            speedCoef = 7.342868;
        case 'GX010250-1'
            speedCoef = 5.807466;
        case 'GX010266-1'
            speedCoef = 5.014323;
        case 'GX010290-1'
            speedCoef = 5.992503;
        case 'GX010291-1'
            speedCoef = 9.084210;
        case 'GX010303-1'
            speedCoef = 8.176533;
        case 'GX010307-1'
            speedCoef = 9.711653;
        case 'GX010315-1'
            speedCoef = 18.698583;
        case 'GX010321-1'
            speedCoef = 17.980894;
        case 'GX010322-1'
            speedCoef = 15.813347;
        case 'GX010327-1'
            speedCoef = 17.473148;
        case 'GX010335-1'
            speedCoef = 8.037636;
        case 'GX010336-1'
            speedCoef = 22.032406;
        case 'GX010337-1'
            speedCoef = 10.716846;
        otherwise
            error('Please input the correct dataset name.');
    end
    speedCoef = speedCoef / 25;
end
