function directionOutput = get_multi_direction_opt(...
        modelResponse, modelDirection, numDirection)
    %GET_MULTI_DIRECTION_OUTPUT 
    
    directionOutput = cell(numDirection, 1);
    directionList = (0:numDirection-1) / numDirection * 2 * pi;
    [m, n] = size(modelResponse);

    notNanId = ~isnan(modelDirection);

    nanSub = find(notNanId);

    nanInd = sub2ind([m,n], nanSub);


    for idxDire = 1 : numDirection
        cosDire = cos(modelDirection(nanInd) - directionList(idxDire));
        cosDire(cosDire<0) = 0;

        Opt = zeros(m, n);
        Opt(nanInd) = modelResponse(nanInd) .* cosDire;
        directionOutput{idxDire} = Opt;
    end

end

