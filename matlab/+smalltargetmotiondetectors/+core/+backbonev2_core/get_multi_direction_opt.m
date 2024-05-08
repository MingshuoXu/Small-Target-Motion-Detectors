function directionOutput = get_multi_direction_opt(...
        modelResponse, modelDirection, numDirection)
    %GET_MULTI_DIRECTION_OUTPUT 
    
    directionOutput = cell(numDirection, 1);
    directionList = (0:numDirection-1) / numDirection * 2 * pi;
    [m, n] = size(modelResponse);

    notNanId = ~isnan(modelDirection);

    notNanSub = find(notNanId);

    notNanInd = sub2ind([m,n], notNanSub);


    for idxDire = 1 : numDirection
        cosDire = cos(modelDirection(notNanInd) - directionList(idxDire));
        cosDire(cosDire<0) = 0;

        Opt = zeros(m, n);
        Opt(notNanInd) = modelResponse(notNanInd) .* cosDire;
        directionOutput{idxDire} = Opt;
    end

end

