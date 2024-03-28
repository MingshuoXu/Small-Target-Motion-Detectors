function Opt = slice_matrix_holding_size(input, shiftX, shiftY)
    
    if ~isinteger(shiftX)
        shiftX = round(shiftX);
    end
    if ~isinteger(shiftY)
        shiftY = round(shiftY);
    end

    [m, n] = size(input);
    if abs(shiftX) > n || abs(shiftY) > m
        Opt = zeros(m, n);
        return;
    end

    Opt = circshift(input, [shiftX, shiftY]);

    if shiftX > 0
        Opt(:, 1:shiftX) = 0;
    else
        Opt(:, end+shiftX+1:end) = 0;
    end

    if shiftY > 0
        Opt(1:shiftY, :) = 0;
    else
        Opt(end+shiftY+1:end, :) = 0;
    end

end