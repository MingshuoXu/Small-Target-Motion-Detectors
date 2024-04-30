function D = compute_pdist2(X, Y)
    % Compute the distance matrix between two sets of data.

    % Get the dimensions of input data
    [m, n] = size(X);
    [p, q] = size(Y);

    % Check if the dimensions of input data match
    if n ~= q
        error('The number of columns in the input data does not match.');
    end

    % Initialize the distance matrix
    D = zeros(m, p);

    % Compute the distance matrix using vectorized calculation
    for i = 1:m
        D(i, :) = sqrt(sum((X(i, :) - Y).^2, 2)');
    end
end
