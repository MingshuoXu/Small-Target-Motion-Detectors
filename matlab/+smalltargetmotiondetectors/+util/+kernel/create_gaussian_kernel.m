function gaussianFilter = create_gaussian_kernel(size, sigma)
    % 计算高斯滤波器的半径
    radius = (size - 1) / 2;

    [x, y] = meshgrid(-radius:radius, -radius:radius);

    gaussianFilter = exp( -(x.^2 + y.^2) / (2 * sigma^2) );
    
    gaussianFilter = gaussianFilter / sum(gaussianFilter, 'all');
end
