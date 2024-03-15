function attentionKernal = create_attention_kernal(...
        kernalSize, zeta, theta)
    %CREATE_ATTENTION_KERNEL Summary of this function goes here
    %   Detailed explanation goes here
    
    % Default values if not provided
    if nargin < 1
        kernalSize = 17;
    elseif mod(kernalSize,2) == 0
        kernalSize = kernalSize + 1;
    end
    if nargin < 2
        zeta = [2, 2.5, 3, 3.5];
    end
    if nargin < 3
        theta = [0, pi/4, pi/2, pi*3/4];
    end

    
    % Initialize the cell array to store the attention kernels
    r = length(zeta);
    s = length(theta);
    attentionKernal = cell(r,s);

    % Calculate the center of the kernel
    Center = ceil(kernalSize/2);

    % Create meshgrid for kernel indices
    [ShiftX, ShiftY] = meshgrid((1:kernalSize) - Center, (kernalSize:-1:1) - Center);

    % Generate attention kernels for each combination of Zeta and Theta
    for i = 1:r
        for j = 1:s
            % Formula to generate the attention kernel
            attentionKernalWithIJ =  2 / pi / zeta(i)^4 ...
                .* ( ...
                zeta(i)^2 - ...
                (ShiftX*cos(theta(j)+pi/2) + ShiftY*sin(theta(j)+pi/2)).^2 ...
                ) ...
                .* exp( -(ShiftX.^2+ShiftY.^2)/2/zeta(i)^2 );

            attentionKernalWithIJ(abs(attentionKernalWithIJ) < 1e-4) = 0;
            attentionKernal{i, j} = attentionKernalWithIJ;
        end
    end

end
