function attentionKernal = create_attention_kernal(...
        kernal_size, Zeta, Theta)
    %CREATE_ATTENTION_KERNEL Summary of this function goes here
    %   Detailed explanation goes here
    
    % Default values if not provided
    if nargin < 1
        kernal_size = 17;
    elseif mod(kernal_size,2) == 0
        kernal_size = kernal_size + 1;
    end
    if nargin < 2
        Zeta = [2, 2.5, 3, 3.5];
    end
    if nargin < 3
        Theta = [0, pi/4, pi/2, pi*3/4];
    end

    % Initialize the cell array to store the attention kernels
    r = length(Zeta);
    s = length(Theta);
    attentionKernal = cell(r,s);

    % Calculate the center of the kernel
    Center = ceil(kernal_size/2);

    % Create meshgrid for kernel indices
    [Y,X] = meshgrid(1:kernal_size, 1:kernal_size);
    % Shift the meshgrid to center it
    ShiftX = X - Center;
    ShiftY = Y - Center;

    % Generate attention kernels for each combination of Zeta and Theta
    for i = 1:r
        for j = 1:s
            % Formula to generate the attention kernel
            attentionKernalWithIJ =  2 / pi / Zeta(i)^4 ...
                .* ( ...
                Zeta(i)^2 - ...
                (ShiftX*cos(Theta(j)) + ShiftY*sin(Theta(j))).^2 ...
                ) ...
                * exp( -(ShiftX.^2+ShiftY.^2)/2/Zeta(i)^2 );
            attentionKernal{i, j} = attentionKernalWithIJ / ...
                sum(attentionKernalWithIJ, 'all');
        end
    end

end
