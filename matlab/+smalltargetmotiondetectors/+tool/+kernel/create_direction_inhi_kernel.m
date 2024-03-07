function directionalInhiKernel = ...
                create_direction_inhi_kernel(...
                direction,Sigma1,Sigma2)

            % Function Description
            % This function generates lateral inhibition kernels along the Theta direction.
            % We adopt a one-dimensional DoG as the lateral inhibition kernel function here.

            %% Parameter Setting

            if ~exist('DSTMD_Directions','var')
                direction = 8;
                % The number of directions should be 4*k, where k=1,2,3,4,5,6...
            end

            if ~exist('Sigma1','var')
                Sigma1 = 1.5;
            end

            if ~exist('Sigma2','var')
                Sigma2 = 3.0;
            end

            KernelSize = direction;

            %% Sampling for DoG

            % Calculate two points where DoG equals zero
            Zero_Point_DoG_X1 = ...
                -sqrt((log(Sigma2/Sigma1)*2*Sigma1^2*Sigma2^2)...
                /(Sigma2^2-Sigma1^2));
            Zero_Point_DoG_X2 = -Zero_Point_DoG_X1;
            % Calculate two points where DoG reaches its minimum value
            Min_Point_DoG_X1 = ...
                -sqrt((3*log(Sigma2/Sigma1)*2*Sigma1^2*Sigma2^2)...
                /(Sigma2^2-Sigma1^2));
            Min_Point_DoG_X2 = -Min_Point_DoG_X1;

            % Set the size of the convolution kernel to be odd
            if mod(KernelSize,2) == 0
                KernelSize = KernelSize +1;
            end

            Half_Kernel_Size = (KernelSize-1)/2;
            Quarter_Kernel_Size = (KernelSize-1)/4;
            % Sampling interval in the central area (>0 part)
            Center_Range_DoG = Zero_Point_DoG_X2-Zero_Point_DoG_X1;
            Center_Step = Center_Range_DoG/Half_Kernel_Size;
            % Sampling interval in the surrounding area (<0 part)
            Surround_Range_DoG = Min_Point_DoG_X2-Zero_Point_DoG_X2;
            Surround_Step = 2*Surround_Range_DoG/Quarter_Kernel_Size;
            % Integration of sampling ranges
            X_Smaller = Zero_Point_DoG_X1-(Quarter_Kernel_Size:-1:1)*Surround_Step;
            X_Larger = Zero_Point_DoG_X2+(1:Quarter_Kernel_Size)*Surround_Step;
            X_Center = Zero_Point_DoG_X1+(0:Half_Kernel_Size)*Center_Step;
            X = [X_Smaller,X_Center,X_Larger];
            % Sampling
            Gauss1 = (1/(sqrt(2*pi)*Sigma1))*exp(-(X.^2)/(2*Sigma1^2));
            Gauss2 = (1/(sqrt(2*pi)*Sigma2))*exp(-(X.^2)/(2*Sigma2^2));

            Inhibition_Kernel = Gauss1 - Gauss2;

            %Inhibition_Kernel = Inhibition_Kernel(1:DSTMD_Directions);
            directionalInhiKernel = reshape(Inhibition_Kernel,[1 1 KernelSize]);
        end
