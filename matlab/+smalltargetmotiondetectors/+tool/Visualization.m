classdef Visualization < handle
    % a class to visualize the model output
    %
    %   Author: Mingshuo Xu
    %   Date: 2022-01-10
    %   LastEditTime: 2024-03-10
    
    properties
        hFig; % figure handle
        showThreshold = 0.8;
        isSaveAsVideo = false;
        savePath;
        videoPath;
        inputClassName;
        isTestPatter = false;
        paraNMS = struct( ...
            'maxRegionSize', 15, ...
            'method', 'sort');
    end
    properties(Hidden)
        saveState = false;
        hVideo;
        timeTic;
        hNMS;
        hasFigHandle = false;
        uiHandle;
    end
    
    methods
        function self = Visualization(className, showThreshold)
            if nargin > 0
                self.inputClassName = className;
                if nargin > 1
                    self.showThreshold = showThreshold;
                end
            elseif nargin == 0
                self.inputClassName = 'None';
            end

        end

        function create_fig_handle(self)
            self.hFig = figure(...
                'Name', ['Show result for ', self.inputClassName], ...
                'NumberTitle', 'off');

            if ~self.isTestPatter
                set(self.hFig, 'menubar', 'none', 'toolbar', 'none');
%                 set(self.hFig, 'HandleVisibility', 'callback');
            end

            self.hNMS = ...
                smalltargetmotiondetectors.tool.MatrixNMS( ...
                self.paraNMS.maxRegionSize, ...
                self.paraNMS.method);
            
            % 添加一个文本框来显示时间信息
            self.uiHandle.timeTextBox = uicontrol( ...
                'Parent', self.hFig, ...
                'Style', 'text', ...
                'String', 'Initiating, please wait...', ...
                'Position', [165, 350, 250, 50], ...
                'FontSize', 12, ... 
                'HorizontalAlignment', 'center');
            
            self.uiHandle.cancelButton = uicontrol( ...
                'Parent', self.hFig, ...
                'Style', 'pushbutton', ...
                'String', 'Cancel', ...
                'Position', [400, 10, 100, 30], ...
                'Callback', @cancelCallback);
            
            
            self.uiHandle.pauseButton = uicontrol( ...
                'Parent', self.hFig, ...
                'Style', 'pushbutton', ...
                'String', 'Pause', ...
                'Position', [300, 10, 100, 30], ...
                'Callback', @pauseCallback);
            
            % 设置取消和暂停标志
            setappdata(self.hFig, 'canceling', 0);
            setappdata(self.hFig, 'pausing', 0);

            self.hasFigHandle = true;
            self.timeTic = tic;


            %---------------------------------%
            function cancelCallback(hObject, ~) 
                setappdata(hObject.Parent, 'canceling', 1);
            end


            %---------------------------------%
            function pauseCallback(hObject, ~)
                pausing = getappdata(hObject.Parent, 'pausing');
                if pausing
                    setappdata(hObject.Parent, 'pausing', 0);
                    set(hObject, 'String', 'Pause');
                else
                    setappdata(hObject.Parent, 'pausing', 1);
                    set(hObject, 'String', 'Resume');
                end
            end

        end

        function show_result( ...
                self, ...
                colorImg, ...
                result ...
                )
            %show_result
            elapsedTime = toc(self.timeTic);
            set(self.uiHandle.timeTextBox, ...
                'String', ...
                sprintf('Elapsed Time: %.4f s/frame', elapsedTime));

            if getappdata(self.hFig, 'canceling')
                close(self.hFig);
                self.hasFigHandle = false;
                fprintf('\n \t --- Manual termination --- \t \n');
                return;
            end

            

            modelOpt = result.response;
            motionDirection = result.direction;
            
            figure(self.hFig);

            hold off;
            imshow(colorImg, 'Border', 'loose');


            axis manual;

            maxOutput = max(modelOpt(:));

            if maxOutput > 0 
                if abs(maxOutput - 1) > 1e-16  % normalization
                    normOutput = modelOpt/maxOutput;
                else
                    normOutput = modelOpt;
                end
                nmsOutput = self.hNMS.nms(normOutput);
                [idX, idY] = find(nmsOutput > self.showThreshold);
                
                hold on;
                plot(idY, idX, '*', ...
                    'MarkerEdgeColor', 'r', ...
                    'MarkerSize', 5 );

                
                if ~isempty(motionDirection) % Direction
                    

                    linearInd_Direction = ...
                        sub2ind(size(motionDirection), idX, idY);

                    indX = cos(motionDirection);
                    indY = sin(motionDirection);
                    cosD = indX(linearInd_Direction); 
                    sinD = indY(linearInd_Direction);
                    cosD( isnan(cosD) ) = 0;
                    sinD( isnan(sinD) ) = 0;

                    lenArrow = 20;
                    % In the figure of imshow, the positive direction of
                    %   the y axis is downward, that is 'axis IJ'.
                    quiver(idY, idX, ...
                        lenArrow * cosD, ...
                        -lenArrow * sinD, ...
                        0);
                end
            end

            drawnow;

            if self.isSaveAsVideo
                self.save_video();
            end
            
            if getappdata(self.hFig, 'pausing')
                while getappdata(self.hFig, 'pausing')
                    pause(0.1); % 等待0.1秒，检查暂停标志是否被设置为假

                    if getappdata(self.hFig, 'canceling')
                        close(self.hFig);
                        self.hasFigHandle = false;
                        fprintf('\n \t --- Manual termination --- \t \n');
                        return;
                    end
                end
            end
            self.timeTic = tic;
        end
        
        function save_video(self)
            if self.saveState == 0
                if isempty(self.savePath)
                    self.savePath = pwd;
                elseif ~isfolder(self.savePath)
                    mkdir(self.savePath)
                    if ~isfolder(self.savePath)
                        error([self.savePath,...
                            ' is not a folder and',...
                            ' cannot be created automatically.']);
                    end
                end
                if isempty(self.videoPath)
                    self.videoPath = 'visualization_video.avi';
                end
                fileName = fullfile(self.savePath,self.videoPath);
                self.hVideo = VideoWriter(fileName);
                self.hVideo.FrameRate = 10;
                
                fprintf('Visual output video is saved as ''%s''.\n',fileName);
                open(self.hVideo);
                self.saveState = true;
            end
            currFrame = getframe(self.hFig);
            writeVideo(self.hVideo,currFrame);
        end

        function delete(self)
            delete(self.hVideo);
            try %#ok<TRYNC> 
                self.hFig.delete();
            end

        end
        
    end % end methods
end

