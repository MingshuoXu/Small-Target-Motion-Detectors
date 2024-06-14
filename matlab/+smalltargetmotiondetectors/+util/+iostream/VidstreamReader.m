classdef VidstreamReader < handle
    %VidstreamReader - Class for reading frames from a video file.
    %   This class provides methods to read frames from a video file and
    %   visualize the progress using a waitbar.
    %
    %   Properties:
    %       hasFrame - Indicates if there are more frames available.
    %       startFrame - Starting frame number.
    %       endFrame - Ending frame number.
    %
    %   Hidden Properties:
    %       hVid - VideoReader object for reading the video file.
    %       currIdx - Index of the current frame being processed.
    %       frameIdx - Index of the current frame in the video.
    %       hWaitbar - Handle to the waitbar.
    %       isShowWaitbar - Flag to indicate if waitbar is shown.
    %       hasDeleteWaitbar - Flag to indicate if waitbar has been deleted.
    %
    %   Methods:
    %       VidstreamReader - Constructor method.
    %       get_next_frame - Retrieves the next frame from the video.
    %       create_waitbar_handle - Creates a waitbar to show progress.
    %       call_waitbar - Updates the waitbar with current progress.
    %       delete - Destructor method.
    %
    %   Example:
    %       % Create VidstreamReader object and read frames
    %       vidReader = VidstreamReader('video.mp4');
    %       while vidReader.hasFrame
    %           [grayFrame, colorFrame] = vidReader.get_next_frame();
    %           % Process frames
    %       end
    %       % Delete the VidstreamReader object
    %       delete(vidReader);
    %
    
    properties
        hasFrame = true;
        startFrame;
        endFrame;
        
    end
    
    
    properties(Hidden)
        hVid;
        currIdx = 1;
        frameIdx; 
        hWaitbar;
        isShowWaitbar = false;
        hasDeleteWaitbar = false;
    end
    
    
    methods
        function self = VidstreamReader(vidName, startFrame, endFrame)
            %VidstreamReader - Constructor method.
            %   Creates an instance of the VidstreamReader class.
            %
            %   Syntax:
            %       vidReader = VidstreamReader(vidName, startFrame, endFrame)
            %
            %   Input Arguments:
            %       vidName - Name of the video file.
            %       startFrame - Starting frame number (optional, default is 1).
            %       endFrame - Ending frame number (optional, default is the last frame).
            %
            %   Output:
            %       vidReader - VidstreamReader object.
            
            self.create_waitbar_handle();

            if nargin == 0
                % Get the full path of this file
                filePath = mfilename('fullpath');
                %   Find the index of 'Small-Target-Motion-Detectors'
                % in the file path
                indexPath = strfind(filePath, ...
                    [filesep, '+smalltargetmotiondetectors', filesep]);
                % Add the path to the package containing the models
                addpath(filePath(1:indexPath));

                [fileName, pathName] = uigetfile(...
                    {'*.*'}, 'Selecting a input video', ...
                    [filePath(1:indexPath-7),'demodata', filesep, ...
                    'RIST_GX010290.mp4']);
                vidName =  fullfile(pathName, fileName);
            end
            
            self.hVid = VideoReader(vidName);
            
            if nargin < 2
                self.startFrame = 1;
            else
                self.startFrame = startFrame;
            end
            
            if nargin == 3
                if endFrame > self.hVid.NumFrames
                    self.endFrame = self.hVid.NumFrames;
                else
                    self.endFrame = endFrame;
                end
            else
                self.endFrame = self.hVid.NumFrames;
            end
        end
        
        function [garyImg, colorImg] = get_next_frame(self)
            %get_next_frame - Retrieves the next frame from the video.
            %   This method reads the next frame from the video file and
            %   returns both grayscale and color versions of the frame.
            %
            %   Output:
            %       grayImg - Grayscale version of the frame.
            %       colorImg - Color version of the frame.
            
            if self.hasFrame
                if self.currIdx == 1
                    colorImg = read(self.hVid, self.startFrame);
                    self.frameIdx = self.startFrame;
                else
                    colorImg =  readFrame(self.hVid);
                end
                garyImg = double(rgb2gray(colorImg)) / 255;
            else
                error('Could not get the frame');
            end
            
            if self.frameIdx < self.endFrame
                self.hasFrame = true;
                self.currIdx = self.currIdx + 1;
                self.frameIdx = self.frameIdx + 1;
            else
                self.hasFrame = false;
            end
            
            if self.isShowWaitbar
                self.call_waitbar();
            elseif ~self.hasDeleteWaitbar
                delete(self.hWaitbar);
                self.hasDeleteWaitbar = true;
            end
        end
        
        function create_waitbar_handle(self)
            %create_waitbar_handle - Creates a waitbar to show progress.
            self.hWaitbar = waitbar(...
                0, 'initiating, please wait...', ...
                'name', 'VidsteamReader', ...
                'Position', [450,450,270,50] ...
                );
        end
        
        function call_waitbar(self)
            %call_waitbar - Updates the waitbar with current progress.
            floatBar = self.frameIdx/self.endFrame;
            waitbarStr = sprintf('FrameIndex: %d', self.frameIdx);
            self.hWaitbar = waitbar(floatBar, self.hWaitbar, waitbarStr);
        end
        
        function delete(self)
            %delete - Destructor method.
            delete(self.hWaitbar);
        end
    end
end
