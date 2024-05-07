import os
import sys
import time
import math

import matplotlib.pyplot as plt
import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, simpledialog, messagebox
import glob

from .matrixnms import MatrixNMS
from .. import model

# Get the full path of this file
filePath = os.path.abspath(__file__)
# Find the index of '/smalltargetmotiondetectors/' in the file path
indexPath = filePath.find(os.path.join(os.sep, 'smalltargetmotiondetectors'))
VID_DEFAULT_FOLDER = os.path.join(filePath[:indexPath-7], 'demodata')
IMG_DEFAULT_FOLDER = os.path.join(VID_DEFAULT_FOLDER, 'imgstream')

ALL_MODEL = model.__all__

class ImgstreamReader:
    def __init__(self, 
                 imgsteamFormat=None, 
                 startFrame=1, 
                 endFrame=None, 
                 startImgName = None, 
                 endImgName = None):
        '''
        ImgstreamReader Constructor - Initializes the ImgstreamReader object.
          This constructor initializes the ImgstreamReader object. It takes optional
          arguments to specify the format of the image stream, the starting frame,
          and the ending frame.
        
          Parameters:
              - imgsteamFormat: Format of the image stream (optional).
              - startFrame: Starting frame index (optional).
              - endFrame: Ending frame index (optional).
        '''

        self.hasFrame = False    # Flag indicating if there are frames available
        self.fileList = []      # List of files in the image stream
        self.currIdx = 0        # Index of the current frame being read
        self.frameIdx = 0       # Current frame index (visible)
        self.hWaitbar = None    # Handle for waitbar UI
        self.isShowWaitbar = False      # Flag indicating whether to show waitbar
        self.hasDeleteWaitbar = False   # Flag indicating whether waitbar has been deleted
        self.imgsteamFormat = imgsteamFormat
        self.startFrame = startFrame
        self.endFrame = startFrame       # Index of the last frame
        # Create waitbar handle
        # self.create_waitbar_handle()

        # Initialize file list based on input arguments
        if startImgName and endImgName is not None:
            self.startImgName = startImgName
            self.endImgName = endImgName
            self.get_filelist_from_imgName()
        elif imgsteamFormat is not None:
            self.get_filelist_from_imgsteamformat()
        else:
            raise Exception('')
        
        self.get_idx()

    def get_idx(self):
        # Find start and end frame indices
        shouldFoundStart = True
        for idx in range(len(self.fileList)):
            if shouldFoundStart:
                if os.path.basename(self.fileList[idx]) == os.path.basename(self.startImgName):
                    startIdx = idx
                    shouldFoundStart = False
            else:
                if os.path.basename(self.fileList[idx]) == os.path.basename(self.endImgName):
                    endIdx = idx
                    break

        # Check if start and end frames are found
        if not 'startIdx' in locals():
            raise Exception('Cannot find the start frame.')
        if not 'endIdx' in locals():
            raise Exception('Cannot find the end frame.')

        # Set fileList to frames between startIdx and endIdx
        self.fileList = self.fileList[startIdx:endIdx]

        # Check if fileList is empty
        if len(self.fileList) == 0:
            self.hasFrame = False

        # Set frameIdx to startIdx
        self.frameIdx = startIdx

        # Set endFrame to endIdx
        self.endFrame = endIdx

    def get_filelist_from_imgName(self):
        '''
          get_filelist_from_imgName
        
          Parameters:
              - self: Instance of the ImgstreamReader class.
        
        '''       
        # Find the index of the dot in the start image name
        dotIndex = self.startImgName.rfind('.')
        startFolder, _ = os.path.split(self.startImgName)
        endFolder, _ = os.path.split(self.endImgName)

        if not check_same_ext_name(self.startImgName, self.endImgName):
            raise Exception('Start image has a different extension than end image.')
        if os.path.basename(startFolder) != os.path.basename(endFolder):
            raise Exception('The image stream must be in the same folder!')
        
        # Update fileList property with files matching the selected extension
        self.fileList = glob.glob(os.path.join(startFolder, '*' + self.startImgName[dotIndex:]))

    def get_filelist_from_imgsteamformat(self):
        '''
        Get start and end frame names from image stream format
        This method generates the start and end frame names based on the specified image stream format
        
        Parameters:
        - self: Instance of the ImgstreamReader class.
        - imgsteamFormat: Format string specifying the image stream format.
        - startFrame: Index of the start frame.
        - endFrame: Index of the end frame.
        
        Returns:
        - startImgName: Name of the start frame file.
        - endImgName: Name of the end frame file.
        '''

        # Retrieve the list of files matching the image stream format
        self.fileList = os.listdir(self.imgsteamFormat)
        
        # Extract the basename and extension from the specified format
        basename, ext1 = os.path.splitext(self.imgsteamFormat)
        basename = basename[:-1] # Remove the trailing backslash if any
        
        # Check if any files match the specified format
        if not self.fileList:
            raise Exception('No files matching the format could be found.')
        else:
            # Determine the end frame index
            if not endFrame:
                endFrame = len(self.fileList)
            else:
                endFrame = min(endFrame, len(self.fileList))
            
            # Extract the names of the first and last files in the list
            name1 = os.path.splitext(self.fileList[0])[0]
            name0 = os.path.splitext(self.fileList[-1])[0]
            
            # Determine if the file names have the same length
            if len(name1) == len(name0):
                # Extract the numeric part from the end file name
                num1 = self.fileList[-1].replace(basename, '').replace(ext1, '')
                numDigits1 = len(num1)
                
                # Generate the start and end frame names with zero-padding
                self.startImgName = basename + '%0' + str(numDigits1) + 'd' + ext1 % self.startFrame
                self.endImgName = basename + '%0' + str(numDigits1) + 'd' + ext1 % self.endFrame
            else:
                # Generate the start and end frame names without zero-padding
                self.startImgName = basename + '%d' + ext1 % self.startFrame
                self.endImgName = basename + '%d' + ext1 % self.endFrame
    
    def get_next_frame(self):
        '''
        get_next_frame - Retrieves the next frame from the image stream.
          This method retrieves the next frame from the image stream and returns
          both grayscale and color versions of the frame. It updates the internal
          state to point to the next frame in the stream.
        
          Parameters:
              - self: Instance of the ImgstreamReader class.
        
          Returns:
              - garyImg: Grayscale version of the retrieved frame.
              - colorImg: Color version (RGB) of the retrieved frame.
        '''

        # Get information about the current frame
        fileInfo = self.fileList[self.currIdx]

        # Try to read the image file
        try:
            colorImg = cv2.imread(fileInfo)
            self.hasFrame = True
        except:
            # If an error occurs while reading the image, set hasFrame to false
            self.hasFrame = False
            raise Exception('Could not read the image.')

        # Convert the color image to grayscale
        garyImg = cv2.cvtColor(colorImg, cv2.COLOR_BGR2GRAY)

        # Update internal state to point to the next frame
        if self.currIdx < len(self.fileList)-1:
            self.hasFrame = True
            self.currIdx = self.currIdx + 1
            self.frameIdx = self.frameIdx + 1
        else:
            # If the end of the image stream is reached, set hasFrame to false
            self.hasFrame = False

        # Update the waitbar if necessary
        # if self.isShowWaitbar:
        #     self.call_waitbar()
        # elif not self.hasDeleteWaitbar:
        #     self.hWaitbar.destroy()
        #     self.hasDeleteWaitbar = True

        return np.double(garyImg), cv2.cvtColor(colorImg, cv2.COLOR_BGR2RGB)
    

class VidstreamReader:
    """
    VidstreamReader - Class for reading frames from a video file.
    This class provides methods to read frames from a video file and visualize
    the progress using a waitbar.

    Properties:
        hasFrame - Indicates if there are more frames available.

    Hidden Properties:
        hVid - VideoCapture object for reading the video file.
        startFrame - Starting frame number.
        currIdx - Index of the current frame being processed.
        frameIdx - Index of the current frame in the video.
        hWaitbar - Handle to the waitbar.
        endFrame - Ending frame number.
        isShowWaitbar - Flag to indicate if waitbar is shown.
        hasDeleteWaitbar - Flag to indicate if waitbar has been deleted.

    Methods:
        __init__ - Constructor method.
        get_next_frame - Retrieves the next frame from the video.
        create_waitbar_handle - Creates a waitbar to show progress.
        call_waitbar - Updates the waitbar with current progress.
        __del__ - Destructor method.

    Example:
        # Create VidstreamReader object and read frames
        vidReader = VidstreamReader('video.mp4')
        while vidReader.hasFrame:
            grayFrame, colorFrame = vidReader.get_next_frame()
            # Process frames
        # Delete the VidstreamReader object
        del vidReader
    """

    def __init__(self, vidName=None, startFrame=1, endFrame=None):
        """
        Constructor method for VidstreamReader class.
        
        Args:
            vidName (str): Name of the video file.
            startFrame (int, optional): Starting frame number. Defaults to 1.
            endFrame (int, optional): Ending frame number. Defaults to None, which indicates the last frame of the video.
            
        Returns:
            VidstreamReader: Instance of the VidstreamReader class.
        """
        # self.create_waitbar_handle()

        if vidName is None:
            # Get the full path of this file
            filePath = os.path.abspath(__file__)
            # Find the index of '/smalltargetmotiondetectors/' in the file path
            indexPath = filePath.find(os.path.join(os.sep, 'smalltargetmotiondetectors'))
            # Add the path to the package containing the models
            sys.path.append(filePath[:indexPath + len('smalltargetmotiondetectors') + 1])

            vidName = filedialog.askopenfilename(
                initialdir=os.path.join(filePath[:indexPath - 6], 'demodata', 'RIST_GX010290.mp4'),
                title='Selecting a input video'
                )

        self.hVid = cv2.VideoCapture(vidName)

        self.currIdx = 1
        self.hasFrame = self.hVid.isOpened()
        self.startFrame = startFrame

        if endFrame is None:
            self.endFrame = int(self.hVid.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            if endFrame > self.hVid.get(cv2.CAP_PROP_FRAME_COUNT):
                self.endFrame = int(self.hVid.get(cv2.CAP_PROP_FRAME_COUNT))
            else:
                self.endFrame = endFrame

    def get_next_frame(self):
        """
        Retrieves the next frame from the video.
        
        Returns:
            tuple: A tuple containing the grayscale and color versions of the frame.
        Raises:
            Exception: If the frame cannot be retrieved.
        """

        if self.hasFrame:
            if self.currIdx == 1:
                self.hVid.set(cv2.CAP_PROP_POS_FRAMES, self.startFrame)
                ret, colorImg = self.hVid.read()
                if not ret:
                    raise Exception('Could not get the frame')
                self.frameIdx = self.startFrame
            else:
                ret, colorImg = self.hVid.read()
                if not ret:
                    raise Exception('Could not get the frame')
                    
            grayImg = cv2.cvtColor(colorImg, cv2.COLOR_BGR2GRAY).astype(float)
        else:
            raise Exception('Could not get the frame')

        if self.frameIdx < self.endFrame:
            self.hasFrame = True
            self.currIdx += 1
            self.frameIdx += 1
        else:
            self.hasFrame = False

        # if self.isShowWaitbar:
        #     self.call_waitbar()
        # elif not self.hasDeleteWaitbar:
        #     # Delete the waitbar if it has not been deleted
        #     del self.hWaitbar
        #     self.hasDeleteWaitbar = True

        return grayImg, cv2.cvtColor(colorImg, cv2.COLOR_BGR2RGB)

    def __del__(self):
        """
        Destructor method.
        """
        self.hVid.release()


class Visualization:
    """
    A class to visualize the model output.
    
    Attributes:
        hFig: Figure handle.
        showThreshold: Threshold for visualization.
        isSaveAsVideo: Flag indicating whether to save visualization as a video.
        savePath: Path to save the visualization.
        videoPath: Path to the video file.
        inputClassName: Name of the input class.
        isTestPatter: Flag indicating whether it is a test pattern.
        paraNMS: Parameters for non-maximum suppression.
        shouldNMS: Flag indicating whether to perform non-maximum suppression.
        saveState: Flag indicating the saving state.
        hVideo: Video handle.
        timeTic: Time tic.
        hNMS: NMS handle.
        hasFigHandle: Flag indicating whether it has figure handle.
        uiHandle: UI handle.
        
    Methods:
        __init__: Constructor method.
    """

    def __init__(self, className='None', showThreshold=0.8):
        """
        Constructor method for Visualization class.
        
        Args:
            className (str, optional): Name of the class. Defaults to 'None'.
            showThreshold (float, optional): Threshold for visualization. Defaults to 0.8.
        """
        self.hFig = None
        self.showThreshold = showThreshold
        self.isSaveAsVideo = False
        self.savePath = None
        self.videoPath = None
        self.inputClassName = className
        self.isTestPatter = False
        self.paraNMS = {
            'maxRegionSize': 15,
            'method': 'sort'
        }
        self.shouldNMS = True
        self.saveState = False
        self.hVideo = None
        self.timeTic = None
        self.hNMS = None
        self.hasFigHandle = False
        self.uiHandle = None

        self.inputClassName = className
        self.showThreshold = showThreshold

    def __del__(self):
        if self.isSaveAsVideo:
            self.hVideo.release()

        try:
            plt.close(self.hFig)
        except:
            pass

    def create_fig_handle(self):
        """
        Creates a figure handle.
        """
        self.hFig = plt.figure(f'Show result for {self.inputClassName}')
        plt.axis('off')
        
        if not self.isTestPatter:
            self.hFig.canvas.toolbar.pack_forget()
            self.hFig.canvas.manager.toolbar.pack_forget()

        if self.shouldNMS:
            self.hNMS = MatrixNMS(self.paraNMS['maxRegionSize'], 
                                  self.paraNMS['method'])

        # create UI handle
        self.uiHandle = {}
        self.uiHandle['timeTextBox'] = tk.Label(master=self.hFig.canvas.get_tk_widget(), 
                                                text='Initiating, please wait...')
        self.uiHandle['timeTextBox'].place(relx=0.5, rely=0.05, anchor='center')

        self.uiHandle['closeButton'] = tk.Button(master=self.hFig.canvas.get_tk_widget(), 
                                                  text='Close', 
                                                  command=self._closeCallback)
        self.uiHandle['closeButton'].place(relx=0.8, rely=0.85, width=80, height=30)
        self.uiHandle['closeButton'].bool = False


        self.uiHandle['pauseButton'] = tk.Button(master=self.hFig.canvas.get_tk_widget(), 
                                                 text='Pause', 
                                                 command=self._pauseCallback)
        self.uiHandle['pauseButton'].place(relx=0.65, rely=0.85, width=80, height=30)
        self.uiHandle['pauseButton'].bool = False

        self.hasFigHandle = True
        self.timeTic = time.time()

    def show_result(self, colorImg=None, result=None):
        """
        Display the result.
        
        Args:
            colorImg (numpy.ndarray): Color image.
            result (dict): Dictionary containing model output and motion direction.
        """
        elapsedTime = time.time() - self.timeTic
        self.uiHandle['timeTextBox'].config(text=f'Elapsed Time: {elapsedTime:.4f} s/frame')

        if self.uiHandle['closeButton'].bool:
            return

        plt.figure(self.hFig.number)
        plt.clf()
        ax = plt.gca()

        if colorImg is not None:
            plt.imshow(colorImg)
            plt.axis('off')
        
        modelOpt = result['response']
        motionDirection = result['direction']
        if modelOpt is not None:
            maxOutput = np.max(modelOpt)

            if maxOutput > 0:
                if self.shouldNMS:
                    nmsOutput = self.hNMS.nms(modelOpt)
                idX, idY = np.where(nmsOutput > self.showThreshold * maxOutput)
                ax.plot(idY, idX, '*', markersize=5, markeredgecolor='r')

                if len(motionDirection):
                    nanStatus = ~np.isnan(motionDirection[idX, idY])
                    quiverX = idX[nanStatus]
                    quiverY = idY[nanStatus]
                    cosD = np.cos(motionDirection[quiverX, quiverY])
                    sinD = np.sin(motionDirection[quiverX, quiverY])
                    lenArrow = 8

                    '''
                    In the figure of imshow, the positive direction of
                      the x axis is downward, that is 'axis IJ'.
                    
                    %---------------------------------------%
                    %   --------> x         y               %
                    %   |                   ^               %
                    %   |                   |               %
                    %   V                   |               %
                    %   y                   --------> x     %
                    %                                       %
                    %   IJ # angles='xy'    image           %
                    %---------------------------------------%
                    ax.quiver(quiverY, quiverX, 
                            lenArrow * cosD, -lenArrow * sinD, 
                            angles='xy', 
                            scale_units='xy', 
                            scale=0.5, 
                            color='red',
                            width=0.003)
                    the above code is same with Matlab, which also works inpython, 
                        but in Python, we can use the below code:
                    '''

                    # angles='uv', which has the same orientation with plane coordinates
                    ax.quiver(quiverY, quiverX,
                            lenArrow * cosD, lenArrow * sinD, 
                            scale_units='xy', 
                            scale=0.5, 
                            color='red',
                            width=0.003)

        plt.draw()
        plt.pause(0.001)

        if self.isSaveAsVideo:
            self.save_video()

        while self.uiHandle['pauseButton'].bool:
            plt.pause(0.1)
            if self.uiHandle['closeButton'].bool:
                return

        self.timeTic = time.time()

    def save_video(self):
        if self.saveState == 0:
            if not self.savePath:
                self.savePath = os.getcwd()
            elif not os.path.isdir(self.savePath):
                os.mkdir(self.savePath)
                if not os.path.isdir(self.savePath):
                    raise FileNotFoundError(f"{self.savePath} is not a folder and cannot be created automatically.")

            if not self.videoPath:
                self.videoPath = 'visualization_video.avi'
            
            fileName = os.path.join(self.savePath, self.videoPath)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.hVideo = cv2.VideoWriter(fileName, fourcc, 10, (640, 480))
            
            print(f"Visual output video is saved as '{fileName}'.")
            self.saveState = True
        
        _, currFrame = self.hFig.canvas.retrieve()
        self.hVideo.write(currFrame)

    def _closeCallback(self, event=None):
        """
        Callback function for close button.
        """
        plt.close(self.hFig)
        self.hasFigHandle = False
        self.uiHandle['pauseButton'].bool = False
        print('\n \t --- Manual termination --- \t \n')

    def _pauseCallback(self, event=None):
        """
        Callback function for pause button.
        """
        if self.uiHandle['pauseButton']['text'] == 'Pause':
            self.uiHandle['pauseButton']['text'] = 'Resume'
            self.uiHandle['pauseButton'].bool = True
        else:
            self.uiHandle['pauseButton']['text'] = 'Pause'
            self.uiHandle['pauseButton'].bool = False   
    

class ModelSelectorGUI:
    def __init__(self, root):
        self.root = root

    def create_gui(self, modelList):
        self.modelLabel = ttk.Label(self.root, text="Select a model:")
        self.modelLabel.grid(row=0, column=0, padx=10, pady=10)

        self.modelCombobox = ttk.Combobox(self.root, values=modelList)
        self.modelCombobox.current(0)
        self.modelCombobox.grid(row=0, column=1, padx=10, pady=10)
        

class InputSelectorGUI:
    def __init__(self, root):
        self.root = root

        self.vidElement = {}
        self.imgElement = {}

        self.imgSelectFolder = None

        self.input_type = None
        self.start_frame = None
        self.end_frame = None
        self.video_file = None
        self.check = False
        self.vidName = None
        self.startImgName = None
        self.endImgName = None

    def create_gui(self):
        self.inputTypeLabel = ttk.Label(self.root, text="Select input type:")
        self.inputTypeLabel.grid(row=1, column=0, padx=10, pady=10)

        self.selectedOption = tk.BooleanVar(value=True)

        self.imgLabel = ttk.Radiobutton(self.root, 
                                        text='Image stream', 
                                        variable=self.selectedOption,
                                        value=True, 
                                        command=self.select_imgstream)
        self.imgLabel.grid(row=2, column=0, padx=10, pady=10)

        self.vidLabel = ttk.Radiobutton(self.root, 
                                        text='Video stream', 
                                        variable=self.selectedOption,
                                        value=False, 
                                        command=self.select_vidstream)
        self.vidLabel.grid(row=2, column=1, padx=10, pady=10)

    def select_vidstream(self):
        for element in self.imgElement.values():
            element.destroy()
        self.vidElement['btn'] = ttk.Button(self.root, text="Select a video", command=self._clicked_vid)
        self.vidElement['btn'].grid(row=3, column=0, padx=10, pady=10)

    def _clicked_vid(self):
        self.vidName = filedialog.askopenfilenames(initialdir=self.vidDefaultFolder)
        self.vidName = self.vidName[0]
        self.vidElement['lbl'] = ttk.Label(self.root, text=self.vidName, wraplength=150)
        self.vidElement['lbl'].grid(row=3, column=1, padx=50, pady=10)

    def select_imgstream(self):
        for element in self.vidElement.values():
            element.destroy()
        self.imgElement['btnStart'] = ttk.Button(self.root, text="select start frame", command=self._clicked_start_img)
        self.imgElement['btnStart'].grid(row=4, column=0,  padx=10, pady=10)
        self.imgElement['btnEnd'] = ttk.Button(self.root, text="select end frame", command=self._clicked_end_img)
        self.imgElement['btnEnd'].grid(row=5, column=0,  padx=10, pady=10)
        self.imgElement['lblFolder'] = ttk.Label(self.root, text="Image floder: ")
        self.imgElement['lblFolder'].grid(row=3, column=0, padx=10, pady=10)
        self.imgElement['lblFloderName'] = ttk.Label(self.root, text="", wraplength=150)
        self.imgElement['lblFloderName'].grid(row=3, column=1, padx=10, pady=30)

    def _clicked_start_img(self):
        startImgFullPath = filedialog.askopenfilenames(
            initialdir=IMG_DEFAULT_FOLDER if self.imgSelectFolder is None else self.imgSelectFolder)
        startFolder, self.startImgName = os.path.split(startImgFullPath[0])
        if self.imgSelectFolder is not None:
            if os.path.basename(startFolder) == os.path.basename(self.imgSelectFolder):
                if self.endImgName is not None:
                    if check_same_ext_name(self.startImgName, self.endImgName):
                        self.check = True
                    else:
                        messagebox.showinfo("Message title", "Start image has a different extension than end image.")
            else:
                messagebox.showinfo("Message title", "The image stream must be in the same folder!")

        self.imgSelectFolder = startFolder
        self.imgElement['lblFloderName'].config(text=self.imgSelectFolder)

        self.imgElement['lblStartImg'] = ttk.Label(self.root, text=self.startImgName)
        self.imgElement['lblStartImg'].grid(row=4, column=1, padx=10, pady=10)

    def _clicked_end_img(self):
        endImgFullPath = filedialog.askopenfilenames(
            initialdir=self.imgDefaultFolder if self.imgSelectFolder is None else self.imgSelectFolder)
        endFolder , self.endImgName = os.path.split(endImgFullPath[0])

        if self.imgSelectFolder is not None:
            if os.path.basename(endFolder) == os.path.basename(self.imgSelectFolder):
                if self.startImgName is not None:
                    if check_same_ext_name(self.startImgName, self.endImgName):
                        self.check = True
                    else:
                        messagebox.showinfo("Message title", "Start image has a different extension than end image.")
            else:
                messagebox.showinfo("Message title", "The image stream must be in the same folder!")

                
        self.imgSelectFolder = endFolder
        self.imgElement['lblFloderName'].config(text=self.imgSelectFolder)

        self.imgElement['lblEndImg'] = ttk.Label(self.root, text=self.endImgName)
        self.imgElement['lblEndImg'].grid(row=5, column=1, padx=10, pady=10)


class ModelAndInputSelectorGUI:
    def __init__(self, root):
        self.root = root

        self.root.title("Small target motion detector - Runner")

        self.objModelSelector = ModelSelectorGUI(root)
        self.objInputSelector = InputSelectorGUI(root)
        
        self.btnRun = ttk.Button(self.root, text="run", command=self._run)
        self.btnRun.grid(row=6, column=1, padx=10, pady=10)

    def create_gui(self):
        self.objModelSelector.create_gui(ALL_MODEL)
        self.objInputSelector.create_gui()

        self.root.mainloop()

        if self.objInputSelector.selectedOption.get():  
            return self.modelName, self.startImgName, self.endImgName
        else:
            return self.modelName, self.vidName, False

    def _run(self):

        self.modelName = self.objModelSelector.modelCombobox.get()
        if self.modelName not in ALL_MODEL:
            messagebox.showinfo("Message title", "Please select a STMD-based model!")
            return

        if self.objInputSelector.selectedOption.get():
            if self.objInputSelector.check:
                self.startImgName = os.path.join(self.objInputSelector.imgSelectFolder, 
                                        self.objInputSelector.startImgName)
                self.endImgName = os.path.join(self.objInputSelector.imgSelectFolder, 
                                        self.objInputSelector.endImgName)
                self.root.destroy()
            else:
                messagebox.showinfo("Message title", "The image stream must be in the same folder!")
        elif self.objInputSelector.vidName is not None:
            self.vidName = self.objInputSelector.vidName
            messagebox.showinfo("Message title", "Please select a input video!")
            self.root.destroy()
        
        

def create_waitbar_handle(self):
    '''
    create_waitbar_handle - Creates a waitbar for displaying progress.
        This method creates a waitbar with an initial progress of 0% and a
        specific message, typically used to indicate the initialization phase
        of the ImgstreamReader.
    
        Parameters:
            - self: Instance of the ImgstreamReader class.
    '''

    # Create a waitbar with specific properties
    pass
    # self.hWaitbar = waitbar(
    #     0, 'Initiating, please wait...', name='ImgstreamReader', position=[450, 450, 270, 50])


def call_waitbar(self):
    '''
    call_waitbar - Updates the waitbar with current frame index.
        This method updates the progress of the waitbar based on the current
        frame index relative to the total number of frames and displays a
        message indicating the current frame index.
    
        Parameters:
            - self: Instance of the ImgstreamReader class.
    '''

    # Calculate the progress as a floating-point number
    floatBar = self.frameIdx / self.endFrame
    
    # Generate a message indicating the current frame index
    waitbarStr = 'FrameIndex: {}'.format(self.frameIdx)
    
    # Update the waitbar with the current progress and message

    pass
    # self.hWaitbar = waitbar(floatBar, self.hWaitbar, waitbarStr)


def check_same_ext_name(startImgName, endImgName):
    _, ext1 = os.path.splitext(startImgName)
    _, ext2 = os.path.splitext(endImgName)
    # Check if the extensions of the start and end images are the same
    if ext1 != ext2:
        return False
    else:
        return True
    

