import os
import sys
import time

import cv2
import glob
import logging
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from .matrixnms import MatrixNMS
from .. import model


# Get the full path of this file
filePath = os.path.realpath(__file__)
gitCodePath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(filePath))))
VID_DEFAULT_FOLDER = os.path.join(gitCodePath, 'demodata')
IMG_DEFAULT_FOLDER = os.path.join(VID_DEFAULT_FOLDER, 'imgstream')
# Add the path to the package containing the models
ALL_MODEL = model.__all__


# configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()] )
logger = logging.getLogger(__name__)


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
        self.endFrame = endFrame       # Index of the last frame

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
        for idx in range(len(self.fileList)+2):
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
        else:
            self.hasFrame = True

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
        startFolder, startBaseN = os.path.split(self.startImgName)
        _, extNameSta = os.path.splitext(startBaseN) 
        endFolder, endBaseN = os.path.split(self.endImgName)
        _, extNameEnd = os.path.splitext(endBaseN) 

        if extNameSta != extNameEnd:
            raise Exception('Start image has a different extension than end image.')
        if os.path.basename(startFolder) != os.path.basename(endFolder):
            raise Exception('The image stream must be in the same folder!')
        
        # Update fileList property with files matching the selected extension
        self.fileList = glob.glob(os.path.join(startFolder, '*' + extNameSta))

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
        self.fileList = glob.glob(self.imgsteamFormat)
        
        # Extract the basename and extension from the specified format
        _dirName, _baseName = os.path.split(self.imgsteamFormat)
        basename, ext1 = os.path.splitext(_baseName)
        if len(basename) > 1 and basename[-1] == '*':
            basename = basename[:-1] # Remove the trailing *
        else:
            basename = ''
        
        # Check if any files match the specified format
        if not self.fileList:
            raise Exception('No files matching the format could be found.')
        else:
            # Determine the end frame index
            if not self.endFrame:
                self.endFrame = len(self.fileList)
            else:
                self.endFrame = min(self.endFrame, len(self.fileList))
            
            # Extract the names of the first and last files in the list
            nameFirst = os.path.splitext(self.fileList[0])[0]
            nameEnd = os.path.splitext(self.fileList[-1])[0]
            # Extract the numeric part from the end file name
            
            # Determine if the file names have the same length
            if len(nameFirst) == len(nameEnd):
                _, _name1 = os.path.split(nameFirst)
                numDigits1 = len(_name1.replace(basename, ''))
                
                # Generate the start and end frame names with zero-padding
                self.startImgName = os.path.join(_dirName,
                                                 "{}{:0{}}{}".format(basename, self.startFrame, numDigits1, ext1) )
                self.endImgName = os.path.join(_dirName,
                                               "{}{:0{}}{}".format(basename, self.endFrame, numDigits1, ext1) )
            else:
                # Generate the start and end frame names without zero-padding
                self.startImgName = os.path.join(_dirName,
                                                 "{}{}{}".format(basename, self.startFrame, ext1) )   
                self.endImgName = os.path.join(_dirName,
                                               "{}{}{}".format(basename, self.endFrame, ext1) )           
    
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
            raise Exception('Could not read the image!')

        # Check if the image was successfully read
        if colorImg is None:
            self.hasFrame = False
            raise Exception('Image is none!')

        # Convert the color image to grayscale
        grayImg = cv2.cvtColor(colorImg, cv2.COLOR_BGR2GRAY).astype(float) / 255

        # Update internal state to point to the next frame
        if self.currIdx < len(self.fileList)-1:
            self.hasFrame = True
            self.currIdx = self.currIdx + 1
            self.frameIdx = self.frameIdx + 1
        else:
            # If the end of the image stream is reached, set hasFrame to false
            self.hasFrame = False

        return np.double(grayImg), cv2.cvtColor(colorImg, cv2.COLOR_BGR2RGB)
    

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


    Methods:
        __init__ - Constructor method.
        get_next_frame - Retrieves the next frame from the video.
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

    def __init__(self, vidName=None, startFrame=0, endFrame=None):
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

        self.currIdx = 0
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
            if self.currIdx == 0:
                self.hVid.set(cv2.CAP_PROP_POS_FRAMES, self.startFrame)
                self.frameIdx = self.startFrame
            ret, colorImg = self.hVid.read()
            if not ret:
                raise Exception('Could not get the frame.')
                    
            grayImg = cv2.cvtColor(colorImg, cv2.COLOR_BGR2GRAY).astype(float) / 255
        else:
            raise Exception('Having reached the last frame.')

        if self.frameIdx < self.endFrame-1:
            self.hasFrame = True
        else:
            self.hasFrame = False
        self.currIdx  += 1
        self.frameIdx += 1

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

    Example:
        # Create an instance of the Visualization class
        vis = Visualization(className='ExampleClass', showThreshold=0.8)
        
        # Create a figure handle
        vis.create_fig_handle()
        
        # Show the result using the show_result method
        vis.show_result(colorImg, result, runTime)
        
        # Clean up resources
        del vis
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

    def __del__(self):
        if self.isSaveAsVideo:
            self.hVideo.release()
            print(f"Visual output video is saved as '{os.path.join(self.savePath, self.videoPath)}'.")
        try:
            plt.close(self.hFig)
        except:
            pass

    def create_fig_handle(self, width = 8, height = 5, dpi = 100):
        """
        Creates a figure handle.
        """
        self.hFig = plt.figure(f'Show result for {self.inputClassName}',
                               figsize=(width, height),
                               dpi=dpi)
        manager = self.hFig.canvas.manager
        manager.window.geometry("+{}+{}".format(
            int((manager.window.winfo_screenwidth() - manager.window.winfo_reqwidth()) / 8), 
            int((manager.window.winfo_screenheight() - manager.window.winfo_reqheight()) / 8) 
            ) )
        plt.axis('off')
        
        if not self.isTestPatter:
            self.hFig.canvas.toolbar.pack_forget()
            self.hFig.canvas.manager.toolbar.pack_forget()
        else:
            logger.info("Test pattern is used for visualization.")

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
        logger.info("Visualization started.")

    def show_result(self, colorImg=None, result={'response': None, 'direction': None}, runTime=None):
        """
        Display the result.
        
        Args:
            colorImg (numpy.ndarray): Color image.
            result (dict): Dictionary containing model output and motion direction.
        """
        
        # Return if the close button is pressed
        if self.uiHandle['closeButton'].bool:
            return
        
        # Calculate and display elapsed time
        elapsedTime = time.time() - self.timeTic if runTime is None else runTime
        self.uiHandle['timeTextBox'].config(text=f'Elapsed Time: {elapsedTime:.4f} s/frame')


        plt.figure(self.hFig.number)
        plt.clf()
        ax = plt.gca()

        # Display the color image if provided
        if colorImg is not None:
            plt.imshow(colorImg)
            plt.axis('off')
        
        modelOpt = result['response']
        motionDirection = result['direction']

        # Check if modelOpt is not None and has any elements
        if modelOpt is not None and np.any(modelOpt):
            # Call the appropriate method based on the type of modelOpt
            if isinstance(modelOpt, np.ndarray):
                self.show_matrix_output(ax, modelOpt, motionDirection)
            elif isinstance(modelOpt, list) and all(isinstance(i, list) and len(i) == 3 for i in modelOpt):
                self.show_dots_output(ax, modelOpt, motionDirection)
            elif isinstance(modelOpt, list) and all(isinstance(i, list) and len(i) == 5 for i in modelOpt):
                self.show_bboxs_output(ax, modelOpt, motionDirection)
            ''' In the figure of imshow, the positive direction of
                the y axis is downward.
                
                %--------------------------------------------------------%
                %   --------> J         --------> x       y        
                %   |                   |                 ^
                %   |                   |                 |
                %   V                   v                 |
                %   I                   y                 |--------> x
                %                                       
                %   Matrix              Figure           Plane-coordinate 
                %                                           system
                %---------------------------------------------------------%
            '''

        plt.draw()
        plt.pause(0.001)

        # Save as video if the flag is set
        if self.isSaveAsVideo:
            self.save_video(ax)

        # Pause if the pause button is pressed, and wait until it is released or the close button is pressed
        while self.uiHandle['pauseButton'].bool:
            plt.pause(0.1)
            if self.uiHandle['closeButton'].bool:
                return
            if 'steppingButton' in self.uiHandle: 
                if self.uiHandle['steppingButton'].bool:
                    self.uiHandle['steppingButton'].bool = False
                    break

        self.timeTic = time.time()

    def show_matrix_output(self, ax, modelOpt, motionDirection):
        """
        Display matrix output with optional motion direction.
        
        Args:
            ax (matplotlib.axes.Axes): The axes to plot on.
            modelOpt (numpy.ndarray): Model output matrix.
            motionDirection (numpy.ndarray): Motion direction matrix.
        """
        maxOutput = np.max(modelOpt)
        if maxOutput > 0:
            if self.shouldNMS:
                nmsOutput = self.hNMS.nms(modelOpt)
            else:
                nmsOutput = modelOpt
            idX, idY = np.where(nmsOutput > self.showThreshold * maxOutput)
            ax.scatter(idY, idX, s=5, c='r', marker = '*')

            if motionDirection is not None:
                if len(motionDirection) > 0:
                    nanStatus = ~np.isnan(motionDirection[idX, idY])
                    quiverX = idX[nanStatus]
                    quiverY = idY[nanStatus]
                    cosD = np.cos(motionDirection[quiverX, quiverY])
                    sinD = np.sin(motionDirection[quiverX, quiverY])
                    lenArrow = 8

                    # angles='uv', which has the same orientation with plane coordinates
                    ax.quiver(quiverY, quiverX,
                            lenArrow * cosD, lenArrow * sinD, 
                            scale_units='xy', 
                            scale=0.5, 
                            color='red',
                            width=0.003)

    def show_dots_output(self, ax, response, direction):
        """
        Display dots output with optional motion direction.
        
        Args:
            ax (matplotlib.axes.Axes): The axes to plot on.
            response (list): List of [x, y, cc] entries.
            direction (list): List of [x, y, dir] entries.
        """
        responseArray = np.array(response)

        maxOutput = np.max(responseArray[:, -1])
        filtered_rows = responseArray[responseArray[:, 2] > self.showThreshold * maxOutput]

        # Get x and y columns
        idY = filtered_rows[:, 0]
        idX = filtered_rows[:, 1]

        ax.scatter(idY, idX, s=5, c='r', marker = '*')

        if direction is not None and len(direction) > 0:
            directionArray = np.array(direction)

            # Filter out rows where the first column is in idX, the second column is in idY, and the third column is not None
            mask = (np.isin(directionArray[:, 0], idY) &
                    np.isin(directionArray[:, 1], idX) &
                    ~np.isnan(directionArray[:, 2]))
            directionArray = directionArray[mask]  
                
            if len(directionArray) > 0:
                quiverY, quiverX, dirTheta = directionArray[:, 0], directionArray[:, 1], directionArray[:, 2]
                
                cosD = np.cos(dirTheta)
                sinD = np.sin(dirTheta)
                lenArrow = 8

                ax.quiver(quiverY, quiverX,
                          lenArrow * cosD, lenArrow * sinD,
                          scale_units='xy',
                          scale=0.5,
                          color='red',
                          width=0.003)

    def show_bboxs_output(self, ax, bboxes, direction):
        """
        Display bounding boxes with optional motion direction.
        
        Args:
            ax (matplotlib.axes.Axes): The axes to plot on.
            bboxes (list): List of [x, y, w, h, cc] entries.
            direction (list): List of [x, y, dir] entries.
        """
        bboxesArray = np.array(bboxes)
        maxOutput = np.max(bboxesArray[:, -1])

        for bbox in bboxes:
            idX, idY, w, h, cc = bbox
            if cc > self.showThreshold * maxOutput:
                rect = plt.Rectangle((idX, idY), w, h, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

        if direction is not None and len(direction) > 0:
            directionArray = np.array(direction)

            # Filter out rows where the first column is in bboxesArray[:, 0], the second column is in bboxesArray[:, 1], and the third column is not None
            mask = (np.isin(directionArray[:, 0], bboxesArray[:, 0]) &
                    np.isin(directionArray[:, 1], bboxesArray[:, 1]) &
                    ~np.isnan(directionArray[:, 2]))
            directionArray = directionArray[mask]               

            if len(directionArray) > 0:
                quiverY, quiverX, dirTheta = directionArray[:, 0], directionArray[:, 1], directionArray[:, 2]
                
                cosD = np.cos(dirTheta)
                sinD = np.sin(dirTheta)
                lenArrow = 6

                ax.quiver(quiverY, quiverX,
                          lenArrow * cosD, lenArrow * sinD,
                          scale_units='xy',
                          scale=0.5,
                          color='red',
                          width=0.002)

    def save_video(self, ax):
        if not self.saveState:
            if not self.savePath:
                self.savePath = os.getcwd()
            elif not os.path.isdir(self.savePath):
                os.mkdir(self.savePath)
                if not os.path.isdir(self.savePath):
                    raise FileNotFoundError(f"{self.savePath} is not a folder and cannot be created automatically.")

            if not self.videoPath:
                self.videoPath = 'visualization_video.avi'

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.hFig.canvas.draw()  # 更新画布
            _width, _height = self.hFig.canvas.get_width_height()
            self.hVideo = cv2.VideoWriter(os.path.join(self.savePath, self.videoPath),
                                          fourcc, 30, (_width, _height))
            
            self.saveState = True
        
        # 提取图像帧
        self.hFig.canvas.draw()  # 更新画布
        _width, _height = self.hFig.canvas.get_width_height()
        image = np.frombuffer(self.hFig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape((_height, _width, 3))

        # 将 RGB 转换为 BGR (OpenCV 使用 BGR 颜色格式)
        imageBGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 写入视频文件
        self.hVideo.write(imageBGR)

    def _closeCallback(self, event=None):
        """
        Callback function for close button.
        """
        plt.close(self.hFig)
        self.hasFigHandle = False
        self.uiHandle['pauseButton'].bool = False
        self.uiHandle['closeButton'].bool = True
        logger.info("Manual termination of visualization.")

    def _pauseCallback(self, event=None):
        """
        Callback function for pause button.
        """
        if self.uiHandle['pauseButton']['text'] == 'Pause':
            self.uiHandle['pauseButton']['text'] = 'Resume'
            self.uiHandle['pauseButton'].bool = True
            # create stepping button
            self.uiHandle['steppingButton'] = tk.Button(master=self.hFig.canvas.get_tk_widget(), 
                                                 text='Stepping', 
                                                 command=self._steppingCallback)
            self.uiHandle['steppingButton'].place(relx=0.5, rely=0.85, width=80, height=30)
            self.uiHandle['steppingButton'].bool = False
        else:
            self.uiHandle['pauseButton']['text'] = 'Pause'
            self.uiHandle['pauseButton'].bool = False 
            # destroy stepping button
            if 'steppingButton' in self.uiHandle:
                self.uiHandle['steppingButton'].destroy()
                del self.uiHandle['steppingButton']

    def _steppingCallback(self, event=None):
        """
        Callback function for stepping button.
        """
        self.uiHandle['steppingButton'].bool = True


class ModelSelectorGUI:
    def __init__(self, root):
        self.root = root

    def create_gui(self, modelList):
        self.modelLabel = ttk.Label(self.root, text="Select a model:", width = 15)
        self.modelLabel.grid(row=0, column=0, padx=10, pady=10)

        self.modelCombobox = ttk.Combobox(self.root, values=modelList, width = 30)
        self.modelCombobox.current(11)
        self.modelCombobox.grid(row=0, column=1, columnspan=2, padx=10, pady=10)
        

class InputSelectorGUI:
    def __init__(self, root):
        self.root = root

        self.vidElement = {}
        self.imgElement = {}

        self.imgSelectFolder = None

        self.inputType = None
        self.startFrame = None
        self.endFrame = None
        self.video_file = None
        self.check = False
        self.vidName = None
        self.startFolder = None
        self.startImgName = None
        self.endFolder = None
        self.endImgName = None

    def create_gui(self):
        self.inputTypeLabel = ttk.Label(self.root, text="Select input from:", width = 15)
        self.inputTypeLabel.grid(row=1, column=0, padx=10, pady=10)

        self.selectedOption = tk.IntVar(value=0)


        self.vidLabel = ttk.Radiobutton(self.root, 
                                        text='Video stream', 
                                        variable=self.selectedOption,
                                        value=1, 
                                        command=self.select_vidstream)
        self.vidLabel.grid(row=1, column=2, padx=10, pady=10)
        
        self.imgLabel = ttk.Radiobutton(self.root, 
                                        text='Image stream', 
                                        variable=self.selectedOption,
                                        value=2, 
                                        command=self.select_imgstream)
        self.imgLabel.grid(row=1, column=1, padx=10, pady=10)

    def select_vidstream(self):
        self.imgSelectFolder = None
        self.startImgName = None
        self.endImgName = None
        for element in self.imgElement.values():
            element.destroy()

        self.vidElement['lblVidIndicate'] = ttk.Label(self.root, text= 'Video\'s path:',  width = 15)
        self.vidElement['lblVidIndicate'].grid(row=2, column=0, padx=10, pady=30)
        self.vidElement['lblVidPath'] = ttk.Label(self.root, 
                                           text="Waiting for the selection",
                                           wraplength=220
                                           )
        self.vidElement['lblVidPath'].grid(row=2, column=1, columnspan=2, padx=10, pady=10)
        
        self.vidElement['btn'] = ttk.Button(self.root, text="Select a video", command=self._clicked_vid)
        self.vidElement['btn'].grid(row=3, column=2, padx=10, pady=10)
        
    def _clicked_vid(self):
        self.vidName = filedialog.askopenfilenames(initialdir=VID_DEFAULT_FOLDER)
        self.vidName = self.vidName[0]
        self.vidElement['lblVidPath'].config(text=self.vidName)

    def select_imgstream(self):
        self.vidName = None
        for element in self.vidElement.values():
            element.destroy()

        self.imgElement['lblFolder'] = ttk.Label(self.root, text="Image's folder: ",  width = 15)
        self.imgElement['lblFolder'].grid(row=2, column=0, padx=10, pady=10)
        self.imgElement['lblFolderName'] = ttk.Label(self.root, text="Waiting for the selection", wraplength=220)
        self.imgElement['lblFolderName'].grid(row=2, column=1, columnspan=2, padx=10, pady=30)

        self.imgElement['btnStart'] = ttk.Button(self.root, text="Select start frame", command=self._clicked_start_img)
        self.imgElement['btnStart'].grid(row=3, column=1,  padx=10, pady=10)
        self.imgElement['btnEnd'] = ttk.Button(self.root, text="Select end frame", command=self._clicked_end_img)
        self.imgElement['btnEnd'].grid(row=4, column=1,  padx=10, pady=10)
        
    def _clicked_start_img(self):
        startImgFullPath = filedialog.askopenfilenames(
            initialdir=IMG_DEFAULT_FOLDER if self.imgSelectFolder is None else self.imgSelectFolder)
        self.startFolder, self.startImgName = os.path.split(startImgFullPath[0])
        if self.endFolder is not None:
            if os.path.basename(self.startFolder) == os.path.basename(self.endFolder):
                if self.endImgName is not None:
                    if check_same_ext_name(self.startImgName, self.endImgName):
                        self.check = True
                    else:
                        messagebox.showinfo("Message title", "Start image has a different extension than end image.")
            else:
                messagebox.showinfo("Message title", "The image stream must be in the same folder!")

        self.imgSelectFolder = self.startFolder
        self.imgElement['lblFolderName'].config(text=self.imgSelectFolder)

        self.imgElement['lblStartImg'] = ttk.Label(self.root, text=self.startImgName)
        self.imgElement['lblStartImg'].grid(row=3, column=2, padx=10, pady=10)

    def _clicked_end_img(self):
        endImgFullPath = filedialog.askopenfilenames(
            initialdir=IMG_DEFAULT_FOLDER if self.imgSelectFolder is None else self.imgSelectFolder)
        self.endFolder , self.endImgName = os.path.split(endImgFullPath[0])

        if self.startFolder is not None:
            if os.path.basename(self.endFolder) == os.path.basename(self.startFolder):
                if self.startImgName is not None:
                    if check_same_ext_name(self.startImgName, self.endImgName):
                        self.check = True
                    else:
                        messagebox.showinfo("Message title", "Start image has a different extension than end image.")
            else:
                messagebox.showinfo("Message title", "The image stream must be in the same folder!")

                
        self.imgSelectFolder = self.endFolder
        self.imgElement['lblFolderName'].config(text=self.imgSelectFolder)

        self.imgElement['lblEndImg'] = ttk.Label(self.root, text=self.endImgName)
        self.imgElement['lblEndImg'].grid(row=4, column=2, padx=10, pady=10)


class ModelAndInputSelectorGUI:
    def __init__(self, root):
        self.root = root

        windowHeight = 350
        windowWidth = 400
        
        startHeight = (root.winfo_screenheight() - windowHeight) // 2
        startWidth = (root.winfo_screenwidth() - windowWidth) // 2

        self.root.geometry('{}x{}+{}+{}'.format(windowWidth, windowHeight, startWidth, startHeight))
        self.root.title("Small target motion detector - Runner")
        self.root.iconbitmap(os.path.join(gitCodePath, 'stmd.ico'))
        
        self.objModelSelector = ModelSelectorGUI(root)
        self.objInputSelector = InputSelectorGUI(root)
        
        self.btnRun = ttk.Button(self.root, text="Run", command=self._run)
        self.btnRun.place(x = 20, y=300)
        self.btnStepping = ttk.Button(self.root, text="Stepping", command=self._stepping)
        self.btnStepping.place(x = 20, y=270)
        self.isStepping = False

    def create_gui(self):
        self.objModelSelector.create_gui(ALL_MODEL)
        self.objInputSelector.create_gui()

        self.root.mainloop()

        if self.objInputSelector.selectedOption.get() == 1:  
            return self.modelName, self.vidName, None, self.isStepping
        elif self.objInputSelector.selectedOption.get() == 2:
            return self.modelName, self.startImgName, self.endImgName, self.isStepping

    def _run(self):
        self.modelName = self.objModelSelector.modelCombobox.get()
        if self.modelName not in ALL_MODEL:
            messagebox.showinfo("Message title", "Please select a STMD-based model!")
            return

        if self.objInputSelector.selectedOption.get() == 1:
            if self.objInputSelector.vidName is not None:
                self.vidName = self.objInputSelector.vidName
                self.root.destroy()
            else:
                messagebox.showinfo("Message title", "Please select a video")
        elif self.objInputSelector.selectedOption.get() == 2:
            if self.objInputSelector.startImgName is None:
                messagebox.showinfo("Message title", "Please select start frame!")
                return
            
            if self.objInputSelector.endImgName is None:
                messagebox.showinfo("Message title", "Please select end frame!")
                return

            if self.objInputSelector.check:
                self.startImgName = os.path.join(self.objInputSelector.imgSelectFolder, 
                                        self.objInputSelector.startImgName)
                self.endImgName = os.path.join(self.objInputSelector.imgSelectFolder, 
                                        self.objInputSelector.endImgName)
                self.root.destroy()
            else:
                messagebox.showinfo("Message title", "The image stream must be in the same folder!")
        else:
            messagebox.showinfo("Message title", "Please select input")
    
    def _stepping(self):
        self.isStepping = True
        self._run()


def check_same_ext_name(startImgName, endImgName):
    _, ext1 = os.path.splitext(startImgName)
    _, ext2 = os.path.splitext(endImgName)
    # Check if the extensions of the start and end images are the same
    if ext1 != ext2:
        return False
    else:
        return True
    

