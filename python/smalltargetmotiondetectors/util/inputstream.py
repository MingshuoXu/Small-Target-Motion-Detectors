import os
import sys
import cv2
from tkinter import filedialog
import glob
import numpy as np

class ImgstreamReader:
    def __init__(self, imgsteamFormat=None, startFrame=1, endFrame=None):
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

        self.hasFrame = True    # Flag indicating if there are frames available
        self.fileList = []      # List of files in the image stream
        self.currIdx = 0        # Index of the current frame being read
        self.frameIdx = 0       # Current frame index (visible)
        self.hWaitbar = None # Handle for waitbar UI
        self.endFrame = 0       # Index of the last frame
        self.isShowWaitbar = False      # Flag indicating whether to show waitbar
        self.hasDeleteWaitbar = False   # Flag indicating whether waitbar has been deleted


        # Create waitbar handle
        # self.create_waitbar_handle()

        # Initialize file list based on input arguments
        if imgsteamFormat is None:
            startImgName, endImgName = self.get_filelist_from_ui()
        else:
            startImgName, endImgName = self.get_filelist_from_imgsteamformat( \
                imgsteamFormat, startFrame, endFrame)

        # Find start and end frame indices
        shouldFoundStart = True
        for idx in range(len(self.fileList)):
            if shouldFoundStart:
                if os.path.basename(self.fileList[idx]) == os.path.basename(startImgName):
                    startIdx = idx
                    shouldFoundStart = False
            else:
                if os.path.basename(self.fileList[idx]) == os.path.basename(endImgName):
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


    def get_filelist_from_ui(self):
        '''
        get_filelist_from_ui - Prompts user to select start and end frames via GUI.
          This method prompts the user to select the start and end frames of the
          image stream using a graphical user interface (GUI). It opens a file dialog
          for the user to choose the start frame, and then another file dialog for
          the end frame, ensuring that both frames have the same file extension.
        
          Parameters:
              - self: Instance of the ImgstreamReader class.
        
          Returns:
              - startImgName: Name of the selected start frame file.
              - endImgName: Name of the selected end frame file.
        '''

        # Get the full path of this file
        filePath = os.path.abspath(__file__)
        # Find the index of '/smalltargetmotiondetectors/' in the file path
        indexPath = filePath.find(os.path.join(os.sep, 'smalltargetmotiondetectors'))
        # Add the path to the package containing the models
        sys.path.append(filePath[:indexPath + len('smalltargetmotiondetectors') + 1])
        

        pathName = os.path.join(filePath[:indexPath-7], 'demodata', 'imgstream')
        # Open file dialog for selecting the start frame
        startImgName = filedialog.askopenfilename(
            initialdir=pathName, 
            title='Pick the start frame from image stream')
        
        # Find the index of the dot in the start image name
        dotIndex = startImgName.rfind('.')
        
        # Open file dialog for selecting the end frame
        endImgName = filedialog.askopenfilename(
            initialdir=pathName, 
            title='Pick the end frame from image stream', 
            filetypes=[(startImgName[dotIndex:], startImgName[dotIndex:])])
        
        # Get the file extensions of the start and end images
        _, ext1 = os.path.splitext(startImgName)
        _, ext2 = os.path.splitext(endImgName)
        
        # Check if the extensions of the start and end images are the same
        if ext1 != ext2:
            raise Exception('Start image has a different extension than end image.')
        
        # Update fileList property with files matching the selected extension
        self.fileList = glob.glob(os.path.join(pathName, '*' + ext1))
        return startImgName, endImgName


    def get_filelist_from_imgsteamformat(self, imgsteamFormat, startFrame, endFrame):
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
        self.fileList = os.listdir(imgsteamFormat)
        
        # Extract the basename and extension from the specified format
        basename, ext1 = os.path.splitext(imgsteamFormat)
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
                startImgName = basename + '%0' + str(numDigits1) + 'd' + ext1 % startFrame
                endImgName = basename + '%0' + str(numDigits1) + 'd' + ext1 % endFrame
            else:
                # Generate the start and end frame names without zero-padding
                startImgName = basename + '%d' + ext1 % startFrame
                endImgName = basename + '%d' + ext1 % endFrame
                
        return startImgName, endImgName
    

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
              - colorImg: Color version of the retrieved frame.
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

        return np.double(garyImg), colorImg
    

    
    def __del__(self):
        '''
            Release any resources associated with the object
        '''
        if self.hWaitbar:
            # Perform cleanup actions, such as closing files or releasing memory
            # For demonstration, let's assume hWaitbar is a file object and close it
            self.hWaitbar.close()
            # Set hWaitbar to None to indicate that the resource has been released
            self.hWaitbar = None

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
        self.create_waitbar_handle()

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
        self.hasFrame = True
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

        return grayImg, colorImg


    def create_waitbar_handle(self):
        """
        Creates a waitbar to show progress.
        """

        # Implementation of create_waitbar_handle method goes here
        pass

    def call_waitbar(self):
        """
        Updates the waitbar with current progress.
        """

        # Implementation of call_waitbar method goes here
        pass

    def __del__(self):
        """
        Destructor method.
        """
        pass
        # self.hVid.release()


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


if __name__ == '__main__':
    pass