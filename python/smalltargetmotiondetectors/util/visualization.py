import os
import time

import matplotlib.pyplot as plt
import numpy as np
import cv2

from .nms_for_matrix import MatrixNMS

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

    def create_fig_handle(self):
        """
        Creates a figure handle.
        """
        self.hFig = plt.figure()
        plt.title(f'Show result for {self.inputClassName}')
        
        if not self.isTestPatter:
            plt.rcParams['toolbar'] = 'None'

        self.hNMS = MatrixNMS(
            self.paraNMS['maxRegionSize'], self.paraNMS['method'])


        # self.uiHandle['timeTextBox'] = plt.text(
        #     0.5, 0.5, 'Initiating, please wait...',
        #     fontsize=12, ha='center', transform=self.hFig.transFigure)

        # self.uiHandle['cancelButton'] = plt.Button(
        #     plt.axes([0.8, 0.85, 0.1, 0.05]), 'Cancel')
        # self.uiHandle['cancelButton'].on_clicked(self.cancelCallback)

        # self.uiHandle['pauseButton'] = plt.Button(
        #     plt.axes([0.65, 0.85, 0.1, 0.05]), 'Pause')
        # self.uiHandle['pauseButton'].on_clicked(self.pauseCallback)

        self.hasFigHandle = True
        self.timeTic = time.time()

    def cancelCallback(self, event):
        """
        Callback function for cancel button.
        """
        event.canvas.get_tk_widget().quit()

    def pauseCallback(self, event):
        """
        Callback function for pause button.
        """
        if self.uiHandle['pauseButton'].label.get_text() == 'Pause':
            self.uiHandle['pauseButton'].label.set_text('Resume')
        else:
            self.uiHandle['pauseButton'].label.set_text('Pause')

    def show_result(self, colorImg, result):
        """
        Display the result.
        
        Args:
            colorImg (numpy.ndarray): Color image.
            result (dict): Dictionary containing model output and motion direction.
        """
        elapsedTime = time.time() - self.timeTic
        # self.uiHandle['timeTextBox'].set_text(f'Elapsed Time: {elapsedTime:.4f} s/frame')

        # if self.uiHandle['cancelButton'].clicked():
        #     plt.close(self.hFig)
        #     self.hasFigHandle = False
        #     print('\n \t --- Manual termination --- \t \n')
        #     return

        modelOpt = result['response']
        motionDirection = result['direction']

        plt.figure(self.hFig.number)
        plt.clf()
        plt.imshow(colorImg)
        ax = plt.gca()

        maxOutput = np.max(modelOpt)

        if maxOutput > 0:
            if self.shouldNMS:
                nmsOutput = self.hNMS.nms(modelOpt)
            idX, idY = np.where(nmsOutput > self.showThreshold * maxOutput)
            ax.plot(idY, idX, '*', markersize=5, markeredgecolor='r')

            if motionDirection is not None:
                nanStatus = ~np.isnan(motionDirection[idX, idY])
                quiverX = idX[nanStatus]
                quiverY = idY[nanStatus]
                cosD = np.cos(motionDirection[quiverX, quiverY])
                sinD = np.sin(motionDirection[quiverX, quiverY])
                lenArrow = 20
                ax.quiver(quiverY, quiverX, lenArrow * cosD, -lenArrow * sinD, angles='xy', scale_units='xy', scale=1, color='b')

        plt.draw()
        plt.pause(0.001)

        if self.isSaveAsVideo:
            self.save_video()

        # if self.uiHandle['pauseButton'].clicked():
        #     while self.uiHandle['pauseButton'].clicked():
        #         plt.pause(0.1)
        #         if self.uiHandle['cancelButton'].clicked():
        #             plt.close(self.hFig)
        #             self.hasFigHandle = False
        #             print('\n \t --- Manual termination --- \t \n')
        #             return

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

    def __del__(self):
        if self.isSaveAsVideo:
            self.hVideo.release()

        try:
            plt.close(self.hFig)
        except:
            pass
