Small Target Motion Detectors (STMD)
STMD is a MATLAB package for detecting small target motion in images or videos. It provides various functionalities and tools that enable users to effectively detect and track small targets in images or videos.

How to Use
——Installation
Clone or download this repository.

Add the repository's folders to MATLAB's search path.

——Examples
STMD provides two DEMO files demonstrating how to perform small target 
motion detection using the package:

demo_imgstream.m: This DEMO file demonstrates how to read images from an 
image stream and perform target detection using the STMD model.

demo_vidstream.m: This DEMO file demonstrates how to read videos from a 
video stream and perform target detection using the STMD model.

——Steps
Instantiate the STMD model.

Create an image or video stream reader.

Initialize the model.

In a loop, read each frame of the image or video and perform inference.

Display the inference results.

Parameter Adjustments
You can choose different input sources as needed, including demo images, 
real-world images, and simulated images.

You can adjust the parameters of the model, such as the maximum region 
size, input resolution, etc.

——Notes
Before running the DEMO files, make sure you have MATLAB installed and have
 set up MATLAB's environment correctly.

Ensure that you have added the repository to MATLAB's search path as per 
the installation instructions.

You are free to modify and extend the DEMO files according to your needs 
and specific application requirements and scenarios.



——Package Structure
+api: Contains API functions and classes for interacting with the STMD 
package.
+core: Contains core algorithms and utilities for motion detection and 
analysis.
+model: Contains models and neural networks used in small target motion 
detection.
	|  ESTMD           	(2008, S.D. Wiederman, PLoS ONE)
 	|  DSTMD           	(2020, H. Wang, IEEE T--Cybernetics)
 	|  STMDPlus        	(2020, H. Wang, IEEE T-NNLS)
 	|  FeedbackSTMD    	(2021, H. Wang, IEEE T-NNLS)
 	|  FSTMD           	(2021, Ling J, Front. Neurorobot)
 	|  ApgSTMD         	(2022, H. Wang, IEEE T--Cybernetics)
   	|  FracSTMD        	(2023, Xu, M., Neurocomputing)
 	|  STMDv2          	--indevelopment

+tool: Contains additional tools and utilities for data processing and 
visualization.
config: Contains configuration files for setting up parameters and options.
demo: Contains demo scripts showcasing how to use the STMD package.
test: Contains unit tests to verify the correctness of the algorithms.

——Features
+api: Provides high-level functions and classes for easy integration of the
 STMD package into other projects.
+core: Implements core algorithms for small target motion detection, such 
as attention kernels, lateral inhibition, and prediction.
+model: Includes models for detecting small moving objects in video 
sequences.
+tool: Offers additional tools and utilities for data preprocessing, 
visualization, and evaluation.
config: Allows users to configure parameters and options for specific 
applications or scenarios.
demo: Provides demonstration scripts to showcase the usage and capabilities 
of the STMD package.
test: Contains unit tests to ensure the correctness and reliability of the 
algorithms.

——Support and Feedback
If you encounter any issues or have any suggestions while using the STMD 
package, feel free to reach out to me. You can raise issues or submit 
feedback on the GitHub repository or email <mingshuoxu@hotmail.com>. 
I will respond promptly and strive to address your concerns.

--------------------------------------------------------------------------
Small-Target-Motion-Detectors (STMD)
STMD 是一个用于检测图像或视频中小目标运动的 MATLAB 包。它提供了各种功能和工具，
使用户能够有效地检测和跟踪图像或视频中的小目标。

——如何使用
安装
克隆或下载此存储库。

在 MATLAB 中添加路径，将存储库中的文件夹包括到 MATLAB 的搜索路径中。

——示例
STMD 提供了两个 DEMO 文件，演示了如何使用该包进行小目标运动检测：

demo_imgstream.m: 这个 DEMO 文件演示了如何从图像流中读取图像，
并使用 STMD 模型进行目标检测。

demo_vidstream.m: 这个 DEMO 文件演示了如何从视频流中读取视频，
并使用 STMD 模型进行目标检测。

——步骤
实例化 STMD 模型。

创建图像或视频流阅读器。

初始化模型。

在循环中读取每一帧图像或视频，并对其进行推理。

显示推理结果。

——参数调整
您可以根据需要选择不同的输入源，包括演示图像、真实世界图像和模拟图像。

您可以调整模型的参数，例如最大区域大小、输入分辨率等。

注意事项
在运行 DEMO 文件之前，请确保您已安装了 MATLAB 并正确设置了 MATLAB 的环境。

确保您已按照安装说明将存储库添加到 MATLAB 的搜索路径中。

根据您的需求和场景，您可以自由修改和扩展 DEMO 文件，以满足特定的应用需求和场景要求。

——支持和反馈
如果您在使用 STMD 包时遇到任何问题或有任何建议，请随时联系我们。您可以在 GitHub 
存储库上提出问题或提交反馈或者发邮件到<mingshuoxu@hotmail.com>。我会尽快回复并尽
力解决您的问题。

——包结构
+api：包含与STMD包交互的API函数和类。
+core：包含用于运动检测和分析的核心算法和实用程序。
+model：包含用于小目标运动检测的模型和神经网络。
+tool：包含用于数据处理和可视化的附加工具和实用程序。
config：包含用于设置参数和选项的配置文件。
demo：包含演示脚本，展示如何使用STMD包。
test：包含用于验证算法正确性的单元测试。

——特性
+api：提供高级函数和类，用于将STMD包轻松集成到其他项目中。
+core：实现用于小目标运动检测的核心算法，如注意力核、侧向抑制和预测。
+model：包含用于在视频序列中检测小运动物体的模型。
+tool：提供用于数据预处理、可视化和评估的附加工具和实用程序。
config：允许用户为特定应用程序或场景配置参数和选项。
demo：提供演示脚本，展示STMD包的用法和功能。
test：包含单元测试，以确保算法的正确性和可靠性。
