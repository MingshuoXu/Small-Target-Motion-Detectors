# Update Log

---
## Update Log - Version 2.0.11 (Feb 05, 2025)

##### Python Enhancements:

1. Bug Fixes: 
    - Resolved critical issues in `inference_gui.py` related to error handling and process termination.

2. Code Refactoring: 
    - Refactored the implementation of `inference_gui` into a class-based structure for improved modularity and maintainability.

3. File Reorganization: 
    - Renamed `inference_gui.py` to `inference_gui_old.py` for archival purposes. 
    - Introduced a simplified version named `inference_gui_single_process.py` to serve as an entry point for beginners.

This version focuses on stability, maintainability, and user accessibility, ensuring a smoother experience for both advanced users and newcomers.

---
## Update Log - Version 2.0.10 (Jan 22, 2025)
1. (Python) Added precompiled `.pyd` files for `STMDNet` and `STMDNetF`, compiled with `Python 3.9` for improved compatibility and performance.
2. (MATLAB) Added precompiled `.p` files for STMDNet and STMDNetF to enhance usability and execution speed.


---
## Update Log - Version 2.0.9 (Jan 9, 2025)
1. (Python) Fix issues in `_clicked_start_img` and `_clicked_end_img` functions.
2. (Python) Resolved the bug preventing correct ordering of image streams named with pure numbers.
3. (Python) Added stepping functionality in both Input and Output GUI.


---
## Update -version 2.0.8 (Dec 13, 2024)
1. (Python) Upgrate the `evaluate_module`.
2. (Python) Fix the iostream.


---
## Update -version 2.0.7 (Nov 14, 2024)
1. (Python) Bind model parameters and their corresponding parameter pointers.
2. (Matlab) `Matlab` temporarily stopped maintenance, then consider directly calling `Python` or `C/C++` APIs.
 

---
## Update -version 2.0.6 (Aug 15, 2024)
1. (Python) Introducing the `evaluate_module`.


---
## Update -version 2.0.5 (May 12, 2024)
1. (Python) Modify the class `CircularList` to be based on the list data structure and annotate it with the `@dataclass` decorator.

---
## Update -version 2.0.4 (May 09, 2024)

1. (Python) Upload all python code.
2. (Python) a better GUI for start `strat_by_python.py`
3. (Matlab) Extend the GUI `strat_by_matlab.m` and fix some mis-typing variable name 
4. (Python) Perfecting `reqiurement.txt` and `setup.py` in Python

---
## Update -version 2.0.3 (April 30, 2024)

1. (Matlab) Rename function `init` to `init_config` to maintain consistency 
    with python.
2. (Matlab) Rename the module tool to util, which stands for utility.
3. (Python) Upload part of python code.
4. (naming_conventions) Update the convention of type naming.


---
## Update -version 2.0.2 (March 12, 2024)

1. (Matlab) Corrected the implementation of the `FeedbackSTMD` model.
2. (Matlab) Added evaluation code for the `RIST` dataset.
3. (Matlab) Added gitignore file.

---
## Update -version 2.0.1 (March 08, 2024)

1. (Matlab) GitHub renaming the download package from `Small-Target-Motion-Detectors` 
    to `Small-Target-Motion-Detectors-main` caused issues with addpath. In this update, 
    this problem have been addressed by ensuring that addpath is not reliant on 
    the name `Small-Target-Motion-Detectors`. Instead, it now identifies 
    `/matlab/smalltargetmotiondetectors/` to prevent addpath from being affected by GitHub's changes.
2. (Matlab) By changing the format of the demo dataset from `TIFF` to `JPG`, 
    the size of the demodata has been reduced from `70MB` to `12MB`.

---
## First commit -version 2.00 (March 07, 2024)

This project is a comprehensive upgrate of the previous version [Version 1: ClassSTMD](https://github.com/MingshuoXu/ClassSTMD), mainly changing the implementation framework by separating the model from the core functionality, making it more convenient for combination.

1. (Matlab) Separation of Model and Core Functionality: In this version, we have separated the implementation of the model from the core functionality, making it easier to develop and maintain them independently. This design facilitates users to combine and extend models according to their needs.
2. (Matlab) Framework Update: We have adopted a new implementation framework improving code readability and scalability. The new framework makes the code clearer, easier to understand, and modify.
3. (Matlab) Performance Optimization: By optimizing and refactoring the code, we have improved the performance and efficiency of the model. The new version maintains accuracy while enhancing processing speed and resource utilization.