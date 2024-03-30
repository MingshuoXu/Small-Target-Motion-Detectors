# MatrixNMS Toolbox
The MatrixNMS Toolbox is a MATLAB toolkit designed for performing Non-Maximum Suppression (NMS) in matrices. 
It offers various methods for suppressing non-maximum values in a matrix, along with automatic selection of methods based on matrix size.

## Features
### maxRegionSize: 
Specifies the size of the region for maximum operation.

### method: 
Allows selection of the method used for non-maximum suppression.

## Methods
### nms: 
Executes non-maximum suppression on the input matrix.
### select_auto_method: 
Automatically selects the suppression method based on the input matrix size.
### sort_nms: 
Implements NMS using a sorting method.
### conv2_nms: 
Utilizes the conv2 function for NMS.
### bubble_nms: 
Implements NMS using a bubble method.
### greedy_nms: 
Executes NMS using a greedy method.
### mappingAutoMethod: 
Maps auto method based on key-value pairs.

## Usage
1. Add the toolbox to MATLAB's search path.

    addpath('/path/to/toolbox'); 

2. Import the toolbox.

    import smalltargetmotiondectors.tool.*;

3. Create a MatrixNMS object specifying the maxRegionSize and method.

    obj = MatrixNMS(5, 'sort');

4. Execute non-maximum suppression on the input matrix.

    nmsMatrix = obj.nms(input);

## Example

    addpath('/path/to/toolbox/matlab');
    import smalltargetmotiondectors.tool.*;

    % Create a MatrixNMS object with maxRegionSize 5 and 'sort' method
    obj = MatrixNMS(5, 'sort');

    % Perform NMS on the input matrix
    nmsMatrix = obj.nms(input);

---

# MatrixNMS 工具箱
MatrixNMS 工具箱是一个专为在矩阵中执行非最大值抑制（NMS）而设计的 MATLAB 工具包。
它提供了多种方法来抑制矩阵中的非最大值，并根据矩阵大小自动选择方法。

## 特点
### maxRegionSize: 
指定最大化操作的区域大小。

### method: 
允许选择用于非最大值抑制的方法。
## 方法
### nms: 
在输入矩阵上执行非最大值抑制。
### select_auto_method: 
根据输入矩阵大小自动选择抑制方法。
### sort_nms: 
使用排序方法实现 NMS。
### conv2_nms: 
使用 conv2 函数进行 NMS。
### bubble_nms: 
使用冒泡方法实现 NMS。
### greedy_nms: 
使用贪婪方法执行 NMS。
### mappingAutoMethod: 
基于键值对映射自动方法。

## 使用方法
1. 将工具箱添加到 MATLAB 的搜索路径中。

    addpath('/path/to/toolbox');

2. 导入工具箱。

    import smalltargetmotiondectors.tool.*;

3. 创建一个 MatrixNMS 对象，指定 maxRegionSize 和 method。

    obj = MatrixNMS(5, 'sort');

4. 在输入矩阵上执行非最大值抑制。

    nmsMatrix = obj.nms(input);

## 示例
    
    addpath('/path/to/toolbox');
    import smalltargetmotiondectors.tool.*;

    % 创建一个 maxRegionSize 为 5，方法为 'sort' 的 MatrixNMS 对象
    obj = MatrixNMS(5, 'sort');

    % 在输入矩阵上执行 NMS
    nmsMatrix = obj.nms(input);
