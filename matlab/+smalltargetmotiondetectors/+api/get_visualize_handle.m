function hVisualization = get_visualize_handle(className, showThreshold)
    % GET_VISUALIZE_HANDLE returns a handle to a visualization object based
    % on the given class name.
    % 
    % Syntax:
    %   hVisualization = get_visualize_handle(className)
    % 
    % Input:
    %   className - Name of the visualization class. 
    % If empty, a default visualization object is created.
    % 
    % Output:
    %   hVisualization - Handle to the visualization object.
    % 
    % Example:
    %   % Creates a default visualization object
    %   hVis = get_visualize_handle(); 
    %   % Creates a visualization object of class 'MyVisualizationClass'
    %   hVis = get_visualize_handle('MyVisualizationClass');

    % Get the full path of this file
    filePath = mfilename('fullpath');
    %   Find the index of '/matlab/+smalltargetmotiondetectors/'
    % in the file path
    indexPath = strfind(filePath, ...
        [filesep, 'matlab', filesep, '+smalltargetmotiondetectors', filesep]);
    % Add the path to the package containing the models
    addpath(filePath(1:indexPath(end)+7));

    import smalltargetmotiondetectors.util.iostream.*;

    if nargin == 2
        docIdx = strfind(className, '.');
        hVisualization = Visualization(...
            className(docIdx(end)+1:end), ...
            showThreshold);
    elseif nargin == 1
        docIdx = strfind(className, '.');
        hVisualization = Visualization(className(docIdx(end)+1:end));
    elseif nargin == 0
        hVisualization = Visualization();
    end

    hVisualization.create_fig_handle();
end
