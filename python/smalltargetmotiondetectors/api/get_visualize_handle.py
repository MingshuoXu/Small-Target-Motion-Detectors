from ..util.iostream import Visualization

def get_visualize_handle(className=None, 
                         showThreshold=None, 
                         width = 8, 
                         height = 5, 
                         dpi = 100):
    """
    Returns a handle to a visualization object based on the given class name.

    Parameters:
        className (str): Name of the visualization class. If None, a default visualization object is created.
        showThreshold (bool): Whether to show threshold. Default is None.

    Returns:
        Visualization: Handle to the visualization object.
    """

    if className and showThreshold is not None:
        objVisualization = Visualization(className, showThreshold)
    elif className:
        objVisualization = Visualization(className)
    else:
        objVisualization = Visualization()
    
    objVisualization.create_fig_handle(width=width, height =height, dpi=dpi)

    return objVisualization

