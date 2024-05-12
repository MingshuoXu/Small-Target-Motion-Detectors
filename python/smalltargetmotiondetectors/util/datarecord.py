class CircularList:
    """
    CircularList represents a circular buffer for storing input matrices.

    Parameters:
    - len: Length of the circular buffer.

    Methods:
    - __init__(lenCell): Constructor method to initialize the CircularCell object with the provided length.
    - init_config(): Initialization method to initialize the circular buffer with empty cells.
    - move_point(): Method to move the circular buffer pointer to the next position.
    - circrecord(iptMatrix): Method to record an input matrix in the circular buffer, moving the pointer to the next position.
    - cover(iptMatrix): Method to cover the current position of the circular buffer with an input matrix.
    """

    def __init__(self, lenCell=None):
        """
        Constructor method to initialize the CircularCell object with the provided length.
        
        Parameters:
        - lenCell: Length of the circular buffer.
        """
        self.len = lenCell
        self.point = 0
        if lenCell is not None:
            self.init_config()            

    def init_config(self):
        """
        Initialization method to initialize the circular buffer with empty cells.
        """
        if self.len is None:
            self.len = 2
        self.data = [None] * self.len
        
    def move_point(self):
        """
        Method to move the circular buffer pointer to the next position.
        """
        if self.point == self.len-1:
            self.point =  0
        else:
            self.point += 1

    def circrecord(self, iptMatrix):
        """
        Method to record an input matrix in the circular buffer, moving the pointer to the next position.
        
        Parameters:
        - iptMatrix: Input matrix to be recorded.
        """
        self.move_point()
        self.cover(iptMatrix)

    def cover(self, iptMatrix):
        """
        Method to cover the current position of the circular buffer with an input matrix.
        
        Parameters:
        - iptMatrix: Input matrix to cover the current position.
        """
        self.data[self.point] = iptMatrix
