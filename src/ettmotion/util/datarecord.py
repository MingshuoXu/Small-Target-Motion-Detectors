from dataclasses import dataclass
from typing import Any, List


@dataclass
class CircularList(list):
    """
    CircularList represents a circular buffer for storing input matrices.
    """
    initLen: int = 0    # Default length of the circular buffer
    pointer: int = -1   # Pointer to current position in the circular buffer

    def __post_init__(self) -> List:
        """
        Post-initialization method to initialize the circular buffer with empty list.
        """
        if self.initLen:
            self.extend([None] * self.initLen)

    def reset(self) -> None:
        """
        Method to reset the circular buffer to a new length.
        """
        self.clear()  # clear List
        self.__post_init__() # Reinitialize CircularList object
        self.pointer = -1 # Reset the pointer to initial position

    def move_pointer(self) -> None:
        """
        Method to move the circular buffer pointer to the next position.
        """
        self.pointer = (self.pointer + 1) % self.initLen

    def cover(self, iptMatrix: Any) -> None:
        """
        Method to cover the current position of the circular buffer with an input matrix.

        Parameters:
        - iptMatrix: Input matrix to cover the current position.
        """
        self[self.pointer] = iptMatrix

    def record_next(self, iptMatrix: Any) -> None:
        """
        Method to record an input matrix in the circular buffer, after moving the pointer to the next position.

        Parameters:
        - iptMatrix: Input matrix to be recorded.
        """
        self.move_pointer()
        self.cover(iptMatrix)


class ModelNameMapping:
    """
    ModelNameMapping represents a mapping between model names and their corresponding class name.
    """
    modelName: list = []    # Model name
    classNameHandle: str = ''  # Visualization handle

    def __init__(self, modelName: list, classNameHandle: str) -> None:
        """
        Constructor to initialize the model name and visualization handle.

        Parameters:
        - model_name: Model name
        - visualize_handle: Visualization handle
        """
        self.modelName = modelName
        self.classNameHandle = classNameHandle


if __name__ == "__main__":
    A = CircularList(3)
    B = CircularList(9)
    A.record_next(1)
    B.record_next(100)
    A.record_next(-6)
    A.record_next('A')
    print(A[:])  
    
    A.initLen = 5
    A.reset()
    A.record_next(0)
    print(A)  
    print(A[:])

    print(B) 
    print(B[:])  
    B.cover('B')  
    B.record_next([1,2])  
    print(B[:])  

    C = CircularList()
    print(C)
    print(C[:])
    print(len(A))

