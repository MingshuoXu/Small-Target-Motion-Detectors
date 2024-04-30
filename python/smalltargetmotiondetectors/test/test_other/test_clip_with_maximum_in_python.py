import numpy as np
import timeit

arr = np.random.randn(250, 500) - 0.5
_times = 1000

# test np.clip()
clip_time = timeit.timeit(lambda: np.clip(arr, 0, None), number=_times)

# test np.maximum()
maximum_time = timeit.timeit(lambda: np.maximum(arr, 0), number=_times)

print("np.clip()    执行时间：", clip_time)
print("np.maximum() 执行时间：", maximum_time)

