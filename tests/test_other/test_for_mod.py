from time import *

iters = 100000000

initLen = 47


point1 = 0
point2 = 0


startTime = time()
for _ in range(iters):
    point1 += 1
    if point1 == initLen:
        point1 = 0
endTime = time()
print('Time used if = :' + str((endTime - startTime) *1000) + 'ms')

startTime = time()
for _ in range(iters):
    point2 = (point2 + 1) % initLen
endTime = time()
print('Time used % = :' + str((endTime - startTime) *1000) + 'ms')

print(point1)
print(point2)

