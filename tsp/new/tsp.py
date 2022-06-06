import numpy as np
from time import time
from copy import deepcopy
from utils import *
from algorithms import *
import matplotlib.pyplot as plt
   

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


start_time = time()

file_location = 'tsp_51_1'
with open(file_location, 'r') as input_data_file:
    input_data = input_data_file.read()
    lines = input_data.split('\n')
    nodeCount = int(lines[0])

points = []
for i in range(1, nodeCount+1):
    line = lines[i]
    parts = line.split()
    points.append(Point(float(parts[0]), float(parts[1])))
    

matrix = (create_adjacency_matrix(points))
sorted_matrix = np.apply_along_axis(np.argsort, 1, matrix)
print(sorted_matrix)

initial_path = list(range(nodeCount))
print(initial_path)
path = deepcopy(initial_path)
for i in range(200):
    path, weight = K_opt(path, matrix, K_max = 2 + i // 10)
print(path)
print(weight)


min_y = min(map(lambda point : point.y, points))
min_x = min(map(lambda point : point.x, points))
max_x = max(map(lambda point : point.x, points))
max_y = max(map(lambda point : point.y, points))
plt.plot(list(map(lambda x: points[x].x, path[:-1])), list(map(lambda x: points[x].y, path[:-1])), '-o', c='blue')
plt.yticks(np.arange(min_y, max_y, 5))
plt.xticks(np.arange(min_x, max_x, 5))
plt.savefig('graph.png')
        
        






















    