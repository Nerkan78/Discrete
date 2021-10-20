#!/opt/miniconda3/bin/envs/Nerkan_env_new3/python
# -*- coding: utf-8 -*-


import math
from collections import namedtuple, Counter
from TSP import *
from graph_utils import *
Point = namedtuple("Point", ['x', 'y'])
import matplotlib.pyplot as plt
import cv2 
from copy import deepcopy
from time import time


def record_video(history, points, video_name, fps=5):
    
    min_y = min(map(lambda point : point.y, points))
    min_x = min(map(lambda point : point.x, points))
    max_x = max(map(lambda point : point.x, points))
    max_y = max(map(lambda point : point.y, points))
    plt.plot(list(map(lambda x: points[x].x, history[0][:-1])), list(map(lambda x: points[x].y, history[0][:-1])), '-o', c='blue')
    plt.yticks(np.arange(min_y, max_y, 5))
    plt.xticks(np.arange(min_x, max_x, 5))
    plt.savefig('graph.png')
    plt.clf()
    frame = cv2.imread('graph.png')

    height, width = frame.shape[0], frame.shape[1] 
    size = (width,height)
    out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    print(len(history))
    for i, path in enumerate(history):
        print(i)
        plt.plot(list(map(lambda x: points[x].x, path[:-1])), list(map(lambda x: points[x].y, path[:-1])), '-o', c='blue')
        plt.yticks(np.arange(min_y, max_y, 5))
        plt.xticks(np.arange(min_x, max_x, 5))
        plt.savefig('graph.png')
        plt.clf()
        frame = cv2.imread('graph.png')
      
        out.write(frame)
    out.release()

def solve_it(input_data):
    # Modify this code to run your optimization algorithm
    start_time = time()
    # parse the input
    lines = input_data.split('\n')

    nodeCount = int(lines[0])
    
    points = []
    for i in range(1, nodeCount+1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))
    min_y = min(map(lambda point : point.y, points))
    min_x = min(map(lambda point : point.x, points))
    max_x = max(map(lambda point : point.x, points))
    max_y = max(map(lambda point : point.y, points))
    # build a trivial solution
    # visit the nodes in the order they appear in the file

    matrix = (create_adjacency_matrix(points))
    sorted_matrix = np.apply_along_axis(np.argsort, 1, matrix)
    print(sorted_matrix)
    print([1,2,3] in [[1,2,3], [1,2]])
    
    best_weight = np.inf
    for i in range(15):
        print(f'i is {i}')
        # path = create_initial_path(matrix, sorted_matrix, 0)

        path = list(range(nodeCount))
        shuffle(path)
        
        start_weight = weight_of_path(path + [path[0]], matrix)
        if start_weight < best_weight:
            best_weight = start_weight
            best_path = deepcopy(path)
        print(f'start weight {start_weight }')

        path, weight = guided_local_search(path, points, matrix, num_iterations = 200)
        true_weight = weight_of_path(path + [path[0]], matrix)
        
        
        
        # path = clear_path(path + [path[0]], points)[0][:-1]
        # fig = plt.figure(figsize = (20, 20))
        # plt.plot(list(map(lambda x: points[x].x, path + [path[0]])), list(map(lambda x: points[x].y, path + [path[0]])), '-o')
        # plt.savefig(f'graph_{i}.png')
        # plt.clf()
        print(f'end weight {true_weight}') 
        
        if true_weight < best_weight:
            best_weight = true_weight
            best_path = deepcopy(path)
            
    
    path = best_path # create_initial_path(matrix, sorted_matrix, 0)
    start_weight = weight_of_path(path + [path[0]], matrix)
    if start_weight < best_weight:
        best_weight = start_weight
        best_path = deepcopy(path)
    print(f'start final')
    print(f'start weight {start_weight }')

    # path, weight = complex_tabu_search(path, points, matrix, num_iterations = 300)
    # path = clear_path(path + [path[0]], points)[0][:-1]
    weight = weight_of_path(path + [path[0]], matrix)
    # fig = plt.figure(figsize = (20, 20))
    # plt.plot(list(map(lambda x: points[x].x, path + [path[0]])), list(map(lambda x: points[x].y, path + [path[0]])), '-o')
    # plt.yticks(np.arange(min_y, max_y, 5))
    # plt.xticks(np.arange(min_x, max_x, 5))
    # plt.savefig(f'graph_neighbors.png')
    # plt.clf()
    print(f'end weight {weight}') 
    if weight < best_weight:
        best_weight = weight
        best_path = deepcopy(path)
        
        
        
        
    # cristofides_path = list(create_initial_path(matrix, 20)[:-1])
    # print('start local search')
    # for epoch in range(10):
        # print(epoch)
        # cristofides_path, history = clear_path(cristofides_path, points, 1)
        # # print(f'path after clearance : \n{cristofides_path}\n' )
        # tmp = deepcopy(cristofides_path)
        # cristofides_path, history = K_opt(cristofides_path, matrix, [])
        # # print(f'path after K opt : \n{cristofides_path}\n' )
        # if tmp == cristofides_path:
            # break

   

    
    
    

    # record_video(history, points, 'history.avi')
    solution = range(0, nodeCount)
    # solution = clear_path([31, 20, 25, 21, 43, 50, 39, 49, 17, 32, 48, 22, 33, 0, 5, 2, 28, 10, 9, 45, 26, 47, 1, 6, 36, 12, 30, 37, 42, 29, 38, 15, 14, 44, 16, 11, 40, 18, 19, 7, 13, 35, 23, 4, 8, 34, 24, 46, 3, 41, 27, 31], points)[0]
    # solution = clear_path(best_path + [best_path[0]], points, matrix)[0][:-1]
    solution = best_path
    # solution = cristofides_path
    # fig = plt.figure(figsize = (20, 20))
    # plt.plot(list(map(lambda x: points[x].x, solution + [solution[0]])), list(map(lambda x: points[x].y, solution + [solution[0]])), '-o')
    
    # plt.yticks(np.arange(min_y, max_y, 5))
    # plt.xticks(np.arange(min_x, max_x, 5))
    
    # plt.savefig('final_graph.png')
    # calculate the length of the tour
    obj = length(points[solution[-1]], points[solution[0]])
    for index in range(0, nodeCount-1):
        obj += length(points[solution[index]], points[solution[index+1]])

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))
    
    end_time = time()
    print(f'Execution time is {end_time - start_time}')
    return output_data


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')

