import numpy as np
from copy import deepcopy


def parse_input():
    N = int(input())
    points = []
    for i in range(N):
        point = input().split(' ')
        point_number = int(point[0])
        point_x = float(point[1])
        point_y = float(point[2])
        points.append([point_number - 1, np.array([point_x, point_y])])
    return points


def create_adjacency_matrix(points):
    N = len(points)
    matrix = np.zeros((N, N))
    for i, point_1 in enumerate(points):
        for j, point_2 in enumerate(points):
            matrix[i][j] = np.linalg.norm([point_1.x - point_2.x, point_1.y - point_2.y])
    return matrix



def swap_two_edges(edge1,edge2, path, adjacency_matrix, old_weight, swap = True):
    new_path = deepcopy(path)
    (u1, v1), (u2, v2) = (edge1,edge2)   
    
    if len(np.unique([u1, v1, u2, v2])) != 4:
        # print(f'repeated values ({u1}, {v1}) ({u2}, {v2})')
        return path, old_weight
    # print(f'non repeated values ({u1}, {v1}) ({u2}, {v2})')                       
    index_u1 = path.index(u1)
    index_v1 = path.index(v1)
    if index_v1 == 0: 
        index_v1 = len(path) - 1
    if index_u1 > index_v1:
        v1, u1 = u1, v1
        index_v1, index_u1 = index_u1, index_v1
    index_u1 += 1
    
    index_u2 = path.index(u2)
    index_v2 = path.index(v2)
    if index_v2 == 0: 
        index_v2 = len(path) - 1
    if index_u2 > index_v2:
        v2, u2 = u2, v2
        index_v2, index_u2 = index_u2, index_v2
    index_u2 += 1
    
    if swap:
        if index_u2 > index_u1:
            new_path[index_v1 : index_u2] = new_path[::-1][len(path) - index_u2 : len(path) - index_v1]
        else:
            new_path[index_v2 : index_u1] = new_path[::-1][len(path) - index_u1 : len(path) - index_v2]
    return new_path, old_weight - adjacency_matrix[u1][v1] - adjacency_matrix[u2][v2] + adjacency_matrix[u1][u2] + adjacency_matrix[v1][v2]