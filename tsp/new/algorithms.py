import numpy as np
from copy import deepcopy
from utils import *

def K_opt(path, matrix, K_max = 10):
    # one iteration
    nodeCount = len(path)
    old_path = deepcopy(path)
    old_weight = matrix[path[-1]][path[0]]
    for index in range(0, nodeCount-1):
        old_weight +=  matrix[path[index]][path[index+1]]
        
    for index in range(nodeCount):
        actual_path = deepcopy(path)
        point = path[index]
        alterations = []
        old_actual_weight = old_weight
        
        for k in range(K_max):
            actual_index = actual_path.index(point)
            next_point = actual_path[actual_index+1 if actual_index < nodeCount-1 else 0]
            
            # pick new vertex
            distances = matrix[next_point, :].copy()
            distances[next_point] = np.inf
            distances[actual_path[actual_index+2 if actual_index < nodeCount-2 else 0]] = np.inf
            distances[point] = np.inf
            new_neighbor_for_new_point = np.random.choice(list(range(nodeCount)), p=(1/distances) / sum( 1/ distances))
            
            index_new_neighbor = actual_path.index(new_neighbor_for_new_point)
            if index_new_neighbor == 0:
                prev_neighbor = actual_path[-1]
            else:
                prev_neighbor = actual_path[index_new_neighbor-1]
                
            # print(f'path is {actual_path} point is {point} next_point is {next_point} new neighbor is {new_neighbor_for_new_point} prev os {prev_neighbor}')
            edge1 = (point, next_point)
            edge2 = (prev_neighbor, new_neighbor_for_new_point)
            
            new_path, new_weight = swap_two_edges(edge1,edge2, actual_path, matrix, old_actual_weight, swap = True)
            alterations.append((deepcopy(new_path), new_weight, k))
            actual_path = deepcopy(new_path)
            old_actual_weight = new_weight
        
        best_path, best_weight, best_k = min(alterations, key = lambda x : x[1])
        # print(f'K is {best_k+2} weight is {best_weight}')
        if best_weight < old_weight:
            path = deepcopy(best_path)
            old_weight = best_weight
            
    return path, old_weight
        
            
