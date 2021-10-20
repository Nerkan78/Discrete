
#!/opt/miniconda3/bin/envs/Nerkan_env_new3/python
import os
import sys
from random import random, shuffle, randint
from collections import namedtuple, Counter
# from shapely.geometry import LineString
from copy import deepcopy
from graph_utils import *
# This code is contributed by Ansh Riyal

    
    

def generate_random_pair_edge(path):
    index_u1 = randint(0, len(path) - 1)
    index_v1 = index_u1 + 1 if index_u1 < len(path) - 1 else 0
    index_u2 = index_u1
    while index_u2 == index_u1 or index_u2 == index_v1:
        index_u2 = randint(0, len(path) - 1)
    index_v2 = index_u2 + 1 if index_u2 < len(path) - 1 else 0
    return (path[index_u1], path[index_v1]) , (path[index_u2], path[index_v2])
        

def Two_OPT(initial_path, dict_points,  epsilon = 0.1, num_iterations = 40):
    path = deepcopy(initial_path)
    k = 0
    condition = True
    while condition:
        pair_of_edges = generate_random_pair_edge(path)
        # untouched_edges = (filter(lambda x : not isInTouch(x[0], x[1]), pairs_of_edges))
        if is_crossed_1(pair_of_edges[0], pair_of_edges[1], dict_points) or random() < epsilon:
            (u1, v1), (u2, v2) = pair_of_edges
            path = swap_two_edges((u1, v1), (u2, v2), path)
        k += 1  
        if k > num_iterations:
            condition = False
    return path
                    

    
def clear_path(initial_path, dict_points, adjacency_matrix, num_iterations=100, history = []):
    path = (initial_path)
    # for k in range(num_iterations):
    k = 0
    while True:
        # print(k)
        edges = ((extract_edges(path)))
        pairs_of_edges = ((combinations(edges, 2)))
        # untouched_edges = (filter(lambda x : not isInTouch(x[0], x[1]), pairs_of_edges))
        crossed_edges = list((filter(lambda x : is_crossed_1(x[0], x[1], dict_points), pairs_of_edges)))
        if len(crossed_edges) == 0:
           break
        else:
            # print(crossed_edges)
            for pair_edges in crossed_edges:
                (u1, v1), (u2, v2) = pair_edges
                path, _ = swap_two_edges((u1, v1), (u2, v2), path, adjacency_matrix, 0)
                history.append(deepcopy(path + [path[0]]))
        k += 1

    print(f'cleared for {k}')
    return path, history
    
def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)    

def K_opt(initial_path, old_weight, adjacency_matrix, points_iterated=None, mode='improve', epsilon_edge = 0.03, epsilon_continue = 0.15, history=[]):
    if mode == 'improve':
        epsilon_edge = -1
        epsilon_continue = -1
        
    path = deepcopy(initial_path)
    if old_weight is None:
        old_weight = weight_of_path(path + [path[0]], adjacency_matrix)
    considered_points = []
    has_changed = False
    # iterations over points
    if points_iterated is None:
        points_iterated = range(len(initial_path))
        # shuffle(points_iterated)
    has_changed = False
    # edges = sorted(extract_edges(path + [path[0]]), key = lambda edge : adjacency_matrix[edge[0]][edge[1]], reverse=True)
    edge_index = 0
    for initial_point in points_iterated:
    # while not has_changed:
        # initial_point = randint(0, len(initial_path)-1)
        # initial_point, next_point = edges[edge_index]
        # print(f'initial point is {initial_point}')
        
        # iterations through k opts
        condition = True
        # closed_points = set([point])
        old_path = deepcopy(path)
        # old_weight = weight_of_path(old_path + [old_path[0]], adjacency_matrix)
        point = initial_point
        k = 1
        while condition:
            # print(f'point is {point}')
            
            index = old_path.index(point)
            next_point = old_path[0] if index + 1 == len(old_path) else old_path[index+1]
            current_edge_weight = adjacency_matrix[point][next_point]
            # find first suitable swap
            new_path = deepcopy(old_path)
            for possible_point in range(len(initial_path)):
                if possible_point == next_point:
                    continue
                if adjacency_matrix[next_point][possible_point] < current_edge_weight or random() < epsilon_edge:
                    break
                
            else:
                condition = False
            if not condition:
                break
            
            
                
            # possible_point = np.argmin(adjacency_matrix[next_point][adjacency_matrix[next_point] != 0])
            # weight = adjacency_matrix[next_point][possible_point]
            # if weight >= current_edge_weight:
                # break
            
            previous_node_for_possible_point = old_path[old_path.index(possible_point)-1]
            new_path, new_weight = swap_two_edges((point, next_point), (previous_node_for_possible_point, possible_point), deepcopy(old_path), adjacency_matrix, old_weight)
            considered_points += [point, next_point, previous_node_for_possible_point, possible_point]
            # new_weight = old_weight - adjacency_matrix[point][next_point] - adjacency_matrix[previous_node_for_possible_point][possible_point] + adjacency_matrix[point][previous_node_for_possible_point] + adjacency_matrix[next_point][possible_point] 
            # assert round(new_weight, 2) == round(weight_of_path(new_path + [new_path[0]], adjacency_matrix), 2), f'{new_weight} {weight_of_path(new_path + [new_path[0]], adjacency_matrix)}'
            if round(new_weight, 2) < round(old_weight, 2) or random() < epsilon_continue:
                old_path = deepcopy(new_path)
                old_weight = new_weight
                point = next_point
                has_changed = True
                k += 1
            else:
                condition = False
                # print('good swap is chosen')
            
           
        # if k > 1:
            # print(point , k)
        path = deepcopy(old_path)
        edge_index += 1
        # if old_path != path:
            # path = deepcopy(old_path)
            # history.append(deepcopy(path + [path[0]]))
    
    
    # print(f'is different {path == initial_path}')
       
    return path, old_weight, considered_points, has_changed
    


def complex_tabu_search(initial_path, points, adjacency_matrix, num_iterations = 100, num_neighbors = 15, epsilon_change_path = 1e+1):
    path = deepcopy(initial_path)
    tabu = []
    L = 40

    best_path = deepcopy(path)
    best_paths = []
    old_weight = weight_of_path(path + [path[0]], adjacency_matrix)
    best_weight = old_weight
    temperature = 0
    
    for epoch in range(num_iterations):
        old_best_weight = best_weight
       
        # path = deepcopy(best_path)
        mode = 'improve'
        # global_points_iterated = list(range(len(initial_path)))
        # shuffle(global_points_iterated)
        best_neighbor_weight = np.inf
        best_neighbor_path = path
        best_considered_points = []
        for n in range(num_neighbors):
        
            # for initial_point in global_points_iterated[:len(global_points_iterated)]:
                # points_iterated = list(range(len(initial_path)))
                # shuffle(points_iterated)
                # points_iterated = [initial_point]
            initial_point = randint(0, len(initial_path) - 1)
            new_path, new_weight, considered_points, has_changed = K_opt(path, old_weight,  adjacency_matrix, [initial_point], mode)
            # new_path = Two_OPT(path, points)
            if not has_changed:
                continue
            # print('has changed')
            seen = False
            for tabu_hash in tabu:
                seen_points, weight = tabu_hash    
                if round(weight, 2) == round(new_weight, 2):
                    seen = True
                    # print('tabu')
                    break
            if not seen:
                
                # assert round(new_weight, 2) == round(weight_of_path(new_path + [new_path[0]], adjacency_matrix), 2)
                # if new_weight > old_weight:
                    # print(f'new weight is {new_weight}')
                    # print(f'old weight is {old_weight}')
                    # print(f'possibility is {np.exp( - epsilon_change_path * (new_weight - old_weight) *  (epoch + 1))}')
                if round(new_weight, 2) < round(best_neighbor_weight, 2) :
                    best_neighbor_path = deepcopy(new_path)
                    best_neighbor_weight = new_weight
                    best_considered_points = deepcopy(considered_points)
                    # print(f'change path, current weight {new_weight}')
                    
                if round(new_weight, 2) < round(best_weight, 2):
                    best_weight = new_weight
                    best_path = deepcopy(new_path)
        # print(f'new_weight is {new_weight} old_weight is {old_weight} possibility is {np.exp( - epsilon_change_path * (new_weight - old_weight + 3) * 2 /  old_weight /   (temperature + 1))}')        
        if round(old_weight, 2) < round(best_neighbor_weight, 2):
            path = deepcopy(best_neighbor_path)
            old_weight = best_neighbor_weight
            tabu.append((best_considered_points, best_neighbor_weight))
            if len(tabu) > L:
                tabu.pop(0)
            temperature = 0
        elif random() < np.exp( - epsilon_change_path * (new_weight - old_weight + 3) *2 / old_weight/  (temperature + 1)):
            path = deepcopy(best_neighbor_path)
            old_weight = best_neighbor_weight
            tabu.append((best_considered_points, best_neighbor_weight))
            if len(tabu) > L:
                tabu.pop(0)
            temperature = 0
            
        else:
            temperature += 1
            
        # print(f'tabu iteration is {epoch} old weight is {old_best_weight} new weight is {best_weight} temperature is {temperature}')
        
                
    return best_path, best_weight
    
def small_tabu_search(initial_path, points, adjacency_matrix, num_iterations = 100, num_neighbors = 30, epsilon_change_path = 1e+1):
    path = deepcopy(initial_path)
    old_weight = weight_of_path(path + [path[0]], adjacency_matrix)
    tabu = []
    L = 40
    best_weight = old_weight
    best_path = deepcopy(path)
    temperature = 0
    # edges = extract_edges(best_path + [best_path[0]])
    # sorted_edges = sorted(edges, key = lambda edge: adjacency_matrix[edge[0]][edge[1]])
    for epoch in range(num_iterations):
        old_best_weight = best_weight
        
        best_weight_neighborhood = np.inf
        k = 0
        while k < num_neighbors:
                              
            while True:
                i = randint(0, len(path) - 1)
                prev_i = i - 1 if i > 0 else len(path) - 1
                next_i = i + 1 if i < len(path) - 1 else 0
                
                
                j = randint(0, len(path) - 1)
                prev_j = j - 1 if j > 0 else len(path) - 1
                next_j = j + 1 if j < len(path) - 1 else 0
                
                if (i != j) and (prev_i != j) and (next_i != j):
                    break
            
            point_i = path[i]
            prev_point_i = path[prev_i]
            next_point_i = path[next_i]
            
            point_j = path[j]
            prev_point_j = path[prev_j]
            next_point_j = path[next_j]
            



                
               # print(f'removed edges are ({prev_point_i}, {point_i} ) ({point_i}, {next_point_i}) ({prev_point_j}, {point_j}) ({point_j}, {next_point_j})')
            # print(f'added edges are ({prev_point_i}, {point_j} ) ({point_j}, {next_point_i}) ({prev_point_j}, {point_i}) ({point_i}, {next_point_j})')
            # if j == next_i:
                # new_weight = old_weight - adjacency_matrix[prev_point_i][point_i] - adjacency_matrix[point_j][next_point_j] \
                                        # + adjacency_matrix[prev_point_i][point_j] + adjacency_matrix[point_i][next_point_j]
            # elif i == next_j:
                # new_weight = old_weight - adjacency_matrix[prev_point_j][point_j] - adjacency_matrix[point_i][next_point_i] \
                                        # + adjacency_matrix[prev_point_j][point_i] + adjacency_matrix[point_j][next_point_i]
            # else:
                # new_weight = old_weight - adjacency_matrix[prev_point_i][point_i] - adjacency_matrix[point_i][next_point_i] - adjacency_matrix[prev_point_j][point_j] - adjacency_matrix[point_j][next_point_j] \
                                        # + adjacency_matrix[prev_point_i][point_j] + adjacency_matrix[point_j][next_point_i] + adjacency_matrix[prev_point_j][point_i] + adjacency_matrix[point_i][next_point_j]
            
            new_path, new_weight = swap_two_edges_by_index((i, next_i), (j, next_j), deepcopy(path), adjacency_matrix, old_weight)
            assert round(new_weight, 2) == round(weight_of_path(new_path + [new_path[0]], adjacency_matrix), 2), f'swap weight is not correct'
            
            seen = False
            for x, y, weight in tabu:
                if ((x == point_i and y == point_j) or (x == point_j and y == point_i)) and round(new_weight, 2) == round(weight, 2):
                    seen = True
                    break
            if seen:
                continue
            
            
           
            
            # p = np.exp( - epsilon_change_path * (new_weight - old_weight) / (temperature+1) - 3)

            
            if round(new_weight, 2) < round(best_weight_neighborhood, 2):
                best_neighbor_path = deepcopy(new_path)
                points_for_tabu = (point_i, point_j, new_weight)
                best_weight_neighborhood = new_weight 
                
            if round(new_weight, 2) < round(best_weight, 2):
                best_weight = new_weight
                best_path = deepcopy(new_path)
            
            
            k += 1
 
                        
        p = np.exp( - epsilon_change_path * (best_weight_neighborhood - old_weight) / (temperature+1) - 3)   
                
        if round(best_weight_neighborhood, 2) < round(old_weight, 2) or random() < p:
            path = deepcopy(best_neighbor_path)
            old_weight = best_weight_neighborhood
            tabu.append(points_for_tabu) 
            if len(tabu) > L:
                tabu.pop(0)
            temperature = 0
        else:
            temperature += 1
        
        
        # if round(old_best_weight) == round(best_weight):
            # temperature += 1
        # else:
            # temperature = 0
        # print(f'tabu iteration is {epoch} old weight is {old_best_weight} new weight is {best_weight} temperature is {temperature} p is {p}')
                    
    return best_path, best_weight
    
    
def fast_local_search(initial_path, initial_weight, augmented_adjacency_matrix, subneighborhoods_activated):
    node_Count = len(initial_path)
    path = deepcopy(initial_path) 
    old_weight = weight_of_path(initial_path + [initial_path[0]], augmented_adjacency_matrix)
    while sum(subneighborhoods_activated) > 0:
        # print(sum(subneighborhoods_activated))
        for i in range(node_Count):
            if subneighborhoods_activated[i] == 1:
                edges = extract_edges(path + [path[0]])
                
                index_i = path.index(i)
                prev_i = index_i - 1 if index_i > 0 else len(path) - 1
                next_i = index_i + 1 if index_i < len(path) - 1 else 0
                prev_point_i = path[prev_i]
                next_point_i = path[next_i]
                
                moves = [((i, next_point_i), edge) for edge in edges] + [((prev_point_i, i), edge) for edge in edges]
                # print(f'weight before {old_weight}' )
                for move in moves:
                    new_path, new_weight = swap_two_edges(move[0], move[1], path, augmented_adjacency_matrix, old_weight)
                    assert round(new_weight, 2) == round(weight_of_path(new_path + [new_path[0]], augmented_adjacency_matrix), 2)
                    if round(new_weight, 2) < round(old_weight, 2):
                        subneighborhoods_activated[move[0][0]] == 1
                        subneighborhoods_activated[move[0][1]] == 1
                        subneighborhoods_activated[move[1][0]] == 1
                        subneighborhoods_activated[move[1][1]] == 1
                        path = deepcopy(new_path)
                        old_weight = new_weight
                        break
                else:
                    subneighborhoods_activated[i] = 0
                # print(f'weight after {old_weight}' )
    return path, old_weight
                
                    
    
def guided_local_search(initial_path, points, adjacency_matrix, num_iterations = 100, num_neighbors = 30, epsilon_change_path = 1e+1):
    node_Count = len(initial_path)
    path = deepcopy(initial_path)
    subneighborhoods_activated = [1] * node_Count

    penalties = np.zeros_like(adjacency_matrix)
    augmented_adjacency_matrix = adjacency_matrix.copy()
    best_path = deepcopy(path)
    old_weight = weight_of_path(path + [path[0]], adjacency_matrix)
    best_weight = old_weight
    alpha  = 0.4
    lambda_ = alpha * old_weight
    
    for epoch in range(num_iterations):
        # if epoch % 10 == 0:
            # subneighborhoods_activated = [1] * node_Count
        old_best_weight = best_weight    
        new_path, new_weight = fast_local_search(path, old_weight, augmented_adjacency_matrix, subneighborhoods_activated)

        edges = extract_edges(new_path + [new_path[0]])
        max_util_edges = []
        max_util = 0
        for u, v in edges:
            util = adjacency_matrix[u][v] / (1 + penalties[u][v])
            if round(max_util, 3) >= round(util, 3) >=  round(max_util, 3) - 3:
                max_util_edges.append((u, v))
            if round(util , 3) > round(max_util, 3):
                max_util_edges = [(u, v)]
                max_util = util
                # print(max_util)
        # print(f'penalized edges {len(max_util_edges)}')
        for u, v in max_util_edges:
            penalties[u][v] += 1
            penalties[v][u] += 1
            subneighborhoods_activated[u] = 1
            subneighborhoods_activated[v] = 1
            
        
        
         
        true_weight = weight_of_path(new_path + [new_path[0]], adjacency_matrix)
            
        if round(true_weight, 2) < round(best_weight, 2):
            best_weight = true_weight
            best_path = deepcopy(new_path)
        
        if round(new_weight, 2) < round(old_weight, 2):
            path = deepcopy(new_path)
            old_weight = new_weight
            
        
        lambda_ = alpha * best_weight
        # print(lambda_)
        augmented_adjacency_matrix = adjacency_matrix + lambda_ * penalties
            

            
        print(f'GFLS iteration is {epoch} old weight is {old_best_weight} new weight is {best_weight} max penalties is {penalties.max()} number of penalties is {(penalties).sum()}')
        
    # print(penalties.max())            
    return best_path, best_weight
                            
                        
                        
        

    
        
        

