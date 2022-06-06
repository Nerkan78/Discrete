from copy import deepcopy
import numpy as np

def check_zero_colors(coloring, color_classes, bad_edges):
    for color in range(len(coloring)):
        # print(coloring)
        if len(color_classes[color]) > 0:
            continue
        for substitute_color in range(color+1, len(coloring)):
            if len(color_classes[substitute_color]) > 0:
                break
        else:
            return coloring, color_classes, bad_edges
        
        for node in range(len(coloring)):
            if coloring[node] == substitute_color:
                coloring[node] = color
        color_classes[color] = deepcopy(color_classes[substitute_color])
        bad_edges[color] = deepcopy(bad_edges[substitute_color])
        
        color_classes[substitute_color] = []
        bad_edges[substitute_color] = []
    return coloring, color_classes, bad_edges
            
            
    
def local_search(coloring, color_classes, bad_edges, edges):
    '''
        coloring : list of ints, represents color of each node
        color_classes : list of size N of lists of int, represents nodes of each color
        bad_edges : list of lists of edges, represents bad edges of each class
        edges : list of list of nodes, represents neighbors of each node
    '''
    node_number = len(coloring)
    condition = True
    while condition:
        condition = False
        print(len(np.unique(coloring)))
        for node in range(node_number):

            old_color = coloring[node]
            # coloring, color_classes, bad_edges = check_zero_colors(coloring, color_classes, bad_edges)
            number_colors = sum(map(lambda x : 1 if len(x) > 0 else 0, color_classes))
            for new_color in range(number_colors + 1 if number_colors < node_number else number_colors):
                if new_color == old_color:
                    continue
                B_old_i = len(bad_edges[old_color])
                B_old_j = len(bad_edges[new_color])
                
                delta_B_i = sum(map(lambda x : 1 if node in x else 0, bad_edges[old_color]))
                delta_B_j = sum(map(lambda x: 1 if coloring[x] == new_color else 0, edges[node]))
                
                C_i = len(color_classes[old_color])
                C_j = len(color_classes[new_color])
                
                delta = 2 * (B_old_i - delta_B_i) * (C_i - 1) - (C_i - 1) ** 2 + 2 * (B_old_j + delta_B_j) * (C_j + 1) - (C_j + 1 ) ** 2 - \
                        ( 2 * B_old_i * C_i - C_i ** 2 + 2 * B_old_j * C_j - C_j ** 2)
                
                if delta < 0:
                    # print(delta)
                    coloring[node] = new_color
                    bad_edges[old_color] = list(filter(lambda x: node not in x, bad_edges[old_color]))
                    bad_edges[new_color] += [(node, neighbor) for neighbor in filter(lambda x: coloring[x] == new_color, edges[node])]
                    color_classes[old_color].remove(node)
                    color_classes[new_color].append(node)
                    condition = True
                    break
    return coloring, color_classes, bad_edges

with open('data\gc_50_3' , 'r') as f:
    input_data = f.read()
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    first_nodes = []
    second_nodes = []
    edges = [[] for i in range(node_count)]
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        first_node = int(parts[0])
        second_node = int(parts[1])
        
        edges[first_node].append(second_node)
        edges[second_node].append(first_node)

coloring = list(range(node_count))
color_classes = [[i] for i in range(node_count)]
bad_edges = [[]] * node_count
# print(edges)
coloring, color_classes, bad_edges  = local_search(coloring, color_classes, bad_edges, edges)
coloring, color_classes, bad_edges = check_zero_colors(coloring, color_classes, bad_edges)
# print(coloring)
# print(color_classes)
# print(bad_edges)
print(coloring)
print(max(coloring)+1)
