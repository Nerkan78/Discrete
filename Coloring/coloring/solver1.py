#!/usr/bin/python
# -*- coding: utf-8 -*-
import os 
import subprocess
from subprocess import check_output
import re
import math
import networkx as nx
from time import time
import numpy as np

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    with open('data.dzn', 'w') as f:
    
        lines = input_data.split('\n')

        first_line = lines[0].split()
        node_count = int(first_line[0])
        edge_count = int(first_line[1])
        
        f.write(f"N_Vertices = {node_count};\n")
        f.write(f"N_Edges = {edge_count};\n")
        
        node_degrees = [0] * node_count
        
        first_nodes = []
        second_nodes = []
        edges = []
        for i in range(1, edge_count + 1):
            line = lines[i]
            parts = line.split()
            edges.append((int(parts[0]), int(parts[1])))
            node_degrees[int(parts[0])] += 1
            node_degrees[int(parts[1])] += 1
            
            first_nodes.append(int(parts[0]) + 1)
            second_nodes.append(int(parts[1]) + 1)
            # f.write(f"| {int(parts[0]) + 1}, {int(parts[1]) + 1} ")
        f.write(f"Adjacency_First_Node = [")
        f.write(', '.join(map(str, first_nodes)))
        f.write(f"];\n")
        f.write(f"Adjacency_Second_Node = [")
        f.write(', '.join(map(str, second_nodes)))
        f.write(f"];\n")
    
        # graph = nx.Graph(edges)
        min_color = 2 #nx.graph_clique_number(graph)
        max_color = node_count // 2
        f.write(f"MinColor = {min_color};\n")
        f.write(f"MaxColor = {max_color};\n")
    coloring = np.arange(node_count)
    adjacency_matrix =  np.zeros((node_count, node_count)).astype(int)
    for edge in edges:
        adjacency_matrix[edge[0]][edge[1]] = 1
        adjacency_matrix[edge[1]][edge[0]] = 1
    start_time = time()
    node_order = np.argsort(node_degrees)[::-1]
    print(coloring)
    for k in range(10):
        
        # while True:
            # classes = [[i, np.where(coloring == i)[0]] for i in range(max(coloring+1))]
            # changed_colors = 0
            # for class1 in classes:
                # for class2 in classes:
                    # if class1[0] == class2[0]:
                        # continue
                    
                    # big_class, small_class = (class1, class2) if len(class1[1]) >= len(class2[1]) else (class2, class1)
                    # for node in small_class[1]:
                        # if not adjacency_matrix[node][big_class[1]].any():
                            # coloring[node] = big_class[0]
                            # big_class[1] = np.append(big_class[1], node)
                            # small_class[1] = np.delete(small_class[1], np.where(small_class[1] == node))
                            # changed_colors += 1
                            # print(f'node {node} old color {small_class[0]} new_color {big_class[0]}')
            # print(changed_colors)
            # if changed_colors == 0:
                # break
                    
        # print(coloring)
        print(max(coloring))
        
        while True:
            changed_colors = 0
            for node in node_order:
                neighbors = np.where(adjacency_matrix[node] != 0)[0]
                available_color = set(np.arange(node_count)) - set(coloring[neighbors])
                proposed_color = min(available_color)
                if proposed_color < coloring[node]:
                    coloring[node] = proposed_color
                    changed_colors += 1
            print(changed_colors)
            if changed_colors == 0:
                break
        print(f'preprocessing is done, max color is {max(coloring)}')
    
 
   
    print(time() - start_time)
    
    # raw_answer = str(check_output("minizinc graph_coloring.mzn data.dzn", shell=True))
    # answer = re.findall('[0-9]+', raw_answer)
    # output_data = str(max(map(int, answer)) + 1) + ' ' + str(1) + '\n'
    # output_data += ' '.join(answer)
    
    
    # OPTIMAL = 0
    # if "==========" in mz_output:
        # OPTIMAL = 1
        
    # answer = re.findall('[0-9]+', mz_output)
    # output_data = str(len(set(answer))) + ' ' + str(OPTIMAL) + '\n'
    # output_data += ' '.join(answer)
    output_data = 'output'
    
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
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')

