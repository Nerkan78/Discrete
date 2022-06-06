#!/opt/miniconda3/bin/envs/Nerkan_env_new3/python
import os
import sys
from random import random, shuffle, randint
from collections import namedtuple, Counter
# from shapely.geometry import LineString
from copy import deepcopy
# class Point:
	# def __init__(self, x, y):
		# self.x = x
		# self.y = y
Point = namedtuple("Point", ['x', 'y'])

# Given three colinear points p, q, r, the function checks if
# point q lies on line segment 'pr'
def onSegment(p, q, r):
	if ( (q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and
		(q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))):
		return True
	return False

def orientation(p, q, r):
	# to find the orientation of an ordered triplet (p,q,r)
	# function returns the following values:
	# 0 : Colinear points
	# 1 : Clockwise points
	# 2 : Counterclockwise
	
	# See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/
	# for details of below formula.
	
	val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y))
	if (val > 0):
		
		# Clockwise orientation
		return 1
	elif (val < 0):
		
		# Counterclockwise orientation
		return 2
	else:
		
		# Colinear orientation
		return 0

# The main function that returns true if
# the line segment 'p1q1' and 'p2q2' intersect.
def doIntersect(p1,q1,p2,q2):
	
	# Find the 4 orientations required for
	# the general and special cases
	o1 = orientation(p1, q1, p2)
	o2 = orientation(p1, q1, q2)
	o3 = orientation(p2, q2, p1)
	o4 = orientation(p2, q2, q1)

	# General case
	if ((o1 != o2) and (o3 != o4)):
		return True

	# Special Cases

	# p1 , q1 and p2 are colinear and p2 lies on segment p1q1
	if ((o1 == 0) and onSegment(p1, p2, q1)):
		return True

	# p1 , q1 and q2 are colinear and q2 lies on segment p1q1
	if ((o2 == 0) and onSegment(p1, q2, q1)):
		return True

	# p2 , q2 and p1 are colinear and p1 lies on segment p2q2
	if ((o3 == 0) and onSegment(p2, p1, q2)):
		return True

	# p2 , q2 and q1 are colinear and q1 lies on segment p2q2
	if ((o4 == 0) and onSegment(p2, q1, q2)):
		return True

	# If none of the cases
	return False

	


# from shapely.geometry import LineString
from itertools import combinations
from copy import deepcopy
import numpy as np
from collections import defaultdict, Counter
import networkx as nx
import math


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

def create_graph(matrix):
    N = len(matrix)
    g = Graph(N)
    for i in range(N):
        for j in range(i+1,N):
            if i != j:
                g.addEdge(i,j, matrix[i,j])
    return g


def create_initial_path(matrix, sorted_matrix, start_node=None, mode='NearestNeighbour'):
    N = len(matrix)
    if start_node is None:
#         start_node = np.random.choice(np.arange(N))
        start_node = 0
    if mode == 'NearestNeighbour':
        path = [start_node]
        current_node = start_node
        while len(path) < N:
            for neighbor in sorted_matrix[current_node][1:]:
                if neighbor not in path:
                    path.append(neighbor)
                    current_node = neighbor
                    break
        return path
        
    elif mode == 'Cristofides':
        graph = create_graph(matrix)
        result = graph.KruskalMST()
        print(' MST is found')
        # result - минимальное остовное дерево

        # Теперь ищем вершины с нечетными степенями

        c = Counter()
        for edge in result:
            c += {edge[0] : 1}
            c += {edge[1] : 1}
        odd_nodes = sorted(list(filter( lambda x : c[x] % 2 == 1, c )))

        # Ищем паросочетание минимального веса. Для этого ищем максимальное среди ребер обратного веса.
        eps = 1e-10
        G = nx.from_numpy_matrix(1 / (matrix[np.ix_(odd_nodes,odd_nodes) ] + eps), create_using=nx.Graph)
        mapping = {i : x for i, x in enumerate(odd_nodes)}
        G = nx.relabel_nodes(G, mapping)
        edges = nx.algorithms.matching.max_weight_matching(G)
        print('matching edges are found')
        # Добавляем ребра в граф
        for edge in edges:
            u, v = edge
            result.append([u, v, matrix[u][v]])

        # Составляем эйлеров цикл
        G = nx.MultiGraph()
        G.add_weighted_edges_from(result)

        best_weight = np.inf

        for start_node in G.nodes:
            graph = G.copy()
            stack = []
            cycle = []
            current_node = start_node

            while True:
                neighbors = graph.neighbors(current_node)
                try:
                    next_node = next(neighbors)
                    stack.append(current_node)
                    graph.remove_edge(next_node, current_node)
                    current_node = next_node
                except StopIteration:
                    cycle.append(current_node)
                    if len(stack) == 0:
                        break
                    current_node = stack.pop()
            nodes = set()
            path = []
            for node in cycle:
                if node not in nodes:
                    nodes.add(node)
                    path.append(node)
            path += [path[0]]
            path_weight = weight_of_path(path, matrix)
            if path_weight < best_weight:
                best_weight = path_weight
                cristofides_path = path
        cristofides_path = cristofides_path [:-1]  
        return cristofides_path     
    else:
        raise NotImplementedError
        
        

        
class unweighted_Graph:
    def __init__(self):

        # default dictionary to store graph
        self.graph = defaultdict(list)
        self.dfs_traverse = []
    # function to add an edge to graph
    def addEdge(self, u, v):
        self.graph[u].append(v)
        self.graph[v].append(u)

    # A function used by DFS
    def DFSUtil(self, v, visited):

        # Mark the current node as visited
        # and print it
        visited.add(v)
        self.dfs_traverse.append(v)
#         print(v, end=' ')

        # Recur for all the vertices
        # adjacent to this vertex
        for neighbour in self.graph[v]:
            if neighbour not in visited:
                self.DFSUtil(neighbour, visited)

    # The function to do DFS traversal. It uses
    # recursive DFSUtil()
    def DFS(self, v):
        self.dfs_traverse = []
        # Create a set to store visited vertices
        visited = set()

        # Call the recursive helper function
        # to print DFS traversal
        self.DFSUtil(v, visited)
        
        
class Graph:
    def __init__(self, vertices):
        self.V = vertices  # No. of vertices
        self.graph = []  # default dictionary
        # to store graph
 
    # function to add an edge to graph
    def addEdge(self, u, v, w):
        self.graph.append([u, v, w])
 
    # A utility function to find set of an element i
    # (uses path compression technique)
    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])
 
    # A function that does union of two sets of x and y
    # (uses union by rank)
    def union(self, parent, rank, x, y):
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)
 
        # Attach smaller rank tree under root of
        # high rank tree (Union by Rank)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
 
        # If ranks are same, then make one as root
        # and increment its rank by one
        else:
            parent[yroot] = xroot
            rank[xroot] += 1
 
    # The main function to construct MST using Kruskal's
        # algorithm
    def KruskalMST(self):
 
        result = []  # This will store the resultant MST
         
        # An index variable, used for sorted edges
        i = 0
         
        # An index variable, used for result[]
        e = 0
 
        # Step 1:  Sort all the edges in
        # non-decreasing order of their
        # weight.  If we are not allowed to change the
        # given graph, we can create a copy of graph
        self.graph = sorted(self.graph,
                            key=lambda item: item[2])
 
        parent = []
        rank = []
 
        # Create V subsets with single elements
        for node in range(self.V):
            parent.append(node)
            rank.append(0)
 
        # Number of edges to be taken is equal to V-1
        while e < self.V - 1:
 
            # Step 2: Pick the smallest edge and increment
            # the index for next iteration
            u, v, w = self.graph[i]
            i = i + 1
            x = self.find(parent, u)
            y = self.find(parent, v)
 
            # If including this edge does't
            #  cause cycle, include it in result
            #  and increment the indexof result
            # for next edge
            if x != y:
                e = e + 1
                result.append([u, v, w])
                self.union(parent, rank, x, y)
            # Else discard the edge
        
#         minimumCost = 0
#         print ("Edges in the constructed MST")
#         for u, v, weight in result:
#             minimumCost += weight
#             print("%d -- %d == %d" % (u, v, weight))
#             print("Minimum Spanning Tree" , minimumCost)
        return result
        # A function used by DFS

    
def extract_edges(path):
    return ((u,v) for u,v in zip(path[:-1], path[1:]))


def isInTouch(edge1, edge2):
    u1, v1 = edge1
    u2, v2 = edge2
    if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
        return True
    return False

def swap_two_edges(edge1,edge2, path, augmented_adjacency_matrix, old_augmented_weight, adjacency_matrix, old_weight, swap = True):
    new_path = deepcopy(path)
    (u1, v1), (u2, v2) = (edge1,edge2)   
    
    if len(np.unique([u1, v1, u2, v2])) != 4:
        # print(f'repeated values ({u1}, {v1}) ({u2}, {v2})')
        return path, old_weight, old_augmented_weight
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
    return new_path, old_weight - adjacency_matrix[u1][v1] - adjacency_matrix[u2][v2] + adjacency_matrix[u1][u2] + adjacency_matrix[v1][v2], \
                     old_augmented_weight - augmented_adjacency_matrix[u1][v1] - augmented_adjacency_matrix[u2][v2] + augmented_adjacency_matrix[u1][u2] + augmented_adjacency_matrix[v1][v2]

def swap_two_edges_by_index(edge1,edge2, path, adjacency_matrix, old_weight):
    new_path = deepcopy(path)
    (index_u1, index_v1), (index_u2, index_v2) = (edge1,edge2)   
    u1 = path[index_u1]
    u2 = path[index_u2]
    v1 = path[index_v1]
    v2 = path[index_v2]    
    if index_v1 == 0: 
        index_v1 = len(path) - 1
    if index_u1 > index_v1:
        v1, u1 = u1, v1
        index_v1, index_u1 = index_u1, index_v1
    index_u1 += 1
    
    if index_v2 == 0: 
        index_v2 = len(path) - 1
    if index_u2 > index_v2:
        v2, u2 = u2, v2
        index_v2, index_u2 = index_u2, index_v2
    index_u2 += 1
    
    if index_u2 > index_u1:
        new_path[index_v1 : index_u2] = new_path[::-1][len(path) - index_u2 : len(path) - index_v1]
    else:
        new_path[index_v2 : index_u1] = new_path[::-1][len(path) - index_u1 : len(path) - index_v2]
    return new_path, old_weight - adjacency_matrix[u1][v1] - adjacency_matrix[u2][v2] + adjacency_matrix[u1][u2] + adjacency_matrix[v1][v2]


def is_crossed_1(edge1, edge2, dict_points):

    # print('Check intersection')
    u1, v1 = edge1
    u2, v2 = edge2
    if v1 == u2 or v1 == v2 or u1 == u2 or u1 == v2:
        return False
    # print((u1, v1), (u2, v2))
    u1 = (dict_points[u1])
    v1 = (dict_points[v1])
    u2 = (dict_points[u2])
    v2 = (dict_points[v2])
    # print((u1, v1), (u2, v2))
    # line1 = LineString([(u1.x, u1.y), (v1.x, v1.y)])
    # line2 = LineString([(u2.x, u2.y), (v2.x, v2.y)])
    # print(line1.crosses(line2))
    

    
    
    # return line1.intersects(line2)
    return doIntersect(u1, v1, u2, v2)


def restore_path(edges, mode='double'):
    path = []
    N = len(edges)
    if mode == 'single':
        current_node = 0
        path = [0]
        while len(path) < N :
            current_node  = edges[current_node]
            path.append(current_node)
    elif mode == 'double':
        node = 0 
        next_node = edges[node][0]
        path.append(node)
        while len(path) < N:
            prev_node = node
            node = next_node
            path.append(node)
            next_node = edges[node][1 - edges[node].index(prev_node)]
    else:
        raise NotImplementedError
    return path + [path[0]]

def possible_combinations(edges):
    edge1, edge2 = edges
    u1, v1 = edge1
    u2, v2 = edge2
    return ((u1, u2), (v1, v2)), ((u1,v2), (u2, v1))

def check_completeness(path):
    return len(set(path)) == len(path) - 1


def weight_of_path(path, matrix):
    return sum(( matrix[u][v] for u, v in zip(path[:-1], path[1:])))
