#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import time
from collections import namedtuple
Item = namedtuple("Item", ['index', 'value', 'weight'])

def estimations(items, taken_items, cur_value, cur_capacity):
#     cur_capacity = capacity - sum(map(lambda x: x.weight, taken_items))
    if cur_capacity < 0:
        # print('low capacity')
        return None, None
#     cur_value = sum(map(lambda x: x.value, taken_items))
    available_items = list(filter(lambda x: x.weight <= cur_capacity, items))
    if len(available_items) == 0:
        return cur_value, cur_value
#     available_items = sorted(available_items, key = lambda x : x.value / x.weight, reverse = True)
    
    # print(f'available_items is \n{available_items}')
    index_cut = 0
    weight_cumsum = 0
    value_cumsum = 0
    while True and index_cut < len(available_items):
        weight_cumsum += available_items[index_cut].weight
        if weight_cumsum > cur_capacity:
            weight_cumsum -= available_items[index_cut].weight
            break
        else:
            value_cumsum += available_items[index_cut].value
            index_cut += 1
            
    # print(f'weight_cumsum is {weight_cumsum} {weight_cumsum > cur_capacity}')
    # print(f'value_cumsum is {value_cumsum}')        
            
    high_estimation = float(value_cumsum) + cur_value
    if index_cut < len(available_items):
        free_weight = cur_capacity - weight_cumsum
        high_estimation += float(available_items[index_cut].value) * free_weight / available_items[index_cut].weight
        
        low_estimation = max(value_cumsum, available_items[index_cut].value) + cur_value
    else:
        low_estimation = value_cumsum + cur_value
    
#     weight_cumsum = np.cumsum(list(map(lambda x : x.weight, available_items)))
#     value_cumsum = np.cumsum(list(map(lambda x : x.value, available_items)))
    
    # print(f'weight_cumsum is {weight_cumsum} {weight_cumsum > cur_capacity}')
    # print(f'value_cumsum is {value_cumsum}')
    
#     index_cut = (weight_cumsum > cur_capacity).argmax()
        
    # print(f'index_cut is {index_cut}')
#     high_estimation = float(value_cumsum[index_cut-1])
#     free_weight = cur_capacity - weight_cumsum[index_cut-1]
#     high_estimation += float(available_items[index_cut].value) * free_weight / available_items[index_cut].weight + cur_value
    
#     low_estimation = max(value_cumsum[index_cut-1], available_items[index_cut].value) + cur_value
    
    
    return low_estimation, high_estimation
    
    
    

class Node:
    def __init__(self, parent, items, item_index, taken_items, value, cur_capacity):
        self.parent = parent
        self.item_index = item_index
        self.taken_items = taken_items
        self.value = value #sum(map(lambda x: x.value, taken_items))
        self.cur_capacity = cur_capacity
        self.low_estimation, self.high_estimation = estimations(items, taken_items, value, cur_capacity)
        
        

def solve_it(input_data):
    # Modify this code to run your optimization algorithm
    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
#     global capacity
    capacity = int(firstLine[1])
    # print(f'capacity is {capacity}')
    
    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))

    items = sorted(items, key = lambda x : x.value / x.weight, reverse = True)
    # a trivial algorithm for filling the knapsack
    # it takes items in-order until the knapsack is full
    value = 0
    weight = 0
    taken = [0]*len(items)
    
    start_time = time.time()
    root_node = Node(None, items, 0, [], 0, capacity)
    # print(f' low estimation is {root_node.low_estimation} high estimation is {root_node.high_estimation}')
    result_node = root_node
    
    open_nodes = []
    open_nodes.append(root_node)
    
    
    global_low_estimation = root_node.low_estimation
    global_value = 0
    while len(open_nodes) > 0:

        # print(global_low_estimation)
        current_node = open_nodes.pop(0)
        item_index = current_node.item_index
        # print(f' item_index is {item_index}')
        # print(f' taken items are {" ".join(map(str, current_node.taken_items))}')
        # print(f' current value is {current_node.value}')
        # print(f' current vapavity is {current_node.cur_capacity}')
        # print(f' low estimation is {current_node.low_estimation} high estimation is {current_node.high_estimation}')
        if current_node.high_estimation < global_low_estimation:
            # print('skip')
            continue

        
        if item_index < len(items) - 1:
            left_child = Node(current_node, items[item_index+1:], item_index + 1, current_node.taken_items, 
                              current_node.value, current_node.cur_capacity )
            right_child = Node(current_node, items[item_index+1:], item_index + 1, current_node.taken_items + [items[item_index]], 
                              current_node.value + items[item_index].value, current_node.cur_capacity - items[item_index].weight)
                              
                              
            
            if left_child.low_estimation is not None:
                global_low_estimation = max(global_low_estimation, left_child.low_estimation)
            if right_child.low_estimation is not None:
                global_low_estimation = max(global_low_estimation, right_child.low_estimation)
            
            if left_child.high_estimation is not None and left_child.high_estimation >= max(global_low_estimation, current_node.value * 1.0):
                open_nodes.append(left_child)
            
            if right_child.high_estimation is not None and right_child.high_estimation >= max(global_low_estimation, current_node.value * 1.0):
                open_nodes.append(right_child)
        elif item_index == len(items) - 1:
            left_child = Node(current_node, [], item_index + 1, current_node.taken_items, current_node.value, current_node.cur_capacity )
            right_child = Node(current_node, [], item_index + 1, current_node.taken_items + [items[item_index]],
                               current_node.value + items[item_index].value, current_node.cur_capacity - items[item_index].weight )
            
            if left_child.low_estimation is not None:
                global_low_estimation = max(global_low_estimation, left_child.low_estimation)
            if right_child.low_estimation is not None:
                global_low_estimation = max(global_low_estimation, right_child.low_estimation)
            
            if left_child.high_estimation is not None and left_child.high_estimation >= global_low_estimation:
                open_nodes.append(left_child)
            
            if right_child.high_estimation is not None and right_child.high_estimation >= global_low_estimation:
                open_nodes.append(right_child)
        
        if current_node.value > global_value:
            global_value = current_node.value
            result_node = current_node
            
        # print(f' right_child item_index is {right_child.item_index}')
        # print(f' right_child taken items are {" ".join(map(str, right_child.taken_items))}')
        # print(f' right_childlow estimation is {right_child.low_estimation} high estimation is {right_child.high_estimation}')
        
        
        
        
        # if weight + item.weight <= capacity:
            # taken[item.index] = 1
            # value += item.value
            # weight += item.weight
    
    # prepare the solution in the specified output format
    for item in result_node.taken_items:
        taken[item.index] = 1
    end_time = time.time()
    duration = end_time - start_time
    output_data = str(result_node.value) + ' ' + str(1) + '\n'
    output_data += ' '.join(map(str, taken))
    output_data += '\n' + str(duration)
    
    # output_data  = '\n'.join(map(str, items))
    return output_data


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

