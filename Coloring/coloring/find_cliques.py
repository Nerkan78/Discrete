import os
import networkx as nx
listdir = os.listdir('data')
for filename in listdir :
    print(filename)
    with open('data/'+filename, 'r') as f:
        input_data = f.read()
        lines = input_data.split('\n')

        first_line = lines[0].split()
        node_count = int(first_line[0])
        edge_count = int(first_line[1])
        density = 2 * edge_count / (node_count) / (node_count-1)
        if density > 0.3 and node_count == 1000:
            continue
        edges = []
        for i in range(1, edge_count + 1):
            line = lines[i]
            parts = line.split()
            edges.append((int(parts[0]), int(parts[1])))

    graph = nx.Graph(edges)
    K = nx.graph_clique_number(graph)
    print(K)
    with open('data/K_'+filename, 'w') as f:
        f.write(f'{node_count} {edge_count}\n')
        f.write(str(K))

