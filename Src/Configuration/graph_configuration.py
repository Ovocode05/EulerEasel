import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from csr_slice import compute_gini_numpy

'''
this script is for new graph configuration, to test on skewed and equally distributed graphs
creating the graph that matches the zipf sequence and then we can change it to csr format
this creates unweighted graphs with values either 1 showing that the link exits or 0 it's opposite.
'''

def graph_configuration(num_nodes, exp):
    rng = np.random.default_rng()
    while True:
        degrees = rng.zipf(exp, num_nodes)
        degrees = np.clip(degrees, 1, num_nodes-1)
        
        if np.sum(degrees)%2==0:
            break
        
    deg_list = [int(d) for d in degrees]
    multigraph = nx.configuration_model(deg_list)
    simplegraph = nx.Graph(multigraph)
    simplegraph.remove_edges_from(nx.selfloop_edges(simplegraph))
    
    csr_matrix = nx.to_scipy_sparse_array(simplegraph, format='csr')
    local_degree = np.diff(csr_matrix.indptr) #this for counting the nnz
    return csr_matrix, local_degree, simplegraph


#343fff-----------------------------------

def draw_unweighted_graph(num_nodes, exp, path=""):
    _, _, simp = graph_configuration(num_nodes, exp)
    subax1 = plt.subplot(121)
    nx.draw(simp, with_labels=True)
    plt.show()
    
    if path!="":
        plt.savefig(path)




