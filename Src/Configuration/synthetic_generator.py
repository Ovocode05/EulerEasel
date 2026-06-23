import numpy as np
from scipy.io import mmwrite 
from scipy.sparse import coo_matrix 
from scipy.stats import zipfian

'''
the ideas behind is the sanity check of the creating a normal equal graph and other creating a very skewed graph
using zipf distribution with a tunable exponent - requency of an item is inversely proportional to its rank in a 
frequency table - then checking the gini index of the both.
Desired results: 
Equal Distribution : [0., 0., 0., ........ 0.]
Mostly Skewed Distribution : [1., ....... , 1.]
'''

def create_skew_mtx(filename, rows, cols, density, s=1.5):
    rows = int(rows)
    cols = int(cols)
    num_el = int(rows*cols*density)
    
    rng = np.random.default_rng()
    row_ind = rng.integers(0, rows, size=num_el)
    col_ind = rng.integers(0, cols, size=num_el)
    
    data = zipfian.rvs(s, n=1000, size=num_el, random_state=rng)
    
    sparse_matrix = coo_matrix((data, (row_ind, col_ind)), shape=(rows, cols))
    mmwrite(filename, sparse_matrix)
    print(f"Successfully saved {filename} ({rows}x{cols}, nnz={sparse_matrix.nnz})")
    

create_skew_mtx("/home/fakeheadset/Projects/EulerEasel/Data/synthetic/equal_dist.mtx", rows=100, cols=100, density=0.01, s=1.01)
