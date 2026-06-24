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

def create_skew_mtx(rows, cols, density, s=1.5):
    rows = int(rows)
    cols = int(cols)
    num_el = int(rows*cols*density)
    
    #equal distribution 
    rng = np.random.default_rng()
    row_ind = rng.integers(0, rows ,size=num_el)
    col_ind = rng.integers(0, cols, size=num_el)
    data = rng.uniform(0, 5, size=num_el)    

    sparse_matrix = coo_matrix((data, (row_ind, col_ind)), shape=(rows, cols))
    mmwrite("/home/fakeheadset/Projects/EulerEasel/Data/synthetic/uniform.mtx", sparse_matrix)
    
    #skewed    
    row_ind_skew = zipfian.rvs(s, n=rows, size=num_el, random_state=rng) -1
    col_ind_skew = rng.integers(0, cols, num_el)
    data_skew = rng.uniform(1.0, 5.0, num_el)
    
    sparse_matrix_skew = coo_matrix((data_skew, (row_ind_skew, col_ind_skew)), shape=(rows, cols))
    mmwrite("/home/fakeheadset/Projects/EulerEasel/Data/synthetic/skewed.mtx", sparse_matrix_skew)


if __name__=="__main__":
    create_skew_mtx(10, 10, 0.1,1.5)