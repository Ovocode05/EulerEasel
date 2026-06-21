import numpy as np
from Src.phase1.csr_slice import compute_gini_numpy

'''
the ideas behind is the sanity check of the creating a normal equal graph and other creating a very skewed graph
using zipf distribution with a tunable exponent - requency of an item is inversely proportional to its rank in a 
frequency table - then checking the gini index of the both.
Desired results: 
Equal Distribution : [0., 0., 0., ........ 0.]
Mostly Skewed Distribution : [1., ....... , 1.]
'''

#creating a sequence from zipf distribution 
def zipf_dist(size, a=2.0):
    rng = np.random.default_rng()
    samples = rng.zipf(a, size)
    samples = samples.flatten()
    
    return samples  

#343fff skewed 
# samples = zipf_dist((16,1), 2.5)
# samples_gini = compute_gini_numpy(samples)

#343fff equal 
# samples_eq = zipf_dist((16,1), 1.2)
# samples = compute_gini_numpy(samples_eq)

