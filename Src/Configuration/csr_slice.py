from scipy.io import mmread
import numpy as np

coo_mat = mmread('/home/fakeheadset/Projects/EulerEasel/Data/bcsstk18.mtx', spmatrix = True) 
csr_mat = coo_mat.tocsr()
total_rows = csr_mat.shape[0]
global_row_degree = csr_mat.getnnz(axis=1) #get all the nnz for each row
# print(csr_mat.sum(axis=1))

#343fff----------------------------------------------

def compute_gini_numpy(x):
    '''
    GIni coeff. is computed on a to check on the impurity or irregularity in the dataset.
    computing Gini coeff. using numpy and for every chunk of the CSR
    '''
    
    n = len(x)
    if n==0 or np.sum(x)==0:
        return 0.0
    
    x_sorted = np.sort(x)
    idx = np.arange(1, n+1)
    numerator = 2 * np.sum(idx * x_sorted)
    denominator = n * np.sum(x_sorted)
    
    gini_coeff = (numerator / denominator) - ((n + 1) / n)
    return gini_coeff

T:int = 512
T_chunks : int = np.floor(csr_mat.shape[0]/T) #no 291
results= []

##343fff ---------------------------------------------

for i in range(0, total_rows, T):
    
    chunk_degrees = global_row_degree[i: i+T] #getting the row nnz chuck by chunk
    chunk_gini = compute_gini_numpy(chunk_degrees) 
    chunk_total_nnz = np.sum(chunk_degrees)  #per chunk
    
    results.append({
        "chunk_idx" : i//T,
        "total_nnz" : chunk_total_nnz,
        "gini": chunk_gini
    })
    
