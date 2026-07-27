import numpy as np
import sys
import os
sys.path.append("/home/fakeheadset/Projects/EulerEasel/Src/include")
import matrix_extractor
import runtime as rn
import CUDAruntime as crn

'''
#343fff
this is the implementation of the linear Thompson sampling algorithm for contextual bandits
weights of the important features. linTS process the options sequentially and updates the model 
after each option is processed.
'''

class linTS:
    def __init__(self, kernel, num_features):
        self.models = {}
        numkernels = len(kernel)
        for k in range(numkernels):
            self.models[k] = {'A': np.identity(num_features), 'b': np.zeros((num_features, 1))}
                


    def choose_kernel(self, kernels, x):
        '''
        x: Context vector of shape (1, num_features) or (num_features,)
        reward: Scalar value
        '''
        scores = {}
        x = x.flatten();
        for k in kernels:
            model = self.models[k]
            theta = np.linalg.solve(model['A'], model['b'])
            cov = np.linalg.inv(model['A'])
            theta_sample = np.random.multivariate_normal(theta.flatten(), cov)
            scores[k] = np.dot(x, theta_sample)
            
        best_kernel = max(scores, key=scores.get)
        return best_kernel, scores[best_kernel]

    def update(self,kernel, x, reward):
        model = self.models[int(kernel)]
        x = x.reshape(-1, 1)
        model['A'] += x @ x.T
        model['b'] += x * reward



if __name__ == "__main__":
    filename= "/home/fakeheadset/Projects/EulerEasel/Data/bcsstk13.mtx"
    ell= matrix_extractor.ELL()
    mat_vec = matrix_extractor.file_paser(filename)
    [r,c,nnz] = matrix_extractor.mat_dim(filename)
    [A, J] = matrix_extractor.ellformat(mat_vec, r, c, nnz,ell)    
    input = np.random.rand(c)
    A_np = np.asarray(A, dtype=np.float64)
    J_np = np.asarray(J, dtype=np.int32)
    # d_vals = crn.deviceBufferDouble(len(csr.vals))
    # d_rptr = crn.deviceBufferInt(len(csr.rptr))
    # d_ind = crn.deviceBufferInt(len(csr.ind))
    
    # d_rptr.h2d(np.asarray(csr.rptr, dtype=np.int32))
    # d_ind.h2d(np.asarray(csr.ind, dtype=np.int32))
    # d_vals.h2d(np.asarray(csr.vals, dtype=np.float64))   
    row, col = A_np.shape
    [a, j] = crn.flatten(J_np, A_np)
    d_a = crn.deviceBufferDouble(len(a))
    d_j = crn.deviceBufferInt(len(j))
    d_a.h2d(np.asarray(a, dtype=np.float64))
    d_j.h2d(np.asarray(j, dtype=np.int32))
    # csr_rows = len(csr.rptr) -1
    d_x = crn.deviceBufferDouble(c)
    d_y = crn.deviceBufferDouble(r)
    d_x.h2d(input)
    d_y.h2d(np.zeros(r, dtype=np.float64))
    threads = 256
    blocks = (r + threads -1)//threads
    # stream1 = crn.cudaStream(True)
    # stream2 = crn.cudaStream(True)
    kernel_runtime = crn.cuda_ell(
        threads,
        blocks,
        d_a, d_j,
        row, col,
        d_x,
        d_y, 
    )

    y_gpu = d_y.d2h()
    print(len(y_gpu))
    print(kernel_runtime)