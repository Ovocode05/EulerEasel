import numpy as np
import sys
import os
sys.path.append("/home/fakeheadset/Projects/EulerEasel/Src/include")
import matrix_extractor
import runtime as rn

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
    ell = matrix_extractor.ELL()
    mat_vec = matrix_extractor.file_paser(filename)
    [r,c,nnz] = matrix_extractor.mat_dim(filename)
    [A, J]= matrix_extractor.ellformat(mat_vec, r, c, nnz, ell)     
    input = np.random.rand(c)
    [output, reward] = rn.proc_ellx4(input, A, J)
    print(reward, "nansec")
    print(len(output))