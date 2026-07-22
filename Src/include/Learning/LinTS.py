import numpy as np
import sys
import os
import numpy as np

# Inject the build artifact path so Python can find your .so files
sys.path.append("/home/fakeheadset/Projects/EulerEasel/Src/include")
import matrix_extractor


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
    filename= "/home/fakeheadset/Projects/EulerEasel/Data/bcsstk18.mtx"
    csr = matrix_extractor.CSR()
    mat_vec = matrix_extractor.file_paser(filename)
    [r,c,nnz] = matrix_extractor.mat_dim(filename)
    
    matrix_extractor.csrformat(mat_vec, r,c,nnz, csr) 
    csr_matrix_obj = matrix_extractor.CsrMatrix(csr, c)
    print(f"3. CsrMatrix instance built. Shape: ({csr_matrix_obj.get_nrow()}, {csr_matrix_obj.get_ncol()})")
    extractor = matrix_extractor.MatrixExtractor(csr_matrix_obj)
    vector = extractor.to_flat_vector(extractor.extract_all())
    print(vector)
    
    #kernels
    hrd = matrix_extractor.HardwareContext();
    str = matrix_extractor.StrategyRegister()
    kernel_vec = matrix_extractor.StrategyRegister.create()
    str.generate(kernel_vec, hrd);
    final = str.get_strategies(hrd)
    final2 = str.get_name(final);
    print(len(final2))
    
    #normalize the vector
    
    
    
    num_features = 2
    first = linTS([0, 1], num_features)
    for i in range(10):
        x = np.random.rand(num_features) # (numfeatures,)
        kernel, score = first.choose_kernel([0, 1], x)
        reward = np.random.rand()
        first.update(kernel, x, reward)
        