import numpy as np
import sys
import os
import sys
from pathlib import Path
import json
sys.path.insert(0, "/home/fakeheadset/Projects/EulerEasel/Src/include")

import matrix_extractor as me
import CUDAruntime as crn
import runtime as rn
from context import LazyFrozenContext
from registry import launch_spmv

print(me)
'''
#343fff
this is the implementation of the linear Thompson sampling algorithm for contextual bandits
weights of the important features. linTS process the options sequentially and updates the model 
after each option is processed.
'''

import numpy as np

class LinTS:
    def __init__(self, kernels, num_features):
        """
        kernels: List of me.Kernel enum objects
        num_features: Length of your matrix feature vector (new_data)
        """
        self.models = {}
        if kernels is not None and num_features is not None:
        # Pre-initialize matrices for every unique C++ Enum kernel
            for k in kernels:
                self.models[k] = {
                    'A': np.identity(num_features, dtype=np.float64), 
                    'b': np.zeros((num_features, 1), dtype=np.float64)
                }

    def choose_kernel(self, active_kernels, x):
        """
        active_kernels: The subset of kernels available on this hardware
        x: Flattened numpy feature vector
        """
        scores = {}
        x = x.flatten()
        
        for k in active_kernels:
            model = self.models[k]
            # Fast, numerically stable way to compute theta
            theta = np.linalg.solve(model['A'], model['b']).flatten()
            
            # Efficient sampling using Cholesky decomposition instead of full inverse
            try:
                L = np.linalg.cholesky(np.linalg.inv(model['A']))
                theta_sample = theta + L @ np.random.standard_normal(len(theta))
            except np.linalg.LinAlgError:
                # Fallback if matrix is temporarily not positive-definite
                theta_sample = np.random.multivariate_normal(theta, np.linalg.inv(model['A']))
                
            scores[k] = np.dot(x, theta_sample)
            
        # Return the actual C++ Enum object of the winner
        best_kernel = max(scores, key=scores.get)
        return best_kernel, scores[best_kernel]

    def update(self, kernel, x, reward):
        """Updates the linear model for the executed kernel."""
        model = self.models[kernel]
        x = x.reshape(-1, 1)
        model['A'] += x @ x.T
        model['b'] += x * reward
        
    def get_best_kernel(self, kernels, x):
        """
        Performs pure exploitation (no random sampling) to find the 
        best kernel based on learned historical rewards.
        """
        expected_rewards = {}
        x = x.flatten()
        
        for k in kernels:
            model = self.models[k]
            # Solve for theta directly: A * theta = b -> theta = inv(A) * b
            theta = np.linalg.solve(model['A'], model['b']).flatten()
            
            # Pure deterministic dot product (no random multivariate normal)
            expected_rewards[k] = np.dot(x, theta)
            
        # Sort kernels from highest reward (fastest) to lowest reward (slowest)
        ranked_kernels = sorted(expected_rewards.items(), key=lambda item: item[1], reverse=True)
        return ranked_kernels
    
    def save(self, filepath:str):
        save_dict= {}
        for kernel_enum, matrices in self.models.items():
            kernel_name = kernel_enum.name
            save_dict[f'model_{kernel_name}_A'] = matrices['A']
            save_dict[f'model_{kernel_name}_b'] = matrices['b']
            
        np.savez_compressed(filepath, **save_dict)
        print(f" Successfully saved models to {filepath}")
        
    def load(self, filepath:str, enum_class):
        self.models = {}
        with np.load(filepath) as data:
            for keys in data.files:
                if keys.startWith("model_"):
                    _, kernel, matrix = keys.split("_")
                    kernel_enum = enum_class[kernel]
                    if kernel_enum not in self.models:
                        self.models[kernel_enum] = {}
                    
                    self.models[kernel_enum][matrix] = data[keys]
                    
        print(f" Successfully loaded models from {filepath}")


if __name__ == "__main__":
    folder_path = Path("/home/fakeheadset/Projects/EulerEasel/Data/datasetnaked/")
    items = [str(item) for item in folder_path.iterdir()] 
    str_reg = me.StrategyRegister()
    hrd = me.HardwareContext()
    strategies = str_reg.get_strategies(hrd)
    kernels = [s.kernel for s in strategies]
    lnts = LinTS(kernels=kernels, num_features=14)
    oracle = {f'oracle_{idx}' : {} for idx in range(len(items))}
    with open('ground_truth.json', 'r') as f:
        data = json.load(f)
    runtime_oracle = data        

    for idx, filename in enumerate(items):
        [r, c, nnz] = me.mat_dim(filename)
        frozen_context = LazyFrozenContext(filename, r, c, nnz)
        temp_csr = frozen_context.ensure_cpu_csr_matrix()
        extractor = me.MatrixExtractor(temp_csr)
        new_data = np.array(extractor.to_flat_vector(extractor.extract_all()), dtype=np.float64)
        best_kernel_enum, score= lnts.choose_kernel(kernels, new_data)
        res = launch_spmv(best_kernel_enum, frozen_context)
        reward = 0
        if isinstance(res,(tuple, list)):
            y, runtime = res
        else:
            runtime = res 
        for i, item in enumerate(runtime_oracle):
            inner_dict = next(iter(item.values()))
            ground_truth_runtime = inner_dict['runtime']
            reward = runtime / ground_truth_runtime
        lnts.update(best_kernel_enum, new_data, reward)
        

    
    lnts.save('/home/fakeheadset/Projects/EulerEasel/Src/include/Model/context.npz')
        
        
            


                    
            
            
        
        


    
    
    
    
    
    
