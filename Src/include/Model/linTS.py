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

            # Cholesky factor of A (A = L L^T)
            L = np.linalg.cholesky(model['A'])

            # Sample z ~ N(0, I)
            z = np.random.standard_normal(len(theta))

            # noise ~ N(0, A^{-1})
            noise = np.linalg.solve(L.T, z)

            # Thompson sample
            theta_sample = theta + noise

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
            save_dict[f'model__{kernel_name}__A'] = matrices['A']
            save_dict[f'model__{kernel_name}__b'] = matrices['b']
            
        np.savez_compressed(filepath, **save_dict)
        print(f" Successfully saved models to {filepath}")
        
    def load(self, filepath: str, enum_class):
        self.models = {}
        with np.load(filepath) as data:
            for keys in data.files:
                if keys.startswith("model_"):
                    parts = keys.split("__")
                    kernel = parts[1]
                    matrix = parts[2]

                    # FIX: Use getattr() instead of square brackets for C++ Pybind11 Enums
                    kernel_enum = getattr(enum_class, kernel)
                    if kernel_enum not in self.models:
                        self.models[kernel_enum] = {}
                    
                    self.models[kernel_enum][matrix] = data[keys]
                    
        print(f" Successfully loaded models from {filepath}")
        
    
def generate_ground_truth_oracle():
        folder_path = Path("/home/fakeheadset/Projects/EulerEasel/Data/datasetnaked/")
        items = sorted([str(item) for item in folder_path.iterdir() if item.suffix == '.mtx']) 
        
        # 1. Discover all available C++ kernels from your strategy register
        str_reg = me.StrategyRegister()
        hrd = me.HardwareContext()
        strategies = str_reg.get_strategies(hrd)
        kernels = [s.kernel for s in strategies]
        
        oracle_records = []
        NUM_RUNS = 30
        
        print(f"Starting brute-force profiling across {len(items)} matrices and {len(kernels)} kernels...")
        
        # 2. Iterate through each sparse matrix
        for idx, filename in enumerate(items):
            matrix_name = os.path.basename(filename)
            print(f"\n[{idx + 1}/{len(items)}] Profiling matrix: {matrix_name}")
            
            # Build matrix context
            [r, c, nnz] = me.mat_dim(filename)
            frozen_context = LazyFrozenContext(filename, r, c, nnz)
            
            kernel_perf = {}
            
            # 3. Benchmark every single kernel on this specific matrix
            for k in kernels:
                runtimes = []
                
                # Warm-up run to eliminate GPU kernel compilation/allocation latency
                try:
                    launch_spmv(k, frozen_context)
                except Exception:
                    continue # Skip kernel if it fails or is incompatible with this matrix shape
                    
                # Run 30 timed iterations
                for _ in range(NUM_RUNS):
                    res = launch_spmv(k, frozen_context)
                    runtime = res[1] if isinstance(res, (tuple, list)) else res
                    runtimes.append(float(runtime))
                    
                if runtimes:
                    # Use median to strip out operating system background jitter noise
                    kernel_perf[k.name] = np.median(runtimes)
            
            if not kernel_perf:
                print(f"  Warning: No kernels successfully executed for {matrix_name}")
                continue
                
            # 4. Find the global best performing (fastest / minimum runtime) variant
            best_kernel_name = min(kernel_perf, key=kernel_perf.get)
            best_runtime = kernel_perf[best_kernel_name]
            
            print(f"  -> Winner: {best_kernel_name} | Median Runtime: {best_runtime:.4f} ms")
            
            # 5. Format structure to match the layout your main script expects
            matrix_entry = {
                matrix_name: {
                    "best_kernel": best_kernel_name,
                    "runtime": best_runtime,
                    "all_kernel_profiles": kernel_perf
                }
            }
            oracle_records.append(matrix_entry)
            
        # 6. Save records directly to JSON output
        output_path = 'ground_truth.json'
        with open(output_path, 'w') as f:
            json.dump(oracle_records, f, indent=4)
            
        print(f"\n Successfully generated and saved oracle file to: {os.path.abspath(output_path)}")




if __name__ == "__main__":
    # generate_ground_truth_oracle()
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
            reward = ground_truth_runtime/runtime
        lnts.update(best_kernel_enum, new_data, reward)
        

    
    lnts.save('/home/fakeheadset/Projects/EulerEasel/Src/include/Model/context.npz')
        
        
            


                    
            
            
        
        


    
    
    
    
    
    
