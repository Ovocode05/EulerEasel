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
from linTS import LinTS


if __name__ == "__main__":
    str_reg = me.StrategyRegister()
    hrd = me.HardwareContext()
    strategies = str_reg.get_strategies(hrd)
    kernels = [s.kernel for s in strategies]
    lnts = LinTS(kernels=kernels, num_features=14)
    filename = '/home/fakeheadset/Projects/EulerEasel/Data/datasetnaked/uni_chimera_i5.mtx'
    lnts.load(filename,)
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
        

    
        
        
            


                    
            
            
        
        


    
    
    
    
    
    
