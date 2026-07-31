import numpy as np
import sys
sys.path.insert(0, "/home/fakeheadset/Projects/EulerEasel/Src/include")

class LazyFrozenContext:
    def __init__(self, filename, r, c, nnz):
        self.filename = filename
        self.r = r
        self.c = c
        self.nnz = nnz
        
        # Share standard input vector across all variants
        self.x = np.random.rand(c).astype(np.float64)
        
        # File parsing handle (loaded lazily on demand)
        self._mat_vec = None

        # --- Lazy CPU Formats (Cached after first generation) ---
        self.csr = None
        self.A_np = None  # ELL values
        self.J_np = None  # ELL column indices
        self.hyb=None

        # --- Lazy GPU Buffers (Cached after first allocation) ---
        self.threads = 256
        self.blocks = (r + 255) // 256
        
        # GPU Shared Vectors
        self.d_x = None
        self.d_y = None
        
        # GPU CSR Specific
        self.d_rptr = None
        self.d_ind = None
        self.d_vals = None
        
        # GPU ELL Specific
        self.d_a = None
        self.d_j = None
        self.ell_rows = 0
        self.ell_cols = 0

    @property
    def mat_vec(self):
        """Lazily parses the file only when a layout builder needs it."""
        if self._mat_vec == None:
            import matrix_extractor as me
            print(f"[System] Lazily parsing source file: {self.filename}")
            self._mat_vec = me.file_paser(self.filename)
        return self._mat_vec

    def ensure_cpu_csr(self):
        if not hasattr(self, 'raw_full_csr') or self.raw_full_csr is None:           
            import matrix_extractor as me
            print("[System] Lazily generating Standalone Full CPU CSR format...")
            self.raw_full_csr = me.CSR()
            me.csrformat(self.mat_vec, self.r, self.c, self.nnz, self.raw_full_csr)
        return self.raw_full_csr  
    
    def ensure_cpu_csr_matrix(self):
        # Use a distinct attribute name for the Extractor's wrapper
        if not hasattr(self, 'extractor_csr_matrix') or self.extractor_csr_matrix is None:
            import matrix_extractor as me
            print("[System] Lazily generating CPU CsrMatrix for feature extraction layer...")
            raw_csr = me.CSR()
            me.csrformat(self.mat_vec, self.r, self.c, self.nnz, raw_csr)
            self.extractor_csr_matrix = me.CsrMatrix(raw_csr, self.c)
        return self.extractor_csr_matrix


    def ensure_cpu_ell(self):
        if self.A_np is None:
            import matrix_extractor as me
            print("[System] Lazily generating CPU ELL format...")
            ell_obj = me.ELL()
            A, J = me.ellformat(self.mat_vec, self.r, self.c, self.nnz, ell_obj)
            self.A_np = np.asarray(A, dtype=np.float64)
            self.J_np = np.asarray(J, dtype=np.int32)
        return self.A_np, self.J_np
    
    # def ensure_cpu_hyb(self):
    #     # FIX: Check self.hyb instead of self.csr to avoid cross-contamination
    #     if self.hyb is None:
    #         import matrix_extractor as me
    #         print("[System] Lazily generating CPU HYB format...")
    #         self.hyb = me.HYB()
            
    #         # Execute the C++ partitioning split
    #         A, J, raw_csr = me.hybd_format(self.hyb, self.mat_vec, self.r, self.c, self.nnz)
            
    #         # Populate fields safely
    #         self.csr = raw_csr
    #         self.A_np = np.asarray(A, dtype=np.float64)
    #         self.J_np = np.asarray(J, dtype=np.int32)
            
    #     return self.hyb, self.A_np, self.J_np, self.csr

    
    
    def ensure_gpu_vectors(self):
        """Allocates tracking vectors on the GPU device."""
        if self.d_x is None:
            import CUDAruntime as crn
            print("[System] Lazily allocating GPU Input/Output vectors...")
            self.d_x = crn.deviceBufferDouble(self.c)
            self.d_y = crn.deviceBufferDouble(self.r)
            self.d_x.h2d(self.x)
            self.d_y.h2d(np.zeros(self.r, dtype=np.float64))

    def ensure_gpu_csr(self):
        if self.d_rptr is None:
            import CUDAruntime as crn
            import numpy as np
            self.ensure_cpu_csr() 
            self.ensure_gpu_vectors()
            
            print("[System] Lazily allocating and uploading GPU CSR structures...")
            # Target raw_full_csr safely!
            self.d_rptr = crn.deviceBufferInt(len(self.raw_full_csr.rptr))
            self.d_ind = crn.deviceBufferInt(len(self.raw_full_csr.ind))
            self.d_vals = crn.deviceBufferDouble(len(self.raw_full_csr.vals))
            
            self.d_rptr.h2d(np.asarray(self.raw_full_csr.rptr, dtype=np.int32))
            self.d_ind.h2d(np.asarray(self.raw_full_csr.ind, dtype=np.int32))
            self.d_vals.h2d(np.asarray(self.raw_full_csr.vals, dtype=np.float64))


    def ensure_gpu_ell(self):
        if self.d_a is None:
            import CUDAruntime as crn
            # 1. Ensure underlying CPU matrices are built
            A_np, J_np = self.ensure_cpu_ell()
            self.ensure_gpu_vectors()
            
            print("[System] Lazily converting, flattening and uploading GPU ELL structures...")
            self.ell_rows, self.ell_cols = A_np.shape
            a_flat, j_flat = crn.flatten(J_np, A_np)
            
            self.d_a = crn.deviceBufferDouble(len(a_flat))
            self.d_j = crn.deviceBufferInt(len(j_flat))
            self.d_a.h2d(np.asarray(a_flat, dtype=np.float64))
            self.d_j.h2d(np.asarray(j_flat, dtype=np.int32))
            
    # def ensure_gpu_hyb(self):
    #     if self.d_a is None or self.d_rptr is None:
    #         import CUDAruntime as crn
    #         import numpy as np
            
    #         # This will now run perfectly because ensure_cpu_hyb won't skip anymore!
    #         self.hyb, A_np, J_np, raw_csr_out = self.ensure_cpu_hyb()
    #         self.raw_csr = raw_csr_out  
    #         self.ensure_gpu_vectors()
            
    #         print("[System] Lazily converting, flattening and uploading GPU Hyb structures...")
            
    #         # Extraction and allocations follow smoothly...
    #         self.ell_rows, self.ell_cols = A_np.shape
    #         a_flat, j_flat = crn.flatten(J_np, A_np)
            
    #         self.d_a = crn.deviceBufferDouble(len(a_flat))
    #         self.d_j = crn.deviceBufferInt(len(j_flat))
    #         self.d_a.h2d(np.asarray(a_flat, dtype=np.float64))
    #         self.d_j.h2d(np.asarray(j_flat, dtype=np.int32))
            
    #         # Reads the true matrix lengths flawlessly
    #         self.d_rptr = crn.deviceBufferInt(len(self.raw_csr.rptr))
    #         self.d_ind = crn.deviceBufferInt(len(self.raw_csr.ind))
    #         self.d_vals = crn.deviceBufferDouble(len(self.raw_csr.vals))
            
    #         self.d_rptr.h2d(np.asarray(self.raw_csr.rptr, dtype=np.int32))
    #         self.d_ind.h2d(np.asarray(self.raw_csr.ind, dtype=np.int32))
    #         self.d_vals.h2d(np.asarray(self.raw_csr.vals, dtype=np.float64))
