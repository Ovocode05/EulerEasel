import numpy 
import sys
sys.path.insert(0, "/home/fakeheadset/Projects/EulerEasel/Src/include")
# Assuming 'run' and 'cudarn' are exposed from your pybind11 module 'me'
# e.g., runner = me.SpMvCPURunner() and gpu_runner = me.SpMvGPURunner()
from context import LazyFrozenContext
import matrix_extractor as me
import CUDAruntime as cudarn
import runtime as run


# def _run_cpu_hyb(ctx, avx=False):
#     # 1. Force the initialization method to run completely first
#     ctx.ensure_cpu_hyb()
    
#     # 2. Safely read and convert the now-guaranteed populated variables
#     if avx:
#         return run.proc_hyb_avx4(ctx.hyb, ctx.x.tolist(), ctx.A_np.tolist(), ctx.J_np.tolist(), ctx.r)
#     return run.proc_hyb(ctx.hyb, ctx.x.tolist(), ctx.A_np.tolist(), ctx.J_np.tolist(), ctx.r)


# Unified mapping targeting your exact C++ tuple outputs [y, runtime]
SPMV_REGISTRY = {
    # --- CPU Kernels (Dispatched to 'run' module) ---
    me.Kernel.CPU_CSR: lambda ctx: (
        ctx.ensure_cpu_csr(),
        # Pass the raw CSR object that C++ expects, not the wrapper!
        run.proc_csr(ctx.raw_full_csr, ctx.x.tolist())
    )[1],
    
    me.Kernel.CPU_ELL: lambda ctx: (
        ctx.ensure_cpu_ell(), # Fixed: Ensure the layout matches the format
        run.proc_ell(ctx.x.tolist(), ctx.A_np, ctx.J_np)
    )[1],
    
    me.Kernel.CPU_CSR_AVX: lambda ctx: (
        ctx.ensure_cpu_csr(),
        # Same fix here for the AVX variant
        run.proc_csrx4(ctx.raw_full_csr, ctx.x.tolist())
    )[1],
    
    me.Kernel.CPU_ELL_AVX_x4: lambda ctx: (
        ctx.ensure_cpu_ell(),
        run.proc_ellx4(ctx.x.tolist(), ctx.A_np, ctx.J_np)
    )[1],
    
    me.Kernel.CPU_ELL_AVX_x16: lambda ctx: (
        ctx.ensure_cpu_ell(),
        run.proc_ell_x16(ctx.x.tolist(), ctx.A_np, ctx.J_np)
    )[1],
    
    # me.Kernel.CPU_HYB:     lambda ctx: _run_cpu_hyb(ctx, avx=False),
    # me.Kernel.CPU_HYB_AVX: lambda ctx: _run_cpu_hyb(ctx, avx=True),

    # --- GPU Kernels (Dispatched to 'cudarn' module) ---
    me.Kernel.GPU_CSR: lambda ctx: (
        ctx.ensure_gpu_csr(),
        cudarn.cuda_csr(
            threads=ctx.threads, blocks=ctx.blocks, 
            d_rptr=ctx.d_rptr, d_ind=ctx.d_ind, d_vals=ctx.d_vals, 
            numrows=ctx.r, d_x=ctx.d_x, d_y=ctx.d_y
        ),
        # ctx.d_y.d2h()
    )[1],
    
    me.Kernel.GPU_ELL: lambda ctx: ( 
        ctx.ensure_gpu_ell(),
        cudarn.cuda_ell(
            threads=ctx.threads, blocks=ctx.blocks, 
            A=ctx.d_a, J=ctx.d_j, rows=ctx.ell_rows, cols=ctx.ell_cols, 
            d_x=ctx.d_x, d_y=ctx.d_y
        ),
        # ctx.d_y.d2h()

    )[1],
    
    # me.Kernel.GPU_HYB: lambda ctx: (
    #     ctx.ensure_gpu_hyb(),
    #     cudarn.cuda_hyb(
    #         threads=ctx.threads, 
    #         blocks=ctx.blocks, 
    #         d_rptr=ctx.d_rptr,     # No longer None! Guaranteed populated by ensure_gpu_hyb()
    #         d_ind=ctx.d_ind,       # No longer None!
    #         d_vals=ctx.d_vals,     # No longer None!
    #         d_a=ctx.d_a, 
    #         d_j=ctx.d_j, 
    #         ell_rows=ctx.r, 
    #         csr_rows=ctx.r, 
    #         cols=ctx.c, 
    #         d_x=ctx.d_x, 
    #         d_y=ctx.d_y, 
    #         stream1=getattr(cudarn, 'cudaStream', lambda x: None)(True), 
    #         stream2=getattr(cudarn, 'cudaStream', lambda x: None)(True)
    #     )
    #     # ctx.d_y.d2h()        
    # )[1],
}

def launch_spmv(kernel_type: me.Kernel, context: LazyFrozenContext):
    launcher = SPMV_REGISTRY.get(kernel_type)
    if launcher is None:
        raise NotImplementedError(f"Kernel '{kernel_type.name}' mapping is missing.")
    return launcher(context)
