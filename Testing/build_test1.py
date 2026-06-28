import sys
import os
import numpy as np

TESTING_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TESTING_DIR)
BUILD_DIR = os.path.join(PROJECT_ROOT, "build")
sys.path.append(BUILD_DIR)

try:
    import eulereasel as ee
    print("✅ Success: module imported perfectly!")
except ImportError as e:
    print(f'❌ Error: ', e)
    sys.exit(1)

# 1. Instantiating via your correct submodule namespaces
mat_vec = ee.datatype.mat_vec()
csr = ee.datatype.CSR()  # Initialize empty container

filename = "/home/fakeheadset/Projects/EulerEasel/Data/bcsstk18.mtx"

# 2. Run your file parser to populate your data elements
ee.functions.file_parser(filename, mat_vec)
[r, c, nnz] = ee.functions.Matrix_dim(filename)

# 3. Pass your empty 'csr' container down to your Csrformat builder to populate it in-place
ee.functions.Csrformat(mat_vec, r, c, nnz, csr)

# 4. Generate your dense input vector x (Make sure size matches your matrix column count c!)
# np.random.randn(c) creates an array of size 'c'. .tolist() converts it to a standard Python list.
x = ee.functions.CentralVector.generate(r, c, nnz)
# 5. Execute your highly optimized AVX CPU kernel
y = ee.functions.SpMV_kernel_AVX(csr, x)

# 6. Ensure your file output path points correctly relative to the project root
results_path = os.path.join(PROJECT_ROOT, "Src/Server/CPU/results/csr_py_avx.txt")
os.makedirs(os.path.dirname(results_path), exist_ok=True)

with open(results_path, "w", encoding="utf-8") as file:
    # Convert the resulting list of numbers to a string block so it can write cleanly
    file.write("\n".join(map(str, y)))

print("🎉 Execution finished! Results written successfully.")
