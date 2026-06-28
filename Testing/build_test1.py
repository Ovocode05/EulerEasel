import sys
import os
import numpy as np

TESTING_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TESTING_DIR)
BUILD_DIR = os.path.join(PROJECT_ROOT, "build")
sys.path.append(BUILD_DIR)

try:
    import eulereasel as ee
    print("Success: module imported perfectly!")
except ImportError as e:
    print(f'Error: ', e)
    sys.exit(1)

mat_vec = ee.datatype.mat_vec()
# csr = ee.datatype.CSR()
ell = ee.datatype.ELL()

filename = "/home/fakeheadset/Projects/EulerEasel/Data/bcsstk18.mtx"

ee.functions.file_parser(filename, mat_vec)
[r, c, nnz] = ee.functions.matdim(filename)

[A, J] = ee.functions.ell.ellformat(mat_vec, r, c, nnz, ell)

x = ee.functions.CentralVector.generate(r, c, nnz)

y = ee.functions.ell.sparseAVX_x16(x, A, J)

results_path = os.path.join(PROJECT_ROOT, "Src/Server/CPU/results/ell_py_avx.txt")
os.makedirs(os.path.dirname(results_path), exist_ok=True)

with open(results_path, "w", encoding="utf-8") as file:
    file.write("\n".join(map(str, y)))

print("Results written successfully.")
