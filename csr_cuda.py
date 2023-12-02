import numba
import numpy as np
from numba import cuda
from scipy.sparse import csr_matrix

cuda_on = cuda.is_available()


if cuda_on:

    @cuda.jit("void(complex128[:], int64[:], int64[:], complex128[:], complex128[:])")
    def dot(data, indices, indptr, x, outvector):
        """dot product of CSR matrix and vector"""

        m = len(indptr) - 1
        start = cuda.grid(1)
        stepsize = cuda.gridsize(1)
        for i in range(start, m, stepsize):
            outvector[i] = 0
            for j in range(indptr[i], indptr[i + 1]):
                outvector[i] += data[j] * x[indices[j]]

    class CSR_CUDA:
        def __init__(self, matrix: csr_matrix):
            """https://en.wikipedia.org/wiki/Sparse_matrix, CSR format"""

            self.data = cuda.to_device(matrix.data)
            self.indices = cuda.to_device(matrix.indices)
            self.indptr = cuda.to_device(matrix.indptr)
            self.shape = matrix.shape

        def dot(self, x):
            """dot product of CSR matrix and vector"""
            outvector = cuda.device_array(self.shape[0])
            dot[self.shape[0], 128](self.data, self.indices, self.indptr, x, outvector)
            return outvector


def cuda_if_avail(matrix):
    if cuda_on:
        return CSR_CUDA(matrix)
    else:
        return matrix


if __name__ == "__main__":
    x = np.array([1, 2, 3, 4], dtype=np.complex128)
    M = csr_matrix(
        [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ]
    )
    csr = cuda_if_avail(M)
    print(csr.dot(x))
