import numba
import numpy as np
from numba import cuda
from numba import uint64, void
from scipy.sparse import csr_matrix, csr_array
import time

cuda_on = cuda.is_available()

precision = 64

if precision == 64:
    np_complex = np.complex64
    nb_complex = numba.complex64

if precision == 128:
    np_complex = np.complex128
    nb_complex = numba.complex128

if cuda_on:
    blocks_per_grid = 512
    threads_per_block = 512

    @cuda.jit(void(uint64[:], uint64[:], nb_complex[:], nb_complex[:], nb_complex[:]))
    def cuda_dot(indices, indptr, data, x, scratchpad):
        """dot product of CSR matrix and vector"""

        m = len(scratchpad)
        start = cuda.grid(1)
        stepsize = cuda.gridsize(1)
        for i in range(start, m, stepsize):
            scratchpad[i] = 0
            for j in range(indptr[i], indptr[i + 1]):
                scratchpad[i] += data[j] * x[indices[j]]

    cudadot = cuda_dot[blocks_per_grid, threads_per_block]

    class CSR_CUDA:
        def __init__(self, matrix: csr_matrix):
            """https://en.wikipedia.org/wiki/Sparse_matrix, CSR format"""

            self.data = cuda.to_device(matrix.data.astype(np_complex))
            self.indices = cuda.to_device(matrix.indices.astype(np.uint64))
            self.indptr = cuda.to_device(matrix.indptr.astype(np.uint64))
            self.shape = matrix.shape

        def dot(self, x, scratchpad=None):
            """dot product of CSR matrix and vector"""
            if scratchpad is None:
                scratchpad = cuda.device_array(self.shape[0], dtype=x.dtype)

            cudadot(self.indices, self.indptr, self.data, x, scratchpad)
            return scratchpad


def cuda_if_avail_csr(matrix):
    if cuda_on:
        return CSR_CUDA(matrix.astype(np_complex))
    else:
        return matrix


def cuda_if_avail_vec(x):
    if cuda_on:
        return cuda.to_device(x.astype(np_complex))
    else:
        return x


def test(
    M: csr_matrix, x: np.ndarray, compare_scipy=True, print_result=False, iterations=100
):
    assert cuda_on
    M_ = cuda_if_avail_csr(M)
    x_ = cuda_if_avail_vec(x)
    scratchpad = cuda.device_array(len(x_), dtype=x.dtype)

    t0 = time.time()
    for _ in range(iterations):
        y_ = M_.dot(x_, scratchpad)
        x_, scratchpad = y_, x_
    cuda.synchronize()
    t1 = time.time()

    print(f"m={len(y_)}, n={len(x_)}")
    print(f"CUDA {(t1-t0)/iterations:.6f} seconds")

    if compare_scipy:
        t0 = time.time()
        for _ in range(iterations):
            x = M.dot(x)
            y = x
        t1 = time.time()
        print(f"Scipy {(t1-t0)/iterations:.6f} seconds")

        if print_result:
            print(y_.copy_to_host())
            print(y)

        np.testing.assert_allclose(y_, y)
        print("correctness test passed")


def maketest(n=256, k=10):
    M = 0
    for i in range(k):
        M += csr_array(
            (np.ones(n, dtype=np.complex64), (np.arange(n), (np.arange(n) + i) % n)),
            shape=(n, n),
        )
    x = np.arange(n, dtype=np.complex64)
    return M, x


def print_test_example(n=4, k=2):
    M, x = maketest(n, k)
    print("\ntest example")
    print(M.todense())
    print(x, "\n")


if __name__ == "__main__":
    print_test_example()

    M, x = maketest(100000)
    test(M, x)
