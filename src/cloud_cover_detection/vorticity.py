"""
Sample program that computes some vorticities.


Based off https://github.com/cubed-dev/cubed/blob/main/examples/pangeo-1-vorticity.ipynb

Collecting data...

Copy host2device    : 0.5123 seconds
CPU                 : 9.4245 seconds
GPU                 : 0.0235 seconds
GPU plain           : 0.0203 seconds
GPU fused           : 0.0158 seconds
GPU super_fast      : 0.0040 seconds
GPU numba_super_fast: 0.0059 seconds

~28x speedup *on just the computation.*
"""

import time
from typing import Callable
import xarray as xr
import numpy as np
import cupy_xarray  # noqa: F401
import nvtx
import cupy
import numba
from numba import cuda

numba.config.CUDA_ENABLE_PYNVJITLINK = True

# SIZE = (5000, 1, 987, 1920)
SIZE = (200, 1, 987, 1920)


def benchmark(f: Callable, *args, **kwargs) -> float:
    """
    Benchmark the given function.

    Runs the function 3 times and returns the median time.
    """
    # warmup
    f(*args, **kwargs)
    times = []
    for _ in range(3):
        start = time.perf_counter()
        f(*args, **kwargs)
        end = time.perf_counter()
        times.append(end - start)
    return np.median(times).item()


def f(ds: xr.Dataset) -> None:
    quad = ds**2
    quad["UV"] = ds.U * ds.V
    result = quad.mean("time")

    cupy.cuda.Stream.null.synchronize()

    return result


@nvtx.annotate("GPU-plain", color="green")
def plain(U: cupy.ndarray, V: cupy.ndarray) -> cupy.ndarray:
    # three output arrays
    # U * U, V * V, U * V
    # In [36]: %timeit plain(U, V)
    # 19.4 ms ± 162 μs per loop (mean ± std. dev. of 7 runs, 10 loops each)
    UU = U * U
    VV = V * V
    UV = U * V

    a = UU.mean()
    b = VV.mean()
    c = UV.mean()

    cupy.cuda.Stream.null.synchronize()

    return cupy.array([a, b, c])


@cupy.fuse(kernel_name="fast_kernel")
def fused_uv(
    u: cupy.ndarray, v: cupy.ndarray
) -> tuple[cupy.ndarray, cupy.ndarray, cupy.ndarray]:
    uu = u * u
    vv = v * v
    uv = u * v
    return uu, vv, uv


def fused(U: cupy.ndarray, V: cupy.ndarray) -> cupy.ndarray:
    # Use CuPy's fused kernel capability for maximum performance
    # This creates a single GPU kernel that does all operations
    # in one pass through the data, maximizing memory locality

    # Execute the fused kernel
    UU, VV, UV = fused_uv(U, V)

    # Create a non-blocking stream for computing means
    stream = cupy.cuda.Stream(non_blocking=True)

    # Use raw reduction kernels for means to get maximum performance
    # These will execute asynchronously in the stream
    with stream:
        a = UU.mean()
        b = VV.mean()
        c = UV.mean()

    # Ensure calculations are complete
    stream.synchronize()

    return cupy.array([a, b, c])


@nvtx.annotate("GPU-superfast", color="purple")
def super_fast(U: cupy.ndarray, V: cupy.ndarray) -> cupy.ndarray:
    """
    Maximally optimized version that:
    1. Uses a single fused kernel for all operations
    2. Performs reduction in the same kernel to minimize memory traffic
    3. Uses advanced CuPy optimizations for maximum throughput
    """
    # Create a custom kernel that does multiplication and reduction in one pass
    # This eliminates intermediate array storage and reduces memory bandwidth needs
    dtype = U.dtype
    kernel_code = """
    extern "C" __global__
    void super_fast_kernel(const {dtype}* u, const {dtype}* v, {dtype}* results, int size) {{
        // Shared memory for parallel reduction
        __shared__ {dtype} sdata[512];

        // Initialize accumulators for UU, VV, UV
        {dtype} acc_uu = 0;
        {dtype} acc_vv = 0;
        {dtype} acc_uv = 0;

        // Grid stride loop for processing large arrays
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
             i < size; 
             i += blockDim.x * gridDim.x) {{
            {dtype} u_val = u[i];
            {dtype} v_val = v[i];

            // Compute products and accumulate in thread's registers
            acc_uu += u_val * u_val;
            acc_vv += v_val * v_val;
            acc_uv += u_val * v_val;
        }}

        // First level of reduction in shared memory
        int tid = threadIdx.x;
        sdata[tid] = acc_uu;
        __syncthreads();

        // Parallel reduction for UU
        for (int s = blockDim.x/2; s > 0; s >>= 1) {{
            if (tid < s) {{
                sdata[tid] += sdata[tid + s];
            }}
            __syncthreads();
        }}

        // Thread 0 writes UU result
        if (tid == 0) atomicAdd(&results[0], sdata[0]);

        // Repeat for VV
        sdata[tid] = acc_vv;
        __syncthreads();

        for (int s = blockDim.x/2; s > 0; s >>= 1) {{
            if (tid < s) {{
                sdata[tid] += sdata[tid + s];
            }}
            __syncthreads();
        }}

        if (tid == 0) atomicAdd(&results[1], sdata[0]);

        // Repeat for UV
        sdata[tid] = acc_uv;
        __syncthreads();

        for (int s = blockDim.x/2; s > 0; s >>= 1) {{
            if (tid < s) {{
                sdata[tid] += sdata[tid + s];
            }}
            __syncthreads();
        }}

        if (tid == 0) atomicAdd(&results[2], sdata[0]);
    }}
    """.format(dtype="float" if dtype == cupy.float32 else "double")

    # TODO: translate this to cuda.core
    # TODO: try numba.cuda
    # TODO: try cuda.cooperative
    # tid = cuda.grid(1)
    # for i in range(tid, in_arr.size, cuda.blockDim.x * cuda.gridDim.x) -ish

    # Compile the kernel
    module = cupy.RawModule(code=kernel_code)
    kernel = module.get_function("super_fast_kernel")

    # Setup output array and call kernel
    size = U.size
    results = cupy.zeros(3, dtype=dtype)

    # Calculate grid and block dimensions for optimal occupancy
    block_size = 512  # Must match the shared memory size in kernel
    grid_size = min(4096, (size + block_size - 1) // block_size)

    # Execute the kernel
    kernel((grid_size,), (block_size,), (U, V, results, size))

    # Compute means by dividing by size
    results /= size

    cupy.cuda.Stream.null.synchronize()

    return results


@cuda.jit
def numba_super_fast_kernel(u, v, results, size):
    # Shared memory for parallel reduction
    sdata = cuda.shared.array(shape=512, dtype=numba.float32)

    # Initialize accumulators for UU, VV, UV
    acc_uu = 0.0
    acc_vv = 0.0
    acc_uv = 0.0

    # Grid stride loop for processing large arrays
    for i in range(cuda.grid(1), size, cuda.gridDim.x * cuda.blockDim.x):
        u_val = u[i]
        v_val = v[i]

        # Compute products and accumulate in thread's registers
        acc_uu += u_val * u_val
        acc_vv += v_val * v_val
        acc_uv += u_val * v_val

    # First level of reduction in shared memory
    tid = cuda.threadIdx.x
    sdata[tid] = acc_uu
    cuda.syncthreads()

    # Parallel reduction for UU
    s = cuda.blockDim.x // 2
    while s > 0:
        if tid < s:
            sdata[tid] += sdata[tid + s]
        cuda.syncthreads()
        s //= 2

    # Thread 0 writes UU result using atomic add
    if tid == 0:
        cuda.atomic.add(results, 0, sdata[0])

    # Repeat for VV
    sdata[tid] = acc_vv
    cuda.syncthreads()

    s = cuda.blockDim.x // 2
    while s > 0:
        if tid < s:
            sdata[tid] += sdata[tid + s]
        cuda.syncthreads()
        s //= 2

    if tid == 0:
        cuda.atomic.add(results, 1, sdata[0])

    # Repeat for UV
    sdata[tid] = acc_uv
    cuda.syncthreads()

    s = cuda.blockDim.x // 2
    while s > 0:
        if tid < s:
            sdata[tid] += sdata[tid + s]
        cuda.syncthreads()
        s //= 2

    if tid == 0:
        cuda.atomic.add(results, 2, sdata[0])


@nvtx.annotate("GPU-numba-superfast", color="blue")
def numba_super_fast(U: cupy.ndarray, V: cupy.ndarray) -> cupy.ndarray:
    """
    Numba implementation of the super_fast function that:
    1. Uses a single kernel for all operations
    2. Performs reduction in the same kernel to minimize memory traffic
    3. Uses Numba CUDA for high performance
    """

    # Setup the kernel
    size = U.size
    # results = cuda.device_array(3, dtype=U.dtype)
    # results.copy_to_device(np.zeros(3, dtype=U.dtype))
    results = cupy.zeros(3, dtype=U.dtype)

    # Calculate grid and block dimensions for optimal occupancy
    block_size = 512  # Must match the shared memory size in kernel
    grid_size = min(4096, (size + block_size - 1) // block_size)

    # Execute the kernel
    numba_super_fast_kernel[grid_size, block_size](U.ravel(), V.ravel(), results, size)
    results /= size

    # Synchronize to ensure completion
    cuda.synchronize()
    return results


def main():
    # CPU
    U = xr.DataArray(
        name="U",
        data=np.random.random(SIZE).astype(np.float32),
        dims=["time", "face", "j", "i"],
    )
    V = xr.DataArray(
        name="V",
        data=np.random.random(SIZE).astype(np.float32),
        dims=["time", "face", "j", "i"],
    )
    ds = xr.merge([U, V])
    ds_gpu = ds.cupy.as_cupy()

    print(f"Copy host2device    : {benchmark(ds.cupy.as_cupy):0.4f} seconds")

    # warmup
    f(ds)
    with cupy.cuda.Stream() as stream:
        f(ds_gpu)
        stream.synchronize()

    U_ = ds_gpu.U.data
    V_ = ds_gpu.V.data

    print(f"CPU                 : {benchmark(f, ds):0.4f} seconds")
    print(
        f"GPU                 : {benchmark(nvtx.annotate('GPU', color='red')(f), ds_gpu):0.4f} seconds"
    )
    print(f"GPU plain           : {benchmark(plain, U_, V_):0.4f} seconds")
    print(f"GPU fused           : {benchmark(fused, U_, V_):0.4f} seconds")
    print(f"GPU super_fast      : {benchmark(super_fast, U_, V_):0.4f} seconds")
    print(f"GPU numba_super_fast: {benchmark(numba_super_fast, U_, V_):0.4f} seconds")

    # Verify all implementations return the same result
    a0, b0, c0 = plain(U.data, V.data)  # host
    a1, b1, c1 = plain(U_, V_)
    a3, b3, c3 = fused(U_, V_)
    a4, b4, c4 = super_fast(U_, V_)
    a5, b5, c5 = numba_super_fast(U_, V_)

    with nvtx.annotate("verify", color="blue"):
        print("\nVerifying results match:")
        print(f"Host :         {a0:.6f}, {b0:.6f}, {c0:.6f}")
        print(f"Plain:         {a1:.6f}, {b1:.6f}, {c1:.6f}")
        print(f"Fast:          {a3:.6f}, {b3:.6f}, {c3:.6f}")
        print(f"Super fast:    {a4:.6f}, {b4:.6f}, {c4:.6f}")
        print(f"Numba fast:    {a5:.6f}, {b5:.6f}, {c5:.6f}")


if __name__ == "__main__":
    main()
