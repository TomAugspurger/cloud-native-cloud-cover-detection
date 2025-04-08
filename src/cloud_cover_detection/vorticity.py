"""
Sample program


Based off https://github.com/cubed-dev/cubed/blob/main/examples/pangeo-1-vorticity.ipynb

Collecting data...
CPU: 6.8654 seconds
CPU: 6.8199 seconds
CPU: 7.2130 seconds
GPU: 0.0244 seconds
GPU: 0.0237 seconds
GPU: 0.0237 seconds

~28x speedup *on just the computation.*

"""

import time
import xarray as xr
import numpy as np
import cupy_xarray  # noqa: F401
import nvtx
import cupy

# SIZE = (5000, 1, 987, 1920)
SIZE = (200, 1, 987, 1920)


def f(ds: xr.Dataset) -> None:
    quad = ds**2
    quad["UV"] = ds.U * ds.V
    quad.mean("time")


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


def plain_streams(U: cupy.ndarray, V: cupy.ndarray) -> cupy.ndarray:
    # %timeit plain_streams(U, V)
    # 75 ms ± 8.1 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    uu_stream = cupy.cuda.Stream()
    vv_stream = cupy.cuda.Stream()
    uv_stream = cupy.cuda.Stream()

    with uu_stream:
        UU = U * U
        a = UU.mean()
    with vv_stream:
        VV = V * V
        b = VV.mean()
    with uv_stream:
        UV = U * V
        c = UV.mean()

    for stream in uu_stream, vv_stream, uv_stream:
        stream.synchronize()

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

    # warmup
    f(ds)
    with cupy.cuda.Stream() as stream:
        f(ds_gpu)
        stream.synchronize()

    for i in range(3):
        start = time.perf_counter()
        f(ds)
        end = time.perf_counter()
        print(f"CPU: {(end - start):0.4f} seconds")

    for i in range(3):
        with nvtx.annotate("GPU", color="red"), cupy.cuda.Stream() as stream:
            start = time.perf_counter()
            f(ds_gpu)
            stream.synchronize()
            end = time.perf_counter()
            print(f"GPU: {(end - start):0.4f} seconds")

    U_ = ds_gpu.U.data
    V_ = ds_gpu.V.data
    for i in range(3):
        with nvtx.annotate("gpu-stream", color="red"):
            start = time.perf_counter()
            plain_streams(U_, V_)
            # already synchronized
            end = time.perf_counter()
            print(f"GPU-streams: {(end - start):0.4f} seconds")

    for i in range(3):
        with nvtx.annotate("gpu-fast", color="green"):
            start = time.perf_counter()
            fused(U_, V_)
            # already synchronized
            end = time.perf_counter()
            print(f"GPU fused: {(end - start):0.4f} seconds")

    for i in range(3):
        with nvtx.annotate("gpu-superfast", color="purple"):
            start = time.perf_counter()
            super_fast(U_, V_)
            # already synchronized
            end = time.perf_counter()
            print(f"GPU super_fast: {(end - start):0.4f} seconds")

    # Verify all implementations return the same result
    a1, b1, c1 = plain(U_, V_)
    a2, b2, c2 = plain_streams(U_, V_)
    a3, b3, c3 = fused(U_, V_)
    a4, b4, c4 = super_fast(U_, V_)

    with nvtx.annotate("verify", color="blue"):
        print("\nVerifying results match:")
        print(f"Plain:         {a1:.6f}, {b1:.6f}, {c1:.6f}")
        print(f"Plain streams: {a2:.6f}, {b2:.6f}, {c2:.6f}")
        print(f"Fast:          {a3:.6f}, {b3:.6f}, {c3:.6f}")
        print(f"Super fast:    {a4:.6f}, {b4:.6f}, {c4:.6f}")


if __name__ == "__main__":
    main()
