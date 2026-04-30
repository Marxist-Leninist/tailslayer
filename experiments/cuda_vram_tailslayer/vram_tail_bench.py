from __future__ import annotations

import argparse
import ctypes
import os
import pathlib
import statistics
import sys
from dataclasses import dataclass

import numpy as np


CUDA_BIN_DIRS = [
    r"C:\Program Files\Python312\Lib\site-packages\nvidia\cuda_runtime\bin",
    r"C:\Program Files\Python312\Lib\site-packages\nvidia\cuda_nvrtc\bin",
]


for dll_dir in CUDA_BIN_DIRS:
    if os.path.isdir(dll_dir):
        os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(dll_dir)

from cuda.bindings import driver as cu  # noqa: E402
from cuda.bindings import nvrtc  # noqa: E402


KERNEL_SRC = br"""
extern "C" __global__ void single_warp_latency(
    const float* table,
    const unsigned int* indices,
    unsigned long long* latencies,
    float* sink,
    int samples
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp = tid >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= samples || lane != 0) return;

    unsigned int idx = indices[warp];
    unsigned long long t0 = clock64();
    volatile float value = table[idx];
    unsigned long long t1 = clock64();
    latencies[warp] = t1 - t0;
    sink[warp] = value;
}

extern "C" __global__ void hedged_warp_latency(
    const float* table0,
    const float* table1,
    const unsigned int* indices,
    unsigned long long* lat0,
    unsigned long long* lat1,
    float* sink0,
    float* sink1,
    int samples
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp = tid >> 5;
    int lane = threadIdx.x & 31;
    int sample = warp >> 1;
    int replica = warp & 1;
    if (sample >= samples || lane != 0) return;

    unsigned int idx = indices[sample];
    if (replica == 0) {
        unsigned long long t0 = clock64();
        volatile float value = table0[idx];
        unsigned long long t1 = clock64();
        lat0[sample] = t1 - t0;
        sink0[sample] = value;
    } else {
        unsigned long long t0 = clock64();
        volatile float value = table1[idx];
        unsigned long long t1 = clock64();
        lat1[sample] = t1 - t0;
        sink1[sample] = value;
    }
}
"""


def _err_value(err) -> int:
    return int(getattr(err, "value", err))


def check(result, label: str = "cuda"):
    if isinstance(result, tuple):
        err = result[0]
        if _err_value(err) != 0:
            raise RuntimeError(f"{label} failed: {result}")
        if len(result) == 1:
            return None
        if len(result) == 2:
            return result[1]
        return result[1:]
    if _err_value(result) != 0:
        raise RuntimeError(f"{label} failed: {result}")
    return None


def compile_ptx(arch: str) -> bytes:
    prog = check(
        nvrtc.nvrtcCreateProgram(KERNEL_SRC, b"vram_tail_bench.cu", 0, [], []),
        "nvrtcCreateProgram",
    )
    options = [f"--gpu-architecture={arch}".encode(), b"--std=c++11"]
    err = nvrtc.nvrtcCompileProgram(prog, len(options), options)
    if _err_value(err[0]) != 0:
        log_size = check(nvrtc.nvrtcGetProgramLogSize(prog), "nvrtcGetProgramLogSize")
        log = bytearray(log_size)
        check(nvrtc.nvrtcGetProgramLog(prog, log), "nvrtcGetProgramLog")
        raise RuntimeError(log.decode(errors="replace"))

    ptx_size = check(nvrtc.nvrtcGetPTXSize(prog), "nvrtcGetPTXSize")
    ptx = bytearray(ptx_size)
    check(nvrtc.nvrtcGetPTX(prog, ptx), "nvrtcGetPTX")
    check(nvrtc.nvrtcDestroyProgram(prog), "nvrtcDestroyProgram")
    return bytes(ptx)


def launch(fn, grid: int, block: int, args: list[ctypes._SimpleCData]) -> float:
    start = check(cu.cuEventCreate(0), "cuEventCreate(start)")
    end = check(cu.cuEventCreate(0), "cuEventCreate(end)")
    params = (ctypes.c_void_p * len(args))(
        *[ctypes.addressof(arg) for arg in args]
    )
    check(cu.cuEventRecord(start, 0), "cuEventRecord(start)")
    check(
        cu.cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, 0, params, 0),
        "cuLaunchKernel",
    )
    check(cu.cuEventRecord(end, 0), "cuEventRecord(end)")
    check(cu.cuEventSynchronize(end), "cuEventSynchronize")
    elapsed_ms = check(cu.cuEventElapsedTime(start, end), "cuEventElapsedTime")
    check(cu.cuEventDestroy(start), "cuEventDestroy(start)")
    check(cu.cuEventDestroy(end), "cuEventDestroy(end)")
    return float(elapsed_ms)


def dev_alloc(host: np.ndarray):
    ptr = check(cu.cuMemAlloc(host.nbytes), "cuMemAlloc")
    check(cu.cuMemcpyHtoD(ptr, host, host.nbytes), "cuMemcpyHtoD")
    return ptr


def empty_dev(size_bytes: int):
    return check(cu.cuMemAlloc(size_bytes), "cuMemAlloc")


def copy_from_dev(ptr, host: np.ndarray) -> np.ndarray:
    check(cu.cuMemcpyDtoH(host, ptr, host.nbytes), "cuMemcpyDtoH")
    return host


@dataclass
class Dist:
    mean: float
    p50: float
    p95: float
    p99: float
    p999: float
    max: float


def dist(values: np.ndarray) -> Dist:
    return Dist(
        mean=float(np.mean(values)),
        p50=float(np.percentile(values, 50)),
        p95=float(np.percentile(values, 95)),
        p99=float(np.percentile(values, 99)),
        p999=float(np.percentile(values, 99.9)),
        max=float(np.max(values)),
    )


def print_dist(name: str, values: np.ndarray) -> None:
    d = dist(values)
    print(
        f"{name:18} mean={d.mean:8.2f} p50={d.p50:8.2f} "
        f"p95={d.p95:8.2f} p99={d.p99:8.2f} p999={d.p999:8.2f} max={d.max:8.2f}"
    )


def run(args: argparse.Namespace) -> int:
    check(cu.cuInit(0), "cuInit")
    device = check(cu.cuDeviceGet(0), "cuDeviceGet")
    major, minor = check(cu.cuDeviceComputeCapability(device), "cuDeviceComputeCapability")
    name_buf = check(cu.cuDeviceGetName(128, device), "cuDeviceGetName")
    gpu_name = bytes(name_buf).split(b"\x00", 1)[0].decode(errors="replace")
    total_mem = check(cu.cuDeviceTotalMem(device), "cuDeviceTotalMem")

    arch = args.arch or f"compute_{major}{minor}"
    print(f"gpu={gpu_name} cc={major}.{minor} total_vram_mb={total_mem / 1024 / 1024:.0f} arch={arch}")

    context = check(cu.cuCtxCreate(0, device), "cuCtxCreate")
    allocations = []
    try:
        ptx = compile_ptx(arch)
        module = check(cu.cuModuleLoadData(ptx), "cuModuleLoadData")
        single_fn = check(cu.cuModuleGetFunction(module, b"single_warp_latency"), "cuModuleGetFunction(single)")
        hedged_fn = check(cu.cuModuleGetFunction(module, b"hedged_warp_latency"), "cuModuleGetFunction(hedged)")

        table_elems = (args.table_mb * 1024 * 1024) // np.dtype(np.float32).itemsize
        rng = np.random.default_rng(args.seed)
        table0 = rng.standard_normal(table_elems, dtype=np.float32)
        table1 = table0.copy()
        indices = rng.integers(0, table_elems, size=args.samples, dtype=np.uint32)

        d_table0 = dev_alloc(table0); allocations.append(d_table0)
        d_table1 = dev_alloc(table1); allocations.append(d_table1)
        d_indices = dev_alloc(indices); allocations.append(d_indices)
        lat_single = np.zeros(args.samples, dtype=np.uint64)
        lat0 = np.zeros(args.samples, dtype=np.uint64)
        lat1 = np.zeros(args.samples, dtype=np.uint64)
        sink = np.zeros(args.samples, dtype=np.float32)
        d_lat_single = empty_dev(lat_single.nbytes); allocations.append(d_lat_single)
        d_lat0 = empty_dev(lat0.nbytes); allocations.append(d_lat0)
        d_lat1 = empty_dev(lat1.nbytes); allocations.append(d_lat1)
        d_sink0 = empty_dev(sink.nbytes); allocations.append(d_sink0)
        d_sink1 = empty_dev(sink.nbytes); allocations.append(d_sink1)

        block = 256
        single_warps = args.samples
        hedged_warps = args.samples * 2
        single_grid = (single_warps * 32 + block - 1) // block
        hedged_grid = (hedged_warps * 32 + block - 1) // block

        single_all = []
        hedged_min_all = []
        hedged0_all = []
        hedged1_all = []
        single_ms = []
        hedged_ms = []

        for repeat in range(args.repeats + args.warmup):
            ms = launch(
                single_fn,
                single_grid,
                block,
                [
                    ctypes.c_uint64(int(d_table0)),
                    ctypes.c_uint64(int(d_indices)),
                    ctypes.c_uint64(int(d_lat_single)),
                    ctypes.c_uint64(int(d_sink0)),
                    ctypes.c_int(args.samples),
                ],
            )
            copy_from_dev(d_lat_single, lat_single)

            hm = launch(
                hedged_fn,
                hedged_grid,
                block,
                [
                    ctypes.c_uint64(int(d_table0)),
                    ctypes.c_uint64(int(d_table1)),
                    ctypes.c_uint64(int(d_indices)),
                    ctypes.c_uint64(int(d_lat0)),
                    ctypes.c_uint64(int(d_lat1)),
                    ctypes.c_uint64(int(d_sink0)),
                    ctypes.c_uint64(int(d_sink1)),
                    ctypes.c_int(args.samples),
                ],
            )
            copy_from_dev(d_lat0, lat0)
            copy_from_dev(d_lat1, lat1)

            if repeat >= args.warmup:
                single_ms.append(ms)
                hedged_ms.append(hm)
                single_all.append(lat_single.copy())
                hedged0_all.append(lat0.copy())
                hedged1_all.append(lat1.copy())
                hedged_min_all.append(np.minimum(lat0, lat1))

        single_values = np.concatenate(single_all)
        h0_values = np.concatenate(hedged0_all)
        h1_values = np.concatenate(hedged1_all)
        hmin_values = np.concatenate(hedged_min_all)

        print(f"table_mb_per_replica={args.table_mb} samples_per_repeat={args.samples} repeats={args.repeats}")
        print_dist("single_cycles", single_values)
        print_dist("replica0_cycles", h0_values)
        print_dist("replica1_cycles", h1_values)
        print_dist("effective_min", hmin_values)
        sd = dist(single_values)
        md = dist(hmin_values)
        print(f"p99_effective_speedup={sd.p99 / md.p99:.3f} p999_effective_speedup={sd.p999 / md.p999:.3f}")
        print(f"single_kernel_ms_mean={statistics.mean(single_ms):.3f} hedged_kernel_ms_mean={statistics.mean(hedged_ms):.3f}")
        print(f"kernel_walltime_ratio_single_over_hedged={statistics.mean(single_ms) / statistics.mean(hedged_ms):.3f}")
    finally:
        for ptr in reversed(allocations):
            try:
                check(cu.cuMemFree(ptr), "cuMemFree")
            except Exception:
                pass
        check(cu.cuCtxDestroy(context), "cuCtxDestroy")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table-mb", type=int, default=128)
    parser.add_argument("--samples", type=int, default=200_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--arch", default=None)
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
