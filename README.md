[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![GitHub stars](https://img.shields.io/github/stars/LaurieWired/tailslayer)](https://github.com/LaurieWired/tailslayer/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/LaurieWired/tailslayer)](https://github.com/LaurieWired/tailslayer/network/members)
[![GitHub contributors](https://img.shields.io/github/contributors/LaurieWired/tailslayer)](https://github.com/LaurieWired/tailslayer/graphs/contributors)
[![Follow @lauriewired](https://img.shields.io/twitter/follow/lauriewired?style=social)](https://twitter.com/lauriewired)

<img width="2490" height="1148" alt="tailslayer3" src="https://github.com/user-attachments/assets/35d6cd98-ab9d-4ea6-8804-dff3a1b8698b" />

# Tailslayer

Tailslayer is a C++ library that reduces tail latency in RAM reads caused by DRAM refresh stalls. 

It replicates data across multiple, independent DRAM channels with uncorrelated refresh schedules, using (undocumented!) channel scrambling offsets that works on AMD, Intel, and Graviton. Once the request comes in, Tailslayer issues hedged reads across all replicas, allowing the work to be performed on whichever result responds first.


<img width="4986" height="2796" alt="cross_platform_nway" src="https://github.com/user-attachments/assets/4b4a5614-00e4-4845-8a4b-f4adecef5b4d" />

## Usage

The library code is available in [hedged_reader.cpp](https://github.com/LaurieWired/tailslayer/blob/main/include/tailslayer/hedged_reader.hpp) and the example using the library can be found in [tailslayer_example.cpp](https://github.com/LaurieWired/tailslayer/blob/main/tailslayer_example.cpp). To use it, copy `include/tailslayer` into your project and `#include <tailslayer/hedged_reader.hpp>`. The library currently works with two channels (updates to come!), but full N-way usage is available in the [benchmark](https://github.com/LaurieWired/tailslayer/tree/main/discovery/benchmark).

You provide the value type and two functions as template parameters:

1. **Signal function**: Add the loop that waits for the external signal. This determines when to read. Return the desired index to read, and the read immediately fires.
2. **Final work function**: This receives the value immediately after it is read. Add the desired value processing code here.

```cpp
#include <tailslayer/hedged_reader.hpp>

TAILSLAYER_ALWAYS_INLINE std::size_t my_signal() {
    // Wait for your event, then return the index to read
    return index_to_read;
}

template <typename T>
TAILSLAYER_ALWAYS_INLINE void my_work(T val) {
    // Use the value
}

int main() {
    using T = uint8_t;
    tailslayer::pin_to_core(tailslayer::CORE_MAIN);

    tailslayer::HedgedReader<T, my_signal, my_work<T>> reader{};
    reader.insert(0x43);
    reader.insert(0x44);
    reader.start_workers();
}
```

Arguments can be passed to either function via `ArgList`:

```cpp
tailslayer::HedgedReader<T, my_signal, my_work<T>,
    tailslayer::ArgList<1, 2>,   // args to signal function
    tailslayer::ArgList<2>       // args to final work function
> reader{};
```

You can also optionally pass in a different channel offset, channel bit, and number of replicas to the constructor. *Note:* Each insert copies the element N times where N is the number of replicas. It does the address calculation work on the backend, allowing tailslayer to act as a hedged vector that uses logical indices. Additionally, each replica is pinned to a separate core, and will spin on that core according to the signal function until the read happens.

## Build the example

```bash
make
./tailslayer_example
```

On Windows, build with CMake from a Visual Studio C++ environment:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
.\build\Release\tailslayer_example.exe
```

Or use the portable LLVM-MinGW build script:

```powershell
.\scripts\build_windows_llvm_mingw.ps1 -ToolchainBin C:\Users\User\cppdev\llvm-mingw-20260421-ucrt-x86_64\bin
.\build-windows\transformer_tailslayer_bench.exe --mode all --iters 1000 --vocab 1048576 --seq-len 16 --d-model 32 --replicas 2 --first-core 0
```

Windows builds use `VirtualAlloc`, `VirtualLock`, and `SetThreadAffinityMask`. If large pages are unavailable, Tailslayer falls back to normal committed pages so examples and experiments can run, but that fallback does not prove physical DRAM-channel placement.

## Transformer experiment

`experiments/transformer_tailslayer/` contains a CPU transformer-shaped benchmark that compares single-worker inference/training with a Tailslayer-style replicated embedding table and duplicate workers.

```powershell
.\build\experiments\transformer_tailslayer\Release\transformer_tailslayer_bench.exe --mode all --iters 1000 --vocab 1048576 --seq-len 16 --d-model 32 --replicas 2 --first-core 0
```

See [experiments/transformer_tailslayer/README.md](experiments/transformer_tailslayer/README.md) for Linux and Windows notes.

## CUDA VRAM experiment

`experiments/cuda_vram_tailslayer/` tests whether the same replicated-read idea helps transformer-like random gathers in NVIDIA VRAM. It measures random global-memory load tails with CUDA `clock64()` for a single table versus two replicated tables.

```powershell
python .\experiments\cuda_vram_tailslayer\vram_tail_bench.py --table-mb 128 --samples 200000 --repeats 5
```

This is an idealized VRAM latency test, not a full CUDA transformer. A lower `effective_min` tail only helps end-to-end inference or training if the CUDA kernel design can consume the first completed replica without paying more synchronization cost than it saves.

## Benchmarks and spike timing

The `discovery/` directory contains supporting code used to characterize DRAM refresh behavior:

- `discovery/benchmark/`: Channel-hedged read benchmark
- `discovery/trefi_probe.c`: Spike timing probe for measuring the refresh cycle

```bash
cd discovery/benchmark
make
sudo chrt -f 99 ./hedged_read_cpp --all --channel-bit 8
```
