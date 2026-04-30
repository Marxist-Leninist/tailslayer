# Transformer Tailslayer Experiment

This experiment applies the Tailslayer idea to a transformer-shaped CPU workload:

- one large replicated token embedding table laid out with the same channel-offset chunking as `HedgedReader`
- a tiny single-head attention block
- inference latency measured as single-worker vs duplicate-worker first response
- a training path that updates only the classifier head after the transformer read path

The training mode is intentionally conservative. Full transformer training has cross-replica gradient synchronization and writes, which are not what Tailslayer optimizes. This benchmark isolates the read-heavy transformer path and then shows the synchronization cost when a correct training update is added.

## Windows Build

Use a Visual Studio C++ environment or another CMake-supported Windows C++ compiler:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release --target transformer_tailslayer_bench
.\build\experiments\transformer_tailslayer\Release\transformer_tailslayer_bench.exe --mode all --iters 1000 --vocab 1048576 --seq-len 16 --d-model 32 --replicas 2 --first-core 0
```

If Windows cannot allocate large pages, the platform layer falls back to normal committed pages. That fallback is useful for testing correctness and duplicate-worker overhead, but it does not prove physical DRAM-channel placement.

## Linux Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target transformer_tailslayer_bench -j
sudo chrt -f 99 ./build/experiments/transformer_tailslayer/transformer_tailslayer_bench --mode all --iters 1000 --vocab 1048576 --seq-len 16 --d-model 32 --replicas 2 --first-core 11
```

For a hardware-valid Tailslayer run, use bare metal, enable huge pages, pick cores that are not busy, and choose the channel offset/bit discovered for that CPU. Virtualized hosts and normal pages can hide or destroy the effect.

## Reading The Output

The benchmark prints lines like:

```text
single_inference        mean_us=...
hedged_inference        mean_us=...
inference_mean_speedup=...
single_training         mean_us=...
hedged_training         mean_us=...
training_mean_speedup=...
```

Values greater than `1.0` are faster for the hedged path. For training, expect little or no speedup unless the read stalls dominate the whole step, because the update still has to be serialized.
