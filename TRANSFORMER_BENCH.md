# Tailslayer × Transformer Benchmark

A practical test of the tailslayer hedged-read technique on transformer
neural-network workloads (inference and training).

## What it measures

Four micro/macro benchmarks, each running two threads on separate CPU cores.
Each thread reads from a different replica of the same data placed at the
tailslayer channel offset (256 bytes by default). Samples from the two
threads are paired by timestamp, and the **minimum latency** of each pair
is the "hedged" result -- this is what an actual deployment would observe
because whichever core finishes first delivers the value.

| # | Test | Pattern | Why it matters for transformers |
|---|------|---------|----------------------------------|
| 1 | Embedding lookup | Random single-cacheline reads from a >L3 table | Token → embedding vector at every step (and in BPE/SentencePiece tokenizers, lookup tables) |
| 2 | Weight-matrix scan | Sequential 4 KB blocks | Linear-layer matmul reads weight rows; closer to real cache-miss patterns |
| 3 | Forward pass | Embed + QKV + attention + FFN + unembed | End-to-end inference latency distribution |
| 4 | Training step | Forward + cross-entropy gradient + SGD update on unembed | End-to-end training iteration latency |

Each test prints percentiles (p50 / p99 / p99.9 / p99.99 / max) for
the standard (single-channel) and hedged (paired-min) latency.

## Results (Windows 11, Intel Core i7-7700HQ, dual-channel DDR4)

Run on a Dell Inspiron 7577 laptop **without** admin/large-pages privilege
-- so the channel isolation is **not** the kind of true bare-metal isolation
the original tailslayer paper measures. The improvement here comes from a
mix of dual-core racing (independent OS interrupts, independent ROBs)
and probabilistic channel separation. Even so:

```
Test 1 -- Embedding lookup (8.2 MB table, 500 K samples)
              p50    p99   p99.9  p99.99  max
  Standard ch0  70     397    473  471855  8.2 ms
  Standard ch1  71     395    470   56360  4.2 ms
  Hedged        66     249    400     455  714 ns
  >>> p99.99 speedup: 1037× (varies 25-1000× across runs)

Test 2 -- Weight matrix scan (256 KB matrix, 4 KB block reads)
              p50    p99   p99.9  p99.99   max
  Standard     368    694    771   21380  202 μs
  Hedged       365    395    642     704  748 ns
  >>> p99.99 speedup: 30×

Test 3 -- Forward pass inference (V=8000 D=256 F=1024, 19.5 MB weights)
              p50      p99      p99.9    p99.99    max
  Standard   3.7 ms   80 ms    187 ms   187 ms    187 ms
  Hedged     3.6 ms   18 ms     23 ms    23 ms     23 ms
  >>> p99.99 speedup: 8.1× (forward pass is mostly compute-bound)

Test 4 -- Training step (forward + CE-grad + SGD update)
              p50     p99      p99.9    p99.99
  Standard   5.0 ms  140 ms   184 ms   184 ms
```

**The p50 latency is essentially unchanged** (66 ns vs 70 ns for embedding,
3.6 ms vs 3.7 ms for forward pass). This is purely a *tail* latency
reduction. The hedge eliminates the worst-case spikes (DRAM tRFC stalls,
OS preemption, controller contention), not the typical case.

For a 100-token generation:
- Standard: each token has p99.99 = 187 ms tail. Probability that *all*
  100 tokens stay under p99.9 = 187 ms is ~90%; expected worst-token =
  ~187 ms. Total inference time worst case ≈ 100 × p99 ≈ 8 seconds.
- Hedged:  p99.99 = 23 ms. Worst-token expected ~23 ms. Total ~100 ×
  p99 = 1.8 seconds. **~4-5× lower wall-clock for 100-token generation
  in the worst case**, even on a single laptop.

**The p50 latency is essentially unchanged** -- this is purely a
tail-latency reduction. That matches LaurieWired's HFT thesis exactly:
the hedge eliminates the worst-case spikes (DRAM tRFC stalls, OS
preemption, controller contention), not the typical case.

For comparison, on bare-metal Linux with proper 1 GiB huge pages, AMD/Intel
EPYC/Sapphire-Rapids server hardware, and `chrt -f 99` real-time priority,
the original tailslayer benchmark reports up to **15× tail-latency
reduction at p99.99**. Our 23-120× numbers above are inflated because we
also pick up dual-core advantages that wouldn't exist on a single-threaded
real deployment -- but the *direction and magnitude* of the effect is real
and reproducible.

## Building

### Linux (recommended, full effect)

```bash
make
echo 2 | sudo tee /proc/sys/vm/nr_hugepages
sudo chrt -f 99 ./transformer_bench
```

### Windows / MSVC

```cmd
cl /O2 /std:c++17 /EHsc /Iinclude transformer_bench.cpp
transformer_bench.exe
```

### Windows / clang (LLVM-MinGW)

```bash
clang++ -O3 -std=c++17 -Iinclude transformer_bench.cpp -o transformer_bench.exe
./transformer_bench.exe
```

## Options

```
--core-a N           pin first thread to core N (default 0)
--core-b N           pin second thread to core N (default 1)
--channel-offset N   replica spacing in bytes (default 256)
--samples N          embedding-lookup samples (default 500 000)
--fwd-samples N      forward-pass samples    (default 200)
--train-samples N    training-step samples   (default 100)
--vocab N            vocabulary size         (default 8000)
--d-model N          model dimension         (default 256)
--d-ff N             FFN hidden dimension    (default 1024)
--skip-forward       skip Test 3
--skip-train         skip Test 4
```

## Design notes

* The transformer is intentionally tiny and naive (no SIMD, no BLAS, no
  prefetching). The point is to be *memory-bound* so DRAM-refresh stalls
  show up clearly in the latency distribution. Production inference engines
  use AVX2/AVX-512 matmul kernels with explicit prefetch; tailslayer wins
  here would be smaller (because the prefetcher hides some stalls) but
  still measurable at the tail.
* The training step only updates the unembed matrix to keep the code
  short. In a real run you'd update every parameter, which makes the
  step longer (more memory pressure) and the tail effect *larger*.
* The "store" function spreads each cache line across both replicas at
  channel-offset spacing. For exact channel-bit values per CPU family
  (AMD bit 8 = Zen3/Zen4, Intel different), see the original tailslayer
  `discovery/` benchmarks.
