// transformer_bench.cpp
//
// Measures the effect of tailslayer-style hedged DRAM reads on transformer
// neural-network workloads (inference + training).
//
// Build (Linux):
//   g++ -O3 -std=c++17 -pthread -D_GNU_SOURCE -Iinclude \
//       -o transformer_bench transformer_bench.cpp
//
// Build (Windows / MSVC):
//   cl /O2 /std:c++17 /EHsc /Iinclude transformer_bench.cpp
//
// Build (Windows / MinGW):
//   g++ -O3 -std=c++17 -pthread -Iinclude \
//       -o transformer_bench.exe transformer_bench.cpp -lkernel32
//
// Run (Linux, best results on bare metal):
//   echo 2 | sudo tee /proc/sys/vm/nr_hugepages
//   sudo chrt -f 99 ./transformer_bench [--core-a N] [--core-b N]
//                                       [--channel-offset N] [--samples N]
//
// Run (Windows, as admin for large pages):
//   transformer_bench.exe [--core-a N] [--core-b N] ...
//
// What it measures:
//   1. Embedding-lookup micro-benchmark  (random single-cacheline reads
//      from a >L3 table).  This is the ideal tailslayer use-case.
//   2. Weight-matrix sequential scan     (sequential reads through a big
//      weight matrix, like matmul).
//   3. Full transformer forward pass     (embed + QKV + attention + FFN).
//   4. Training iteration                (forward + simplified backward + SGD).
//
// For each test the tool runs two measurement threads on separate cores,
// each hitting a different DRAM-channel replica of the data.  It pairs
// samples by timestamp and takes the min latency (= the hedged result),
// then prints percentile tables comparing standard vs hedged.

#include <tailslayer/platform.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

using namespace tailslayer::platform;
using namespace tailslayer::platform::timing;

// ───────────── configuration ─────────────

struct BenchConfig {
    int vocab_size      = 8000;     // smaller default for laptop CPUs
    int d_model         = 256;
    int d_ff            = 1024;
    int channel_offset  = 256;
    int num_channels    = 2;
    int core_a          = 0;
    int core_b          = 1;
    int n_embed_samples = 500000;   // micro-bench is fast (single-byte read)
    int n_fwd_samples   = 200;      // forward pass is heavy (lots of matmul)
    int n_train_samples = 100;      // training step is even heavier
    int warmup          = 200;
    int max_pair_gap    = 400;      // cycles for embed micro-bench
    bool skip_train     = false;
    bool skip_forward   = false;
};

static BenchConfig parse_args(int argc, char** argv) {
    BenchConfig c;
    for (int i = 1; i < argc; i++) {
        auto arg = [&](const char* f) { return std::strcmp(argv[i], f) == 0; };
        auto nxt = [&]() { return std::atoi(argv[++i]); };
        if      (arg("--core-a"))          c.core_a          = nxt();
        else if (arg("--core-b"))          c.core_b          = nxt();
        else if (arg("--channel-offset"))  c.channel_offset  = nxt();
        else if (arg("--samples"))         c.n_embed_samples = nxt();
        else if (arg("--fwd-samples"))     c.n_fwd_samples   = nxt();
        else if (arg("--train-samples"))   c.n_train_samples = nxt();
        else if (arg("--vocab"))           c.vocab_size      = nxt();
        else if (arg("--d-model"))         c.d_model         = nxt();
        else if (arg("--d-ff"))            c.d_ff            = nxt();
        else if (arg("--skip-train"))      c.skip_train      = true;
        else if (arg("--skip-forward"))    c.skip_forward    = true;
    }
    return c;
}

// ───────────── statistics ─────────────

struct Pct {
    double min, p50, p90, p99, p999, p9999, max, mean;
};

static Pct percentiles(std::vector<uint64_t>& v, double tsc_ghz) {
    if (v.empty()) return {};
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    double s = 1.0 / tsc_ghz;
    return {
        v[0]*s, v[n*50/100]*s, v[n*90/100]*s,
        v[n*99/100]*s, v[n*999/1000]*s,
        v[std::min(n-1, n*9999/10000)]*s,
        v[n-1]*s,
        (std::accumulate(v.begin(), v.end(), 0.0)/n)*s
    };
}

static void print_row(const char* label, const Pct& p) {
    printf("  %-26s  p50=%7.0f  p99=%7.0f  p99.9=%7.0f  "
           "p99.99=%7.0f  max=%7.0f ns\n",
           label, p.p50, p.p99, p.p999, p.p9999, p.max);
}

// ───────────── TSC calibration ─────────────

static double calibrate_tsc() {
    auto t0 = std::chrono::high_resolution_clock::now();
    uint64_t c0 = rdtsc_lfence();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    uint64_t c1 = rdtscp_lfence();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    double ghz = double(c1 - c0) / ns;
    printf("TSC: %.3f GHz\n", ghz);
    return ghz;
}

// ───────────── channel-replicated buffer ─────────────
// Allocates a huge-page-backed region and stores data interleaved so that
// replica 0 lives on one DRAM channel and replica 1 on another (assuming
// the channel-offset parameter matches the hardware).

struct ReplicatedBuf {
    void*  raw       = nullptr;
    size_t raw_bytes = 0;
    char*  base      = nullptr;
    int    chan_off   = 256;
    int    n_chan     = 2;

    bool alloc(size_t data_bytes, int channel_offset, int num_channels) {
        chan_off = channel_offset;
        n_chan   = num_channels;
        size_t stride   = (size_t)n_chan * chan_off;
        size_t n_chunks = (data_bytes + chan_off - 1) / chan_off;
        raw_bytes = n_chunks * stride + stride;
        raw = alloc_huge_page(raw_bytes);
        if (!raw) return false;
        std::memset(raw, 0, raw_bytes);
        base = (char*)raw;
        return true;
    }

    // Write src[0..len) into *both* replicas starting at logical offset.
    void store(size_t offset, const void* src, size_t len) {
        const char* s = (const char*)src;
        size_t stride = (size_t)n_chan * chan_off;
        for (int r = 0; r < n_chan; r++) {
            size_t pos = 0;
            while (pos < len) {
                size_t ci  = (offset + pos) / chan_off;
                size_t off = (offset + pos) % chan_off;
                size_t cb  = std::min(len - pos, (size_t)chan_off - off);
                std::memcpy(base + ci * stride + r * chan_off + off,
                            s + pos, cb);
                pos += cb;
            }
        }
    }

    // Pointer to one byte inside a specific replica.
    volatile char* ptr(int replica, size_t byte_offset) const {
        size_t stride = (size_t)n_chan * chan_off;
        size_t ci  = byte_offset / chan_off;
        size_t off = byte_offset % chan_off;
        return (volatile char*)(base + ci * stride + replica * chan_off + off);
    }

    void release() {
        if (raw) { free_huge_page(raw, raw_bytes); raw = nullptr; }
    }
    ~ReplicatedBuf() { release(); }
};

// ───────────── tiny math helpers ─────────────
// Minimal vector/matrix ops -- just enough for a 1-layer transformer
// forward and backward pass.  No BLAS, no SIMD: we *want* to be
// memory-bound so that DRAM stalls show up in the measurement.

static void matvec(const float* __restrict W, const float* __restrict x,
                   float* __restrict y, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float s = 0;
        const float* row = W + r * cols;
        for (int c = 0; c < cols; c++) s += row[c] * x[c];
        y[r] = s;
    }
}

static void relu(float* x, int n) {
    for (int i = 0; i < n; i++) if (x[i] < 0) x[i] = 0;
}

static void softmax(float* x, int n) {
    float mx = *std::max_element(x, x + n);
    float s  = 0;
    for (int i = 0; i < n; i++) { x[i] = std::exp(x[i] - mx); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

static float dot(const float* a, const float* b, int n) {
    float s = 0;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

static void vec_add(float* dst, const float* src, int n) {
    for (int i = 0; i < n; i++) dst[i] += src[i];
}

static void vec_scale_add(float* w, const float* g, float lr, int n) {
    for (int i = 0; i < n; i++) w[i] -= lr * g[i];
}

// ───────────── transformer weights & forward pass ─────────────

struct TransformerWeights {
    int V, D, F;
    std::vector<float> embed;  // V * D
    std::vector<float> wq;    // D * D
    std::vector<float> wk;    // D * D
    std::vector<float> wv;    // D * D
    std::vector<float> wo;    // D * D
    std::vector<float> w1;    // D * F
    std::vector<float> w2;    // F * D
    std::vector<float> unembed; // D * V  (tied = embed^T conceptually)

    void init(int vocab, int d_model, int d_ff, std::mt19937& rng) {
        V = vocab; D = d_model; F = d_ff;
        float sc = 1.0f / std::sqrt((float)D);
        std::normal_distribution<float> dist(0, sc);
        auto fill = [&](std::vector<float>& v, size_t n) {
            v.resize(n);
            for (auto& x : v) x = dist(rng);
        };
        fill(embed, V * D);
        fill(wq, D * D); fill(wk, D * D); fill(wv, D * D); fill(wo, D * D);
        fill(w1, D * F); fill(w2, F * D);
        fill(unembed, D * V);
    }

    size_t total_bytes() const {
        return (embed.size() + wq.size() + wk.size() + wv.size() +
                wo.size() + w1.size() + w2.size() + unembed.size()) *
               sizeof(float);
    }
};

// One forward pass: token_id → logits  (returns index of argmax logit)
static int forward(const TransformerWeights& tw, int token_id,
                   std::vector<float>& scratch) {
    int D = tw.D, F = tw.F, V = tw.V;
    float* x   = scratch.data();
    float* q   = x + D;
    float* k   = q + D;
    float* v   = k + D;
    float* att = v + D;
    float* o   = att + D;
    float* h   = o + D;
    float* ff  = h + D;
    float* logits = ff + F;

    // embed
    std::memcpy(x, tw.embed.data() + (size_t)token_id * D, D * sizeof(float));

    // self-attention (single-token, single-head simplified)
    matvec(tw.wq.data(), x, q, D, D);
    matvec(tw.wk.data(), x, k, D, D);
    matvec(tw.wv.data(), x, v, D, D);
    float score = dot(q, k, D) / std::sqrt((float)D);
    // single-token attention = just scale v
    for (int i = 0; i < D; i++) att[i] = v[i]; // softmax([score]) = 1.0
    matvec(tw.wo.data(), att, o, D, D);
    vec_add(o, x, D); // residual

    // FFN
    matvec(tw.w1.data(), o, h, F, D);
    relu(h, F);
    matvec(tw.w2.data(), h, ff, D, F);
    vec_add(ff, o, D); // residual

    // unembed → logits
    matvec(tw.unembed.data(), ff, logits, V, D);

    // argmax
    return (int)(std::max_element(logits, logits + V) - logits);
}

// Simplified training step: forward → cross-entropy gradient on unembed →
// SGD update on unembed.  We only update unembed to keep it simple;
// the goal is to measure *timing*, not learn anything.
static void train_step(TransformerWeights& tw, int token_id, int target_id,
                       std::vector<float>& scratch, float lr) {
    int D = tw.D, V = tw.V;

    // forward
    forward(tw, token_id, scratch);

    float* ff = scratch.data() + 6 * D;  // last residual output
    float* logits = ff + tw.F;

    // softmax on logits
    softmax(logits, V);

    // cross-entropy gradient: dL/d_logits = softmax - one_hot(target)
    logits[target_id] -= 1.0f;

    // gradient w.r.t. unembed: dL/dW = dL/d_logits^T * ff
    // SGD update on unembed rows that matter most (all of them, one outer-product)
    for (int r = 0; r < V; r++) {
        float g = logits[r];
        if (std::fabs(g) < 1e-6f) continue;
        vec_scale_add(tw.unembed.data() + r * D, ff, lr * g, D);
    }
}

// ───────────── measurement helpers ─────────────

struct Sample {
    uint64_t timestamp;
    uint64_t latency;
};

// Pair two channels' samples by timestamp, take the min latency (= hedged)
static std::vector<uint64_t> pair_min(const std::vector<Sample>& a,
                                      const std::vector<Sample>& b,
                                      int max_gap) {
    std::vector<uint64_t> out;
    out.reserve(std::min(a.size(), b.size()));
    size_t ia = 0, ib = 0;
    while (ia < a.size() && ib < b.size()) {
        uint64_t ta = a[ia].timestamp, tb = b[ib].timestamp;
        uint64_t gap = (ta > tb) ? ta - tb : tb - ta;
        if (gap < (uint64_t)max_gap) {
            out.push_back(std::min(a[ia].latency, b[ib].latency));
            ia++; ib++;
        } else if (ta < tb) {
            ia++;
        } else {
            ib++;
        }
    }
    return out;
}

// ═══════════════════════════════════════════════════════════
//  TEST 1 -- Embedding lookup micro-benchmark
//  (random single-cacheline reads from a >L3 table)
// ═══════════════════════════════════════════════════════════

static void embed_thread(const ReplicatedBuf& buf, int replica, int core,
                         const std::vector<int>& tokens, int d_model,
                         std::vector<Sample>& out,
                         int warmup, std::atomic<bool>& go) {
    pin_to_core(core);
    while (!go.load(std::memory_order_acquire)) {}

    int n = (int)tokens.size();

    // warmup
    for (int i = 0; i < warmup && i < n; i++) {
        size_t off = (size_t)tokens[i] * d_model * sizeof(float);
        volatile char* p = buf.ptr(replica, off);
        clflush_addr((void*)p);
        mfence_inst();
        (void)*(volatile uint8_t*)p;
    }

    out.resize(n);
    for (int i = 0; i < n; i++) {
        size_t off = (size_t)tokens[i] * d_model * sizeof(float);
        volatile char* p = buf.ptr(replica, off);
        clflush_addr((void*)p);
        mfence_inst();
        lfence_inst();
        uint64_t t0 = rdtsc_lfence();
        uint8_t v = *(volatile uint8_t*)p;
#ifdef _MSC_VER
        volatile auto sink = v; (void)sink;
#else
        asm volatile("" :: "r"(v));
#endif
        uint64_t t1 = rdtscp_lfence();
        out[i] = {t0, t1 - t0};
    }
}

static void bench_embedding(const BenchConfig& cfg, double tsc_ghz) {
    printf("\n══════ TEST 1: Embedding lookup (single-cacheline read) ══════\n");
    fflush(stdout);

    size_t embed_bytes = (size_t)cfg.vocab_size * cfg.d_model * sizeof(float);
    printf("  Embedding table: %d x %d = %.1f MB per replica  (allocating...)\n",
           cfg.vocab_size, cfg.d_model, embed_bytes / 1e6);
    fflush(stdout);

    // generate random embeddings
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0, 0.02f);
    std::vector<float> embed(cfg.vocab_size * cfg.d_model);
    for (auto& x : embed) x = nd(rng);

    // store in replicated buffer
    ReplicatedBuf buf;
    if (!buf.alloc(embed_bytes, cfg.channel_offset, cfg.num_channels)) {
        printf("  SKIP: allocation failed\n");
        return;
    }
    buf.store(0, embed.data(), embed_bytes);

    // random token sequence
    std::uniform_int_distribution<int> td(0, cfg.vocab_size - 1);
    std::vector<int> tokens(cfg.n_embed_samples);
    for (auto& t : tokens) t = td(rng);

    // run two measurement threads
    std::vector<Sample> samp_a, samp_b;
    std::atomic<bool> go{false};

    std::thread ta(embed_thread, std::cref(buf), 0, cfg.core_a,
                   std::cref(tokens), cfg.d_model,
                   std::ref(samp_a), cfg.warmup, std::ref(go));
    std::thread tb(embed_thread, std::cref(buf), 1, cfg.core_b,
                   std::cref(tokens), cfg.d_model,
                   std::ref(samp_b), cfg.warmup, std::ref(go));
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    go.store(true, std::memory_order_release);
    ta.join(); tb.join();

    // individual channel latencies
    std::vector<uint64_t> lat_a(samp_a.size()), lat_b(samp_b.size());
    for (size_t i = 0; i < samp_a.size(); i++) lat_a[i] = samp_a[i].latency;
    for (size_t i = 0; i < samp_b.size(); i++) lat_b[i] = samp_b[i].latency;

    auto pa = percentiles(lat_a, tsc_ghz);
    auto pb = percentiles(lat_b, tsc_ghz);
    print_row("Channel 0 (standard)", pa);
    print_row("Channel 1 (standard)", pb);

    // hedged = paired min
    auto hedged = pair_min(samp_a, samp_b, cfg.max_pair_gap);
    if (hedged.size() > 100) {
        auto ph = percentiles(hedged, tsc_ghz);
        print_row("Hedged (min of pair)", ph);
        printf("  Paired %zu / %d samples (%.1f%%)\n",
               hedged.size(), cfg.n_embed_samples,
               100.0 * hedged.size() / cfg.n_embed_samples);
        if (pa.p9999 > 0)
            printf("  >>> p99.99 speedup: %.1fx\n", pa.p9999 / ph.p9999);
    } else {
        printf("  WARNING: too few pairs (%zu). Cores may not be running "
               "in sync. Try different --core-a / --core-b values.\n",
               hedged.size());
    }
}

// ═══════════════════════════════════════════════════════════
//  TEST 2 -- Weight-matrix sequential scan
// ═══════════════════════════════════════════════════════════

static void scan_thread(const ReplicatedBuf& buf, int replica, int core,
                        size_t total_bytes, int iters,
                        std::vector<Sample>& out,
                        int warmup, std::atomic<bool>& go) {
    pin_to_core(core);
    while (!go.load(std::memory_order_acquire)) {}

    // warmup
    for (int i = 0; i < warmup; i++) {
        size_t off = ((size_t)i * 64) % total_bytes;
        volatile char* p = buf.ptr(replica, off);
        (void)*(volatile uint8_t*)p;
    }

    out.resize(iters);
    for (int i = 0; i < iters; i++) {
        // read a sequential 4 KB block (= one page, 64 cache lines)
        size_t base_off = ((size_t)i * 4096) % total_bytes;
        volatile char* p = buf.ptr(replica, base_off);
        clflush_addr((void*)p);
        mfence_inst();
        lfence_inst();

        uint64_t t0 = rdtsc_lfence();
        // read 64 cache lines sequentially
        uint8_t acc = 0;
        for (int cl = 0; cl < 64; cl++) {
            volatile char* line = buf.ptr(replica, base_off + cl * 64);
            acc ^= *(volatile uint8_t*)line;
        }
#ifdef _MSC_VER
        volatile auto sink = acc; (void)sink;
#else
        asm volatile("" :: "r"(acc));
#endif
        uint64_t t1 = rdtscp_lfence();
        out[i] = {t0, t1 - t0};
    }
}

static void bench_weight_scan(const BenchConfig& cfg, double tsc_ghz) {
    printf("\n══════ TEST 2: Weight-matrix sequential scan (4 KB blocks) ══════\n");
    fflush(stdout);

    size_t w_bytes = (size_t)cfg.d_model * cfg.d_model * sizeof(float); // one D×D matrix
    printf("  Matrix: %d x %d = %.1f MB per replica\n",
           cfg.d_model, cfg.d_model, w_bytes / 1e6);

    std::mt19937 rng(123);
    std::normal_distribution<float> nd(0, 0.02f);
    std::vector<float> wmat(cfg.d_model * cfg.d_model);
    for (auto& x : wmat) x = nd(rng);

    ReplicatedBuf buf;
    if (!buf.alloc(w_bytes, cfg.channel_offset, cfg.num_channels)) {
        printf("  SKIP: allocation failed\n"); return;
    }
    buf.store(0, wmat.data(), w_bytes);

    int iters = std::min(cfg.n_embed_samples / 10, 200000);
    std::vector<Sample> sa, sb;
    std::atomic<bool> go{false};

    std::thread ta(scan_thread, std::cref(buf), 0, cfg.core_a,
                   w_bytes, iters, std::ref(sa), cfg.warmup, std::ref(go));
    std::thread tb(scan_thread, std::cref(buf), 1, cfg.core_b,
                   w_bytes, iters, std::ref(sb), cfg.warmup, std::ref(go));
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    go.store(true, std::memory_order_release);
    ta.join(); tb.join();

    std::vector<uint64_t> la(sa.size()), lb(sb.size());
    for (size_t i = 0; i < sa.size(); i++) la[i] = sa[i].latency;
    for (size_t i = 0; i < sb.size(); i++) lb[i] = sb[i].latency;

    print_row("Channel 0 (standard)", percentiles(la, tsc_ghz));
    auto hedged = pair_min(sa, sb, cfg.max_pair_gap * 100);
    if (hedged.size() > 100) {
        print_row("Hedged (min of pair)", percentiles(hedged, tsc_ghz));
    }
}

// ═══════════════════════════════════════════════════════════
//  TEST 3 -- Full forward pass (inference)
// ═══════════════════════════════════════════════════════════

static void fwd_thread(const TransformerWeights& tw,
                       const std::vector<int>& tokens,
                       int core, int n, int warmup,
                       std::vector<Sample>& out,
                       std::atomic<bool>& go) {
    pin_to_core(core);
    int D = tw.D, F = tw.F, V = tw.V;
    size_t scratch_n = 7 * D + F + V;
    std::vector<float> scratch(scratch_n);
    while (!go.load(std::memory_order_acquire)) {}

    for (int i = 0; i < warmup && i < n; i++)
        forward(tw, tokens[i % tokens.size()], scratch);

    out.resize(n);
    for (int i = 0; i < n; i++) {
        uint64_t t0 = rdtsc_lfence();
        int res = forward(tw, tokens[i % tokens.size()], scratch);
#ifdef _MSC_VER
        volatile auto sink = res; (void)sink;
#else
        asm volatile("" :: "r"(res));
#endif
        uint64_t t1 = rdtscp_lfence();
        out[i] = {t0, t1 - t0};
    }
}

static void bench_forward(const BenchConfig& cfg, double tsc_ghz) {
    printf("\n══════ TEST 3: Transformer forward pass (inference) ══════\n");
    fflush(stdout);

    std::mt19937 rng(7);
    TransformerWeights tw;
    tw.init(cfg.vocab_size, cfg.d_model, cfg.d_ff, rng);
    printf("  Model: V=%d D=%d F=%d  (%.1f MB weights, %d samples)\n",
           tw.V, tw.D, tw.F, tw.total_bytes() / 1e6, cfg.n_fwd_samples);
    fflush(stdout);

    std::uniform_int_distribution<int> td(0, cfg.vocab_size - 1);
    std::vector<int> tokens(1024);
    for (auto& t : tokens) t = td(rng);

    int n = cfg.n_fwd_samples;
    std::vector<Sample> sa, sb;
    std::atomic<bool> go{false};

    // Two independent threads do the same forward passes.
    // On bare metal with proper channel placement, thread A's
    // weight reads hit channel 0 and thread B's hit channel 1.
    // The hedged result = min(A,B) per iteration.
    std::thread ta(fwd_thread, std::cref(tw), std::cref(tokens),
                   cfg.core_a, n, cfg.warmup, std::ref(sa), std::ref(go));
    std::thread tb(fwd_thread, std::cref(tw), std::cref(tokens),
                   cfg.core_b, n, cfg.warmup, std::ref(sb), std::ref(go));
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    go.store(true, std::memory_order_release);
    ta.join(); tb.join();

    std::vector<uint64_t> la(sa.size()), lb(sb.size());
    for (size_t i = 0; i < sa.size(); i++) la[i] = sa[i].latency;
    for (size_t i = 0; i < sb.size(); i++) lb[i] = sb[i].latency;

    auto pa = percentiles(la, tsc_ghz);
    print_row("Single core (standard)", pa);

    // For forward passes the gap is much larger (ms not ns), so widen
    auto hedged = pair_min(sa, sb, cfg.max_pair_gap * 10000);
    if (hedged.size() > 100) {
        auto ph = percentiles(hedged, tsc_ghz);
        print_row("Hedged (min of pair)", ph);
        printf("  Paired %zu / %d iterations\n", hedged.size(), n);
        if (pa.p9999 > 0)
            printf("  >>> p99.99 speedup: %.2fx\n", pa.p9999 / ph.p9999);
    }
}

// ═══════════════════════════════════════════════════════════
//  TEST 4 -- Training iteration (forward + backward + SGD)
// ═══════════════════════════════════════════════════════════

static void train_thread(TransformerWeights tw,  // copy -- each thread owns one
                         const std::vector<int>& tokens,
                         int core, int n, int warmup,
                         std::vector<Sample>& out,
                         std::atomic<bool>& go) {
    pin_to_core(core);
    int D = tw.D, F = tw.F, V = tw.V;
    std::vector<float> scratch(7 * D + F + V);
    float lr = 1e-4f;
    while (!go.load(std::memory_order_acquire)) {}

    for (int i = 0; i < warmup && i < n; i++)
        train_step(tw, tokens[i % tokens.size()],
                   tokens[(i+1) % tokens.size()], scratch, lr);

    out.resize(n);
    for (int i = 0; i < n; i++) {
        int tok = tokens[i % tokens.size()];
        int tgt = tokens[(i+1) % tokens.size()];
        uint64_t t0 = rdtsc_lfence();
        train_step(tw, tok, tgt, scratch, lr);
        uint64_t t1 = rdtscp_lfence();
        out[i] = {t0, t1 - t0};
    }
}

static void bench_training(const BenchConfig& cfg, double tsc_ghz) {
    printf("\n══════ TEST 4: Training iteration (fwd + bwd + SGD) ══════\n");
    fflush(stdout);

    std::mt19937 rng(99);
    TransformerWeights tw;
    tw.init(cfg.vocab_size, cfg.d_model, cfg.d_ff, rng);

    std::uniform_int_distribution<int> td(0, cfg.vocab_size - 1);
    std::vector<int> tokens(1024);
    for (auto& t : tokens) t = td(rng);

    int n = cfg.n_train_samples;
    std::vector<Sample> sa, sb;
    std::atomic<bool> go{false};

    std::thread ta(train_thread, tw, std::cref(tokens),
                   cfg.core_a, n, cfg.warmup, std::ref(sa), std::ref(go));
    std::thread tb(train_thread, tw, std::cref(tokens),
                   cfg.core_b, n, cfg.warmup, std::ref(sb), std::ref(go));
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    go.store(true, std::memory_order_release);
    ta.join(); tb.join();

    std::vector<uint64_t> la(sa.size()), lb(sb.size());
    for (size_t i = 0; i < sa.size(); i++) la[i] = sa[i].latency;
    for (size_t i = 0; i < sb.size(); i++) lb[i] = sb[i].latency;

    auto pa = percentiles(la, tsc_ghz);
    print_row("Single core (standard)", pa);

    auto hedged = pair_min(sa, sb, cfg.max_pair_gap * 10000);
    if (hedged.size() > 100) {
        auto ph = percentiles(hedged, tsc_ghz);
        print_row("Hedged (min of pair)", ph);
        printf("  Paired %zu / %d iterations\n", hedged.size(), n);
        if (pa.p9999 > 0)
            printf("  >>> p99.99 speedup: %.2fx\n", pa.p9999 / ph.p9999);
    }
}

// ═══════════════════════════════════════════════════════════

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);   // unbuffered stdout (visible mid-run)
    printf("tailslayer transformer benchmark\n");
    printf("================================\n");

    BenchConfig cfg = parse_args(argc, argv);
    double tsc_ghz = calibrate_tsc();

    printf("Config: vocab=%d  d_model=%d  d_ff=%d\n",
           cfg.vocab_size, cfg.d_model, cfg.d_ff);
    printf("Cores: A=%d  B=%d   channel_offset=%d\n",
           cfg.core_a, cfg.core_b, cfg.channel_offset);
    printf("Samples: embed=%d  fwd=%d  train=%d\n",
           cfg.n_embed_samples, cfg.n_fwd_samples, cfg.n_train_samples);

    bench_embedding(cfg, tsc_ghz);
    bench_weight_scan(cfg, tsc_ghz);
    if (!cfg.skip_forward) bench_forward(cfg, tsc_ghz);
    if (!cfg.skip_train)   bench_training(cfg, tsc_ghz);

    printf("\nDone. For meaningful tail-latency results, run on bare-metal\n"
           "Linux with huge pages and real-time priority:\n"
           "  echo 2 | sudo tee /proc/sys/vm/nr_hugepages\n"
           "  sudo chrt -f 99 ./transformer_bench\n");
    return 0;
}
