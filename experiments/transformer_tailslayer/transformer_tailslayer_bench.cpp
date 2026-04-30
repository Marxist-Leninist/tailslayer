#include <tailslayer/platform.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using clock_type = std::chrono::high_resolution_clock;

struct Config {
    std::string mode = "all";
    int iterations = 200;
    int warmup = 20;
    int vocab = 65536;
    int seq_len = 16;
    int d_model = 32;
    int replicas = 2;
    int channel_offset = 256;
    int first_core = 0;
    float learning_rate = 0.08f;
};

struct LatencyStats {
    double mean_us = 0.0;
    double p50_us = 0.0;
    double p95_us = 0.0;
    double p99_us = 0.0;
    double max_us = 0.0;
};

double elapsed_us(clock_type::time_point start, clock_type::time_point end) {
    return std::chrono::duration<double, std::micro>(end - start).count();
}

LatencyStats summarize(std::vector<double> values) {
    LatencyStats s{};
    if (values.empty()) return s;
    std::sort(values.begin(), values.end());
    s.mean_us = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    auto pct = [&](double p) {
        std::size_t idx = static_cast<std::size_t>(std::ceil(p * values.size())) - 1;
        idx = std::min(idx, values.size() - 1);
        return values[idx];
    };
    s.p50_us = pct(0.50);
    s.p95_us = pct(0.95);
    s.p99_us = pct(0.99);
    s.max_us = values.back();
    return s;
}

void print_stats(const std::string& name, const std::vector<double>& values) {
    LatencyStats s = summarize(values);
    std::cout << std::left << std::setw(24) << name
              << "mean_us=" << std::setw(10) << std::fixed << std::setprecision(2) << s.mean_us
              << "p50_us=" << std::setw(10) << s.p50_us
              << "p95_us=" << std::setw(10) << s.p95_us
              << "p99_us=" << std::setw(10) << s.p99_us
              << "max_us=" << s.max_us << "\n";
}

std::uint64_t splitmix64(std::uint64_t& x) {
    std::uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

float small_weight(std::uint64_t& state) {
    std::uint64_t bits = splitmix64(state);
    int raw = static_cast<int>((bits >> 32) & 0xffff);
    return (static_cast<float>(raw) / 32768.0f - 1.0f) * 0.02f;
}

template <typename T>
class ReplicatedArray {
public:
    ReplicatedArray(std::size_t logical_count, int replicas, int channel_offset)
        : logical_count_(logical_count), replicas_(replicas), channel_offset_(channel_offset) {
        if (replicas_ < 1) throw std::runtime_error("replicas must be >= 1");
        if (channel_offset_ < static_cast<int>(sizeof(T))) {
            throw std::runtime_error("channel offset must be at least sizeof(T)");
        }

        elements_per_chunk_ = static_cast<std::size_t>(channel_offset_) / sizeof(T);
        if ((elements_per_chunk_ & (elements_per_chunk_ - 1)) != 0) {
            throw std::runtime_error("channel offset / sizeof(T) must be a power of two");
        }

        chunk_mask_ = elements_per_chunk_ - 1;
        chunk_shift_ = tailslayer::platform::count_trailing_zeroes(elements_per_chunk_);
        stride_in_elements_ = static_cast<std::size_t>(replicas_) * elements_per_chunk_;
        chunks_ = (logical_count_ + elements_per_chunk_ - 1) / elements_per_chunk_;
        allocation_elements_ = chunks_ * stride_in_elements_;
        allocation_bytes_ = allocation_elements_ * sizeof(T);

        data_ = static_cast<T*>(tailslayer::platform::allocate_superpage(allocation_bytes_));
        if (data_ == nullptr) throw std::runtime_error("failed to allocate replicated array");
        std::memset(data_, 0, allocation_bytes_);
        tailslayer::platform::lock_memory(data_, allocation_bytes_);
    }

    ~ReplicatedArray() {
        tailslayer::platform::free_superpage(data_, allocation_bytes_);
    }

    ReplicatedArray(const ReplicatedArray&) = delete;
    ReplicatedArray& operator=(const ReplicatedArray&) = delete;

    T& at(int replica, std::size_t logical_index) {
        return data_[offset(replica, logical_index)];
    }

    const T& at(int replica, std::size_t logical_index) const {
        return data_[offset(replica, logical_index)];
    }

    void set_all(std::size_t logical_index, T value) {
        for (int r = 0; r < replicas_; ++r) {
            at(r, logical_index) = value;
        }
    }

    std::size_t bytes() const { return allocation_bytes_; }

private:
    std::size_t offset(int replica, std::size_t logical_index) const {
        std::size_t chunk_idx = logical_index >> chunk_shift_;
        std::size_t offset_in_chunk = logical_index & chunk_mask_;
        return chunk_idx * stride_in_elements_
             + static_cast<std::size_t>(replica) * elements_per_chunk_
             + offset_in_chunk;
    }

    std::size_t logical_count_ = 0;
    int replicas_ = 1;
    int channel_offset_ = 0;
    std::size_t elements_per_chunk_ = 0;
    std::size_t chunk_mask_ = 0;
    std::size_t chunk_shift_ = 0;
    std::size_t stride_in_elements_ = 0;
    std::size_t chunks_ = 0;
    std::size_t allocation_elements_ = 0;
    std::size_t allocation_bytes_ = 0;
    T* data_ = nullptr;
};

struct Example {
    std::vector<int> tokens;
    int label = 0;
};

struct ForwardOutput {
    float prob = 0.0f;
    std::vector<float> pooled;
};

struct Scratch {
    std::vector<float> x;
    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
    std::vector<float> attention;
    std::vector<float> scores;
};

class TinyTransformer {
public:
    explicit TinyTransformer(const Config& cfg)
        : cfg_(cfg),
          embedding_(static_cast<std::size_t>(cfg.vocab) * cfg.d_model,
                     cfg.replicas,
                     cfg.channel_offset),
          wq_(static_cast<std::size_t>(cfg.d_model) * cfg.d_model),
          wk_(static_cast<std::size_t>(cfg.d_model) * cfg.d_model),
          wv_(static_cast<std::size_t>(cfg.d_model) * cfg.d_model),
          classifier_(cfg.d_model, 0.0f) {
        init_weights();
    }

    Scratch make_scratch() const {
        Scratch s;
        std::size_t activations = static_cast<std::size_t>(cfg_.seq_len) * cfg_.d_model;
        s.x.resize(activations);
        s.q.resize(activations);
        s.k.resize(activations);
        s.v.resize(activations);
        s.attention.resize(activations);
        s.scores.resize(cfg_.seq_len);
        return s;
    }

    ForwardOutput make_output() const {
        ForwardOutput out;
        out.pooled.resize(cfg_.d_model);
        return out;
    }

    void forward(int replica, const Example& ex, Scratch& s, ForwardOutput& out) const {
        int d = cfg_.d_model;
        int n = cfg_.seq_len;

        for (int t = 0; t < n; ++t) {
            std::size_t row = static_cast<std::size_t>(ex.tokens[t]) * d;
            for (int j = 0; j < d; ++j) {
                s.x[static_cast<std::size_t>(t) * d + j] = embedding_.at(replica, row + j);
            }
        }

        project(s.x, wq_, s.q);
        project(s.x, wk_, s.k);
        project(s.x, wv_, s.v);

        float inv_scale = 1.0f / std::sqrt(static_cast<float>(d));
        for (int t = 0; t < n; ++t) {
            float max_score = -1.0e30f;
            for (int src = 0; src < n; ++src) {
                float dot = 0.0f;
                for (int j = 0; j < d; ++j) {
                    dot += s.q[static_cast<std::size_t>(t) * d + j]
                         * s.k[static_cast<std::size_t>(src) * d + j];
                }
                float score = dot * inv_scale;
                s.scores[src] = score;
                max_score = std::max(max_score, score);
            }

            float denom = 0.0f;
            for (int src = 0; src < n; ++src) {
                s.scores[src] = std::exp(s.scores[src] - max_score);
                denom += s.scores[src];
            }
            float inv_denom = 1.0f / denom;

            for (int j = 0; j < d; ++j) {
                float acc = 0.0f;
                for (int src = 0; src < n; ++src) {
                    acc += (s.scores[src] * inv_denom)
                         * s.v[static_cast<std::size_t>(src) * d + j];
                }
                s.attention[static_cast<std::size_t>(t) * d + j] = acc;
            }
        }

        std::fill(out.pooled.begin(), out.pooled.end(), 0.0f);
        for (int t = 0; t < n; ++t) {
            for (int j = 0; j < d; ++j) {
                out.pooled[j] += s.attention[static_cast<std::size_t>(t) * d + j];
            }
        }
        for (float& x : out.pooled) x /= static_cast<float>(n);

        float logit = bias_;
        for (int j = 0; j < d; ++j) logit += out.pooled[j] * classifier_[j];
        out.prob = 1.0f / (1.0f + std::exp(-logit));
    }

    float train_classifier(const ForwardOutput& out, int label) {
        float y = static_cast<float>(label);
        float grad = out.prob - y;
        for (int j = 0; j < cfg_.d_model; ++j) {
            classifier_[j] -= cfg_.learning_rate * grad * out.pooled[j];
        }
        bias_ -= cfg_.learning_rate * grad;
        float p = std::min(std::max(out.prob, 1.0e-6f), 1.0f - 1.0e-6f);
        return -(y * std::log(p) + (1.0f - y) * std::log(1.0f - p));
    }

    std::size_t replicated_embedding_bytes() const {
        return embedding_.bytes();
    }

private:
    void project(const std::vector<float>& input,
                 const std::vector<float>& weight,
                 std::vector<float>& output) const {
        int d = cfg_.d_model;
        int n = cfg_.seq_len;
        for (int t = 0; t < n; ++t) {
            for (int col = 0; col < d; ++col) {
                float acc = 0.0f;
                for (int row = 0; row < d; ++row) {
                    acc += input[static_cast<std::size_t>(t) * d + row]
                         * weight[static_cast<std::size_t>(row) * d + col];
                }
                output[static_cast<std::size_t>(t) * d + col] = acc;
            }
        }
    }

    void init_weights() {
        std::uint64_t state = 0x1234abcddcba4321ULL;
        for (int token = 0; token < cfg_.vocab; ++token) {
            for (int j = 0; j < cfg_.d_model; ++j) {
                float value = small_weight(state);
                if (j == 0) value += (token & 1) ? 1.0f : -1.0f;
                embedding_.set_all(static_cast<std::size_t>(token) * cfg_.d_model + j, value);
            }
        }

        std::fill(wq_.begin(), wq_.end(), 0.0f);
        std::fill(wk_.begin(), wk_.end(), 0.0f);
        std::fill(wv_.begin(), wv_.end(), 0.0f);
        for (int i = 0; i < cfg_.d_model; ++i) {
            wq_[static_cast<std::size_t>(i) * cfg_.d_model + i] = 1.0f;
            wk_[static_cast<std::size_t>(i) * cfg_.d_model + i] = 1.0f;
            wv_[static_cast<std::size_t>(i) * cfg_.d_model + i] = 1.0f;
        }
    }

    const Config& cfg_;
    ReplicatedArray<float> embedding_;
    std::vector<float> wq_;
    std::vector<float> wk_;
    std::vector<float> wv_;
    std::vector<float> classifier_;
    float bias_ = 0.0f;
};

std::vector<Example> make_dataset(const Config& cfg) {
    std::vector<Example> data(static_cast<std::size_t>(cfg.iterations + cfg.warmup));
    std::uint64_t state = 0xfeedfacedeadbeefULL;
    for (auto& ex : data) {
        ex.tokens.resize(cfg.seq_len);
        int odd_count = 0;
        for (int t = 0; t < cfg.seq_len; ++t) {
            int token = static_cast<int>(splitmix64(state) % static_cast<std::uint64_t>(cfg.vocab));
            ex.tokens[t] = token;
            odd_count += token & 1;
        }
        ex.label = odd_count > (cfg.seq_len / 2) ? 1 : 0;
    }
    return data;
}

class HedgedRunner {
public:
    HedgedRunner(const Config& cfg, const TinyTransformer& model, const std::vector<Example>& data)
        : cfg_(cfg), model_(model), data_(data), outputs_(cfg.replicas) {
        tailslayer::platform::pin_to_core(cfg_.first_core + cfg_.replicas);
        for (auto& out : outputs_) out = model_.make_output();
        for (int r = 0; r < cfg_.replicas; ++r) {
            workers_.emplace_back(&HedgedRunner::worker_loop, this, r);
        }
        while (ready_count_.load(std::memory_order_acquire) < cfg_.replicas) {
            tailslayer::platform::spin_pause();
        }
    }

    ~HedgedRunner() {
        stop_.store(true, std::memory_order_release);
        epoch_.fetch_add(1, std::memory_order_release);
        for (auto& worker : workers_) {
            if (worker.joinable()) worker.join();
        }
    }

    double run_once(int index, int& winner) {
        task_index_.store(index, std::memory_order_relaxed);
        done_count_.store(0, std::memory_order_relaxed);
        winner_.store(-1, std::memory_order_relaxed);
        start_ = clock_type::now();
        epoch_.fetch_add(1, std::memory_order_release);

        while ((winner = winner_.load(std::memory_order_acquire)) < 0) {
            tailslayer::platform::spin_pause();
        }
        double first_us = first_latency_us_.load(std::memory_order_acquire);

        while (done_count_.load(std::memory_order_acquire) < cfg_.replicas) {
            tailslayer::platform::spin_pause();
        }
        return first_us;
    }

    const ForwardOutput& output(int replica) const {
        return outputs_[replica];
    }

private:
    void worker_loop(int replica) {
        tailslayer::platform::pin_to_core(cfg_.first_core + replica);
        Scratch scratch = model_.make_scratch();
        std::uint64_t seen_epoch = epoch_.load(std::memory_order_acquire);
        ready_count_.fetch_add(1, std::memory_order_release);

        while (true) {
            std::uint64_t current_epoch = epoch_.load(std::memory_order_acquire);
            if (current_epoch == seen_epoch) {
                tailslayer::platform::spin_pause();
                continue;
            }
            seen_epoch = current_epoch;
            if (stop_.load(std::memory_order_acquire)) break;

            int index = task_index_.load(std::memory_order_relaxed);
            model_.forward(replica, data_[static_cast<std::size_t>(index)], scratch, outputs_[replica]);
            auto end = clock_type::now();

            int expected = -1;
            if (winner_.compare_exchange_strong(expected, -2, std::memory_order_acq_rel)) {
                first_latency_us_.store(elapsed_us(start_, end), std::memory_order_release);
                winner_.store(replica, std::memory_order_release);
            }
            done_count_.fetch_add(1, std::memory_order_release);
        }
    }

    const Config& cfg_;
    const TinyTransformer& model_;
    const std::vector<Example>& data_;
    std::vector<std::thread> workers_;
    std::vector<ForwardOutput> outputs_;
    std::atomic<std::uint64_t> epoch_{0};
    std::atomic<int> ready_count_{0};
    std::atomic<bool> stop_{false};
    std::atomic<int> task_index_{0};
    std::atomic<int> done_count_{0};
    std::atomic<int> winner_{-1};
    std::atomic<double> first_latency_us_{0.0};
    clock_type::time_point start_{};
};

std::vector<double> run_single_inference(const Config& cfg,
                                         const TinyTransformer& model,
                                         const std::vector<Example>& data) {
    tailslayer::platform::pin_to_core(cfg.first_core);
    Scratch scratch = model.make_scratch();
    ForwardOutput out = model.make_output();
    std::vector<double> latencies;
    latencies.reserve(cfg.iterations);

    for (int i = 0; i < cfg.iterations + cfg.warmup; ++i) {
        auto start = clock_type::now();
        model.forward(0, data[static_cast<std::size_t>(i)], scratch, out);
        auto end = clock_type::now();
        if (i >= cfg.warmup) latencies.push_back(elapsed_us(start, end));
    }
    return latencies;
}

std::vector<double> run_hedged_inference(const Config& cfg,
                                         const TinyTransformer& model,
                                         const std::vector<Example>& data) {
    HedgedRunner runner(cfg, model, data);
    std::vector<double> latencies;
    latencies.reserve(cfg.iterations);

    for (int i = 0; i < cfg.iterations + cfg.warmup; ++i) {
        int winner = -1;
        double first_us = runner.run_once(i, winner);
        (void)winner;
        if (i >= cfg.warmup) latencies.push_back(first_us);
    }
    return latencies;
}

std::vector<double> run_single_training(const Config& cfg,
                                        TinyTransformer& model,
                                        const std::vector<Example>& data,
                                        float& mean_loss) {
    tailslayer::platform::pin_to_core(cfg.first_core);
    Scratch scratch = model.make_scratch();
    ForwardOutput out = model.make_output();
    std::vector<double> latencies;
    latencies.reserve(cfg.iterations);
    double loss_sum = 0.0;

    for (int i = 0; i < cfg.iterations + cfg.warmup; ++i) {
        auto start = clock_type::now();
        model.forward(0, data[static_cast<std::size_t>(i)], scratch, out);
        float loss = model.train_classifier(out, data[static_cast<std::size_t>(i)].label);
        auto end = clock_type::now();
        if (i >= cfg.warmup) {
            latencies.push_back(elapsed_us(start, end));
            loss_sum += loss;
        }
    }
    mean_loss = static_cast<float>(loss_sum / std::max(1, cfg.iterations));
    return latencies;
}

std::vector<double> run_hedged_training(const Config& cfg,
                                        TinyTransformer& model,
                                        const std::vector<Example>& data,
                                        float& mean_loss) {
    HedgedRunner runner(cfg, model, data);
    std::vector<double> latencies;
    latencies.reserve(cfg.iterations);
    double loss_sum = 0.0;

    for (int i = 0; i < cfg.iterations + cfg.warmup; ++i) {
        int winner = -1;
        auto start = clock_type::now();
        runner.run_once(i, winner);
        float loss = model.train_classifier(runner.output(winner), data[static_cast<std::size_t>(i)].label);
        auto end = clock_type::now();
        if (i >= cfg.warmup) {
            latencies.push_back(elapsed_us(start, end));
            loss_sum += loss;
        }
    }
    mean_loss = static_cast<float>(loss_sum / std::max(1, cfg.iterations));
    return latencies;
}

void usage(const char *prog) {
    std::cerr
        << "Usage: " << prog << " [options]\n"
        << "  --mode all|inference|training\n"
        << "  --iters N\n"
        << "  --warmup N\n"
        << "  --vocab N\n"
        << "  --seq-len N\n"
        << "  --d-model N\n"
        << "  --replicas N\n"
        << "  --channel-offset BYTES\n"
        << "  --first-core N\n"
        << "  --learning-rate X\n";
}

Config parse_args(int argc, char **argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need_value = [&](const char *name) -> char* {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };

        if (arg == "--mode") cfg.mode = need_value("--mode");
        else if (arg == "--iters") cfg.iterations = std::stoi(need_value("--iters"));
        else if (arg == "--warmup") cfg.warmup = std::stoi(need_value("--warmup"));
        else if (arg == "--vocab") cfg.vocab = std::stoi(need_value("--vocab"));
        else if (arg == "--seq-len") cfg.seq_len = std::stoi(need_value("--seq-len"));
        else if (arg == "--d-model") cfg.d_model = std::stoi(need_value("--d-model"));
        else if (arg == "--replicas") cfg.replicas = std::stoi(need_value("--replicas"));
        else if (arg == "--channel-offset") cfg.channel_offset = std::stoi(need_value("--channel-offset"));
        else if (arg == "--first-core") cfg.first_core = std::stoi(need_value("--first-core"));
        else if (arg == "--learning-rate") cfg.learning_rate = std::stof(need_value("--learning-rate"));
        else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    return cfg;
}

void print_config(const Config& cfg, const TinyTransformer& model) {
    std::cout << "mode=" << cfg.mode
              << " iterations=" << cfg.iterations
              << " warmup=" << cfg.warmup
              << " vocab=" << cfg.vocab
              << " seq_len=" << cfg.seq_len
              << " d_model=" << cfg.d_model
              << " replicas=" << cfg.replicas
              << " channel_offset=" << cfg.channel_offset
              << " first_core=" << cfg.first_core
              << "\n";
    std::cout << "replicated_embedding_bytes=" << model.replicated_embedding_bytes() << "\n";
}

void compare_speedup(const char *name,
                     const std::vector<double>& single,
                     const std::vector<double>& hedged) {
    LatencyStats a = summarize(single);
    LatencyStats b = summarize(hedged);
    std::cout << name << "_mean_speedup=" << (a.mean_us / b.mean_us)
              << " " << name << "_p99_speedup=" << (a.p99_us / b.p99_us) << "\n";
}

} // namespace

int main(int argc, char **argv) {
    try {
        Config cfg = parse_args(argc, argv);
        if (cfg.mode != "all" && cfg.mode != "inference" && cfg.mode != "training") {
            throw std::runtime_error("--mode must be all, inference, or training");
        }

        auto data = make_dataset(cfg);

        if (cfg.mode == "all" || cfg.mode == "inference") {
            TinyTransformer model(cfg);
            print_config(cfg, model);
            auto single = run_single_inference(cfg, model, data);
            auto hedged = run_hedged_inference(cfg, model, data);
            print_stats("single_inference", single);
            print_stats("hedged_inference", hedged);
            compare_speedup("inference", single, hedged);
        }

        if (cfg.mode == "all" || cfg.mode == "training") {
            TinyTransformer single_model(cfg);
            TinyTransformer hedged_model(cfg);
            float single_loss = 0.0f;
            float hedged_loss = 0.0f;
            auto single = run_single_training(cfg, single_model, data, single_loss);
            auto hedged = run_hedged_training(cfg, hedged_model, data, hedged_loss);
            print_stats("single_training", single);
            print_stats("hedged_training", hedged);
            compare_speedup("training", single, hedged);
            std::cout << "single_training_mean_loss=" << single_loss
                      << " hedged_training_mean_loss=" << hedged_loss << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        usage(argv[0]);
        return 1;
    }

    return 0;
}
