#ifndef TAILSLAYER_HEDGED_READER_HPP
#define TAILSLAYER_HEDGED_READER_HPP

#include "platform.hpp"

#include <iostream>
#include <array>
#include <thread>
#include <atomic>
#include <cstdint>
#include <cassert>
#include <cstring>

namespace tailslayer {

inline constexpr int DEFAULT_CHANNEL_OFFSET = 256;
inline constexpr int DEFAULT_CHANNEL_BIT = 8;
inline constexpr int DEFAULT_NUM_CHANNELS = 2;
inline constexpr std::size_t DEFAULT_NUM_REPLICAS = 2;
inline constexpr std::uint64_t SUPERPAGE_SIZE = (1ULL << 30);

inline constexpr int CORE_MEAS_A = 11;
inline constexpr int CORE_MEAS_B = 12;
inline constexpr int CORE_MAIN = 14;

namespace detail {

#if defined(__x86_64__) || defined(__i386__) || \
    defined(_M_X64)     || defined(_M_IX86)
    using platform::timing::clflush_addr;
    using platform::timing::mfence_inst;
    using platform::timing::rdtsc_lfence;
    using platform::timing::rdtscp_lfence;
#endif
} // namespace detail

static inline int pin_to_core(int core_id) {
    return platform::pin_to_core(core_id);
}

template <auto... Vals>
struct ArgList {};

template <typename T, auto wait_work, auto final_work,
          typename WaitArgs = ArgList<>, typename WorkArgs = ArgList<>,
          std::size_t N = DEFAULT_NUM_REPLICAS>
class HedgedReader;

template <typename T, auto wait_work, auto final_work,
          auto... WaitArgs, auto... WorkArgs, std::size_t N>
class HedgedReader<T, wait_work, final_work, ArgList<WaitArgs...>, ArgList<WorkArgs...>, N> {
public:
    HedgedReader(int channel_offset = DEFAULT_CHANNEL_OFFSET,
                 int channel_bit = DEFAULT_CHANNEL_BIT,
                 std::size_t num_channels = DEFAULT_NUM_CHANNELS) :
                 channel_offset_(channel_offset), channel_bit_(channel_bit), num_channels_(num_channels), logical_index_(0) {

        assert(channel_offset_ % sizeof(T) == 0 && "Channel offset must be a multiple of sizeof(T)");
        assert(N <= num_channels_ && "Can't have more replicas than memory channels");

        std::size_t elements_per_chunk = channel_offset_ / sizeof(T);
        chunk_mask_ = elements_per_chunk - 1;
        chunk_shift_ = TS_CTZ64(elements_per_chunk);
        stride_in_elements_ = (num_channels_ * channel_offset_) / sizeof(T);
        std::size_t stride_bytes = num_channels_ * channel_offset_;
        std::size_t max_strides = SUPERPAGE_SIZE / stride_bytes;
        capacity_ = max_strides * elements_per_chunk;

        setup_memory();
        setup_replica_cores();
    }

    std::size_t size() const { return logical_index_; }
    std::size_t capacity() const { return capacity_; }

    void insert(T val) {
        assert(logical_index_ + 1 < capacity_ && "Tried to insert out of bounds");
        for (std::size_t i = 0; i < N; ++i) {
            T* target_addr = get_next_logical_index_address(i, logical_index_);
            *target_addr = val;
        }
        ++logical_index_;
    }

    void start_workers() {
        for (std::size_t i = 0; i < N; ++i) {
            workers_[i] = std::thread(&HedgedReader::worker_func, this, i);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    ~HedgedReader() {
        for (auto& t : workers_) {
            if (t.joinable()) t.join();
        }
        if (replica_page_) {
            platform::free_huge_page(replica_page_, SUPERPAGE_SIZE);
            replica_page_ = nullptr;
        }
    }

private:
    int channel_bit_;
    int channel_offset_;
    std::size_t num_channels_;
    void* replica_page_ = nullptr;
    std::size_t logical_index_;
    std::size_t capacity_;

    std::size_t chunk_shift_;
    std::size_t chunk_mask_;
    std::size_t stride_in_elements_;

    std::array<T*, N> replicas_{};
    std::array<int, N> cores_{};
    std::array<std::thread, N> workers_{};

    void worker_func(std::size_t worker_idx) {
        platform::pin_to_core(cores_[worker_idx]);

        std::size_t read_index = wait_work(WaitArgs...);

        T* target_addr = get_next_logical_index_address(worker_idx, read_index);
        final_work(*target_addr, WorkArgs...);
    }

    TS_FORCE_INLINE T* get_next_logical_index_address(std::size_t replica_idx,
                                                      std::size_t logical_index) const {
        std::size_t chunk_idx = logical_index >> chunk_shift_;
        std::size_t offset_in_chunk = logical_index & chunk_mask_;
        std::size_t element_offset = (chunk_idx * stride_in_elements_) + offset_in_chunk;
        return replicas_[replica_idx] + element_offset;
    }

    void setup_replica_cores() {
        cores_[0] = CORE_MEAS_A;
        if (num_channels_ > 1 && N > 1) {
            cores_[1] = CORE_MEAS_B;
        }
        for (std::size_t i = 2; i < N; ++i) {
            cores_[i] = CORE_MEAS_B + static_cast<int>(i) - 1;
        }
    }

    bool setup_memory() {
        replica_page_ = platform::alloc_huge_page(SUPERPAGE_SIZE);
        if (!replica_page_) {
            replica_page_ = nullptr;
            return false;
        }
        std::memset(replica_page_, 0x42, SUPERPAGE_SIZE);

        char* base = static_cast<char*>(replica_page_);
        for (std::size_t i = 0; i < N; ++i) {
            replicas_[i] = reinterpret_cast<T*>(base + (i * channel_offset_));
        }
        return true;
    }
};

} // namespace tailslayer

#endif // TAILSLAYER_HEDGED_READER_HPP
