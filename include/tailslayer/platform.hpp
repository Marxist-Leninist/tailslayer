#ifndef TAILSLAYER_PLATFORM_HPP
#define TAILSLAYER_PLATFORM_HPP

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>

#if defined(_MSC_VER)
#define TAILSLAYER_ALWAYS_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define TAILSLAYER_ALWAYS_INLINE [[gnu::always_inline]] inline
#else
#define TAILSLAYER_ALWAYS_INLINE inline
#endif

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <intrin.h>
#include <windows.h>
#else
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace tailslayer::platform {

inline unsigned count_trailing_zeroes(std::uint64_t value) {
    if (value == 0) return 64;

#if defined(_MSC_VER) && defined(_M_X64)
    unsigned long idx = 0;
    _BitScanForward64(&idx, value);
    return static_cast<unsigned>(idx);
#elif defined(_MSC_VER)
    unsigned long idx = 0;
    if (static_cast<std::uint32_t>(value) != 0) {
        _BitScanForward(&idx, static_cast<unsigned long>(value));
        return static_cast<unsigned>(idx);
    }
    _BitScanForward(&idx, static_cast<unsigned long>(value >> 32));
    return static_cast<unsigned>(idx + 32);
#elif defined(__GNUC__) || defined(__clang__)
    return static_cast<unsigned>(__builtin_ctzll(value));
#else
    unsigned count = 0;
    while ((value & 1ULL) == 0) {
        value >>= 1;
        ++count;
    }
    return count;
#endif
}

inline void sleep_ms(unsigned milliseconds) {
#if defined(_WIN32)
    Sleep(milliseconds);
#else
    usleep(milliseconds * 1000);
#endif
}

TAILSLAYER_ALWAYS_INLINE void spin_pause() {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    _mm_pause();
#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
    asm volatile("pause" ::: "memory");
#elif defined(_WIN32)
    YieldProcessor();
#else
    std::this_thread::yield();
#endif
}

inline int pin_to_core(int core_id) {
#if defined(_WIN32)
    if (core_id < 0) return -1;

    constexpr int bits_per_group = static_cast<int>(sizeof(DWORD_PTR) * 8);
    if (core_id >= bits_per_group) {
        std::fprintf(stderr,
                     "tailslayer: Windows fallback pin_to_core only handles cores 0-%d without processor groups\n",
                     bits_per_group - 1);
        return -1;
    }

    DWORD_PTR mask = static_cast<DWORD_PTR>(1) << core_id;
    return SetThreadAffinityMask(GetCurrentThread(), mask) ? 0 : -1;
#else
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    return sched_setaffinity(0, sizeof(cpuset), &cpuset);
#endif
}

inline void *allocate_superpage(std::size_t bytes) {
#if defined(_WIN32)
    void *ptr = nullptr;

    SIZE_T large_page_min = GetLargePageMinimum();
    if (large_page_min != 0 && (bytes % large_page_min) == 0) {
        ptr = VirtualAlloc(nullptr, bytes,
                           MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES,
                           PAGE_READWRITE);
        if (ptr != nullptr) return ptr;
    }

    std::fprintf(stderr,
                 "tailslayer: large-page VirtualAlloc failed or is unavailable; falling back to normal committed pages.\n"
                 "tailslayer: this fallback is useful for functional tests, but it does not prove DRAM-channel placement.\n");
    return VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
#ifndef MAP_HUGE_SHIFT
#define MAP_HUGE_SHIFT 26
#endif
    void *ptr = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | (30 << MAP_HUGE_SHIFT),
                     -1, 0);
    if (ptr == MAP_FAILED) {
        std::fprintf(stderr,
                     "tailslayer: 1GB huge-page mmap failed; falling back to normal anonymous pages.\n"
                     "tailslayer: this fallback is useful for functional tests, but it does not prove DRAM-channel placement.\n");
        ptr = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    }
    if (ptr == MAP_FAILED) return nullptr;
    return ptr;
#endif
}

inline void free_superpage(void *ptr, std::size_t bytes) {
    if (ptr == nullptr) return;

#if defined(_WIN32)
    (void)bytes;
    VirtualFree(ptr, 0, MEM_RELEASE);
#else
    munmap(ptr, bytes);
#endif
}

inline int lock_memory(void *ptr, std::size_t bytes) {
#if defined(_WIN32)
    return VirtualLock(ptr, bytes) ? 0 : -1;
#else
    return mlock(ptr, bytes);
#endif
}

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
TAILSLAYER_ALWAYS_INLINE void clflush_addr(void *addr) {
#if defined(_MSC_VER)
    _mm_clflush(addr);
#else
    asm volatile("clflush (%0)" :: "r"(addr) : "memory");
#endif
}

TAILSLAYER_ALWAYS_INLINE void mfence_inst() {
#if defined(_MSC_VER)
    _mm_mfence();
#else
    asm volatile("mfence" ::: "memory");
#endif
}

TAILSLAYER_ALWAYS_INLINE void lfence_inst() {
#if defined(_MSC_VER)
    _mm_lfence();
#else
    asm volatile("lfence" ::: "memory");
#endif
}

TAILSLAYER_ALWAYS_INLINE std::uint64_t rdtsc_lfence() {
#if defined(_MSC_VER)
    _mm_lfence();
    return __rdtsc();
#else
    std::uint64_t lo, hi;
    asm volatile("lfence\n\t"
                 "rdtsc"
                 : "=a"(lo), "=d"(hi));
    return (hi << 32) | lo;
#endif
}

TAILSLAYER_ALWAYS_INLINE std::uint64_t rdtscp_lfence() {
#if defined(_MSC_VER)
    unsigned int aux = 0;
    std::uint64_t t = __rdtscp(&aux);
    _mm_lfence();
    return t;
#else
    std::uint64_t lo, hi;
    std::uint32_t aux;
    asm volatile("rdtscp"
                 : "=a"(lo), "=d"(hi), "=c"(aux));
    asm volatile("lfence" ::: "memory");
    return (hi << 32) | lo;
#endif
}
#else
TAILSLAYER_ALWAYS_INLINE void clflush_addr(void *) {}
TAILSLAYER_ALWAYS_INLINE void mfence_inst() {
    std::atomic_thread_fence(std::memory_order_seq_cst);
}
TAILSLAYER_ALWAYS_INLINE void lfence_inst() {
    std::atomic_thread_fence(std::memory_order_seq_cst);
}
TAILSLAYER_ALWAYS_INLINE std::uint64_t rdtsc_lfence() {
    return static_cast<std::uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
}
TAILSLAYER_ALWAYS_INLINE std::uint64_t rdtscp_lfence() {
    return rdtsc_lfence();
}
#endif

} // namespace tailslayer::platform

#endif // TAILSLAYER_PLATFORM_HPP
