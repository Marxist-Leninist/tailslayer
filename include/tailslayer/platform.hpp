#ifndef TAILSLAYER_PLATFORM_HPP
#define TAILSLAYER_PLATFORM_HPP

// Cross-platform abstraction for tailslayer.
// Linux: mmap + MAP_HUGETLB (1 GiB pages), sched_setaffinity, inline asm
// Windows: VirtualAlloc + MEM_LARGE_PAGES (2 MiB), SetThreadAffinityMask, intrinsics

#include <cstdint>
#include <cstdio>
#include <cstring>

#ifdef _MSC_VER
#define TS_FORCE_INLINE __forceinline
#else
#define TS_FORCE_INLINE [[gnu::always_inline]] inline
#endif

#ifdef _MSC_VER
#define TS_CTZ64(x) _tzcnt_u64(x)
#else
#define TS_CTZ64(x) __builtin_ctzll(x)
#endif

// ──────────────────────── Windows ────────────────────────
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <intrin.h>
#include <immintrin.h>

namespace tailslayer { namespace platform {

inline bool enable_large_page_privilege() {
    HANDLE token;
    if (!OpenProcessToken(GetCurrentProcess(),
                          TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &token))
        return false;
    TOKEN_PRIVILEGES tp{};
    tp.PrivilegeCount = 1;
    tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
    if (!LookupPrivilegeValue(NULL, SE_LOCK_MEMORY_NAME,
                              &tp.Privileges[0].Luid)) {
        CloseHandle(token);
        return false;
    }
    BOOL ok = AdjustTokenPrivileges(token, FALSE, &tp, sizeof(tp), NULL, NULL);
    DWORD err = GetLastError();
    CloseHandle(token);
    return ok && (err == ERROR_SUCCESS);
}

inline void* alloc_huge_page(std::uint64_t size) {
    if (enable_large_page_privilege()) {
        SIZE_T lp = GetLargePageMinimum();
        if (lp) {
            SIZE_T alloc = (size + lp - 1) & ~(lp - 1);
            void* p = VirtualAlloc(NULL, alloc,
                MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES, PAGE_READWRITE);
            if (p) {
                fprintf(stderr, "[tailslayer] %llu bytes via large pages "
                        "(%llu each)\n",
                        (unsigned long long)alloc, (unsigned long long)lp);
                return p;
            }
        }
    }
    void* p = VirtualAlloc(NULL, size, MEM_COMMIT | MEM_RESERVE,
                           PAGE_READWRITE);
    if (p) {
        fprintf(stderr, "[tailslayer] WARNING: large pages unavailable, "
                        "using regular pages. Channel isolation not "
                        "guaranteed.\n");
        VirtualLock(p, size);
    }
    return p;
}

inline void free_huge_page(void* ptr, std::uint64_t /*size*/) {
    if (ptr) VirtualFree(ptr, 0, MEM_RELEASE);
}

inline int pin_to_core(int core_id) {
    DWORD_PTR mask = 1ULL << core_id;
    return SetThreadAffinityMask(GetCurrentThread(), mask) ? 0 : -1;
}

inline void sleep_us(unsigned us) {
    Sleep((us + 999) / 1000);
}

}} // namespace tailslayer::platform

// ──────────────────────── Linux / POSIX ────────────────────────
#else
#include <sys/mman.h>
#include <sched.h>
#include <unistd.h>

namespace tailslayer { namespace platform {

inline void* alloc_huge_page(std::uint64_t size) {
    void* p = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB |
                   (30 << MAP_HUGE_SHIFT), -1, 0);
    if (p == MAP_FAILED) {
        p = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB |
                 (21 << MAP_HUGE_SHIFT), -1, 0);
    }
    if (p == MAP_FAILED) {
        perror("[tailslayer] mmap hugepage");
        return nullptr;
    }
    mlock(p, size);
    return p;
}

inline void free_huge_page(void* ptr, std::uint64_t size) {
    if (ptr) munmap(ptr, size);
}

inline int pin_to_core(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    return sched_setaffinity(0, sizeof(cpuset), &cpuset);
}

inline void sleep_us(unsigned us) {
    usleep(us);
}

}} // namespace tailslayer::platform
#endif // _WIN32

// ──────────────────────── Cross-platform timing ────────────────────────
namespace tailslayer { namespace platform { namespace timing {

#if (defined(__x86_64__) || defined(_M_X64) || \
     defined(__i386__)   || defined(_M_IX86))

#ifdef _MSC_VER
    TS_FORCE_INLINE std::uint64_t rdtsc_lfence() {
        _mm_lfence();
        return __rdtsc();
    }
    TS_FORCE_INLINE std::uint64_t rdtscp_lfence() {
        unsigned int aux;
        std::uint64_t t = __rdtscp(&aux);
        _mm_lfence();
        return t;
    }
    TS_FORCE_INLINE void clflush_addr(volatile void* a) {
        _mm_clflush(const_cast<void*>(a));
    }
    TS_FORCE_INLINE void mfence_inst() { _mm_mfence(); }
    TS_FORCE_INLINE void lfence_inst() { _mm_lfence(); }
#else
    static inline std::uint64_t rdtsc_lfence() {
        std::uint64_t lo, hi;
        asm volatile("lfence\n\trdtsc" : "=a"(lo), "=d"(hi));
        return (hi << 32) | lo;
    }
    static inline std::uint64_t rdtscp_lfence() {
        std::uint64_t lo, hi;
        std::uint32_t aux;
        asm volatile("rdtscp" : "=a"(lo), "=d"(hi), "=c"(aux));
        asm volatile("lfence" ::: "memory");
        return (hi << 32) | lo;
    }
    static inline void clflush_addr(volatile void* a) {
        asm volatile("clflush (%0)" :: "r"(a) : "memory");
    }
    static inline void mfence_inst() {
        asm volatile("mfence" ::: "memory");
    }
    static inline void lfence_inst() {
        asm volatile("lfence" ::: "memory");
    }
#endif

#elif defined(__aarch64__) || defined(_M_ARM64)
    static inline std::uint64_t rdtsc_lfence() {
        std::uint64_t v;
        asm volatile("isb\nmrs %0, cntvct_el0" : "=r"(v));
        return v;
    }
    static inline std::uint64_t rdtscp_lfence() { return rdtsc_lfence(); }
    static inline void clflush_addr(volatile void* a) {
        asm volatile("dc civac, %0\ndsb sy" :: "r"(a) : "memory");
    }
    static inline void mfence_inst() {
        asm volatile("dsb sy" ::: "memory");
    }
    static inline void lfence_inst() {
        asm volatile("isb" ::: "memory");
    }

#else
    #include <chrono>
    static inline std::uint64_t rdtsc_lfence() {
        return std::chrono::high_resolution_clock::now()
                   .time_since_epoch().count();
    }
    static inline std::uint64_t rdtscp_lfence() { return rdtsc_lfence(); }
    static inline void clflush_addr(volatile void*) {}
    static inline void mfence_inst() {}
    static inline void lfence_inst() {}
#endif

}}} // namespace tailslayer::platform::timing

#endif // TAILSLAYER_PLATFORM_HPP
