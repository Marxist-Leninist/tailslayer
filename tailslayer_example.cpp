#include <tailslayer/hedged_reader.hpp>
#include <iostream>

// Example with arguments
TS_FORCE_INLINE std::size_t dummy_read_signal2(int arg1, int arg2) {
    std::cout << "Hi with args: " << arg1 << " " << arg2 << "\n";
    return 0;
}

template <typename T>
TS_FORCE_INLINE void dummy_final_work2(T val, int arg2) {
    std::cout << "Hi with args: " << +val << " " << arg2 << "\n";
}

// Example with no arguments
TS_FORCE_INLINE std::size_t dummy_read_signal() {
#if defined(__x86_64__) || defined(__i386__) || \
    defined(_M_X64)     || defined(_M_IX86)
    uint64_t start = tailslayer::platform::timing::rdtsc_lfence();
    uint64_t elapsed{0};
    do {
        elapsed = tailslayer::platform::timing::rdtsc_lfence() - start;
    } while (elapsed < 2000000000);
#endif

    std::size_t index_to_read = 1;
    return index_to_read;
}

template <typename T>
TS_FORCE_INLINE void dummy_final_work(T val) {
#ifdef _MSC_VER
    volatile auto sink = val;
    (void)sink;
#else
    asm volatile("" :: "r"(val));
#endif
    std::cout << "Val: " << +val << "\n";
}

int main() {
    using target_size_t = uint8_t;
    tailslayer::pin_to_core(tailslayer::CORE_MAIN);

    std::cout << "Start tailslayer demo.\n";

    tailslayer::HedgedReader<target_size_t, dummy_read_signal,
                             dummy_final_work<target_size_t>> reader{};
    reader.insert(0x43);
    reader.insert(0x44);
    reader.start_workers();

    std::cout << "End tailslayer demo.\n";
    return 0;
}
