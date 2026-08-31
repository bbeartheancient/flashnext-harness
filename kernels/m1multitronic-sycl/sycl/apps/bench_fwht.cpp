#include "afi/device.hpp"
#include "afi/fwht.hpp"

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

static void bench_one(sycl::queue &q, int n, int64_t rows, afi::FwhtImpl impl) {
    const size_t elems = static_cast<size_t>(rows) * static_cast<size_t>(n);
    float *src = sycl::malloc_device<float>(elems, q);
    float *dst = sycl::malloc_device<float>(elems, q);
    q.fill(src, 0.1f, elems).wait();
    afi::fwht_device(q, src, dst, n, rows, afi::ortho_scale(n), impl);
    const int iters = 20;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        afi::fwht_device(q, src, dst, n, rows, afi::ortho_scale(n), impl);
    }
    auto t1 = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    const double gib = (2.0 * elems * sizeof(float)) / (1024.0 * 1024.0 * 1024.0);
    std::cout << "impl=" << afi::fwht_impl_name(impl) << " n=" << n
              << " rows=" << rows << "  " << ms << " ms  "
              << (gib / (ms / 1000.0)) << " GiB/s (in+out)\n";
    sycl::free(src, q);
    sycl::free(dst, q);
}

// Half-native in/out — bandwidth credited at the 2-byte element size.
template <typename T>
static void bench_one_typed(sycl::queue &q, int n, int64_t rows, afi::FwhtImpl impl,
                            afi::FwhtElem elem, const char *label) {
    const size_t elems = static_cast<size_t>(rows) * static_cast<size_t>(n);
    T *src = sycl::malloc_device<T>(elems, q);
    T *dst = sycl::malloc_device<T>(elems, q);
    q.fill(src, static_cast<T>(0.1f), elems).wait();
    afi::fwht_device_typed(q, src, dst, n, rows, afi::ortho_scale(n), elem, impl);
    const int iters = 20;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        afi::fwht_device_typed(q, src, dst, n, rows, afi::ortho_scale(n), elem,
                               impl);
    }
    auto t1 = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    const double gib =
        (2.0 * elems * sizeof(T)) / (1024.0 * 1024.0 * 1024.0);
    std::cout << "impl=" << afi::fwht_impl_name(impl) << " " << label << " n="
              << n << " rows=" << rows << "  " << ms << " ms  "
              << (gib / (ms / 1000.0)) << " GiB/s (in+out)\n";
    sycl::free(src, q);
    sycl::free(dst, q);
}

int main(int argc, char **argv) {
    int dev_index = 0;
    int64_t rows = 4096;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if ((a == "-d" || a == "--device") && i + 1 < argc) {
            dev_index = std::stoi(argv[++i]);
        } else if ((a == "-r" || a == "--rows") && i + 1 < argc) {
            rows = std::stoll(argv[++i]);
        }
    }
    try {
        auto dev = afi::pick_dgpu(dev_index);
        std::cout << "device: " << dev.get_info<sycl::info::device::name>() << "\n";
        std::cout << "xmx_ok: " << (afi::fwht_xmx_ok(dev) ? "yes" : "no") << "\n";
        sycl::queue q(dev);
        for (int n : {16, 32, 64, 128, 256}) {
            bench_one(q, n, rows, afi::FwhtImpl::Slm);
            bench_one_typed<sycl::half>(q, n, rows, afi::FwhtImpl::Slm,
                                        afi::FwhtElem::F16, "fp16");
            bench_one_typed<sycl::ext::oneapi::bfloat16>(q, n, rows, afi::FwhtImpl::Slm,
                                            afi::FwhtElem::BF16, "bf16");
            if (afi::fwht_xmx_ok(dev)) {
                bench_one(q, n, rows, afi::FwhtImpl::Xmx);
                bench_one_typed<sycl::half>(q, n, rows, afi::FwhtImpl::Xmx,
                                            afi::FwhtElem::F16, "fp16");
                bench_one_typed<sycl::ext::oneapi::bfloat16>(q, n, rows, afi::FwhtImpl::Xmx,
                                                afi::FwhtElem::BF16, "bf16");
            }
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "bench-fwht: " << e.what() << "\n";
        return 1;
    }
}
