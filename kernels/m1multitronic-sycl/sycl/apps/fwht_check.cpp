#include "afi/device.hpp"
#include "afi/fwht.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

int check_n(sycl::queue &q, int n, int64_t rows, uint32_t seed, afi::FwhtImpl impl) {
    const float scale = 1.0f; // unnormalized: match H @ x
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> host(static_cast<size_t>(rows) * static_cast<size_t>(n));
    for (auto &v : host) v = dist(rng);
    auto expect = host;
    afi::fwht_host(expect.data(), n, rows, scale);

    float *src = sycl::malloc_device<float>(host.size(), q);
    float *dst = sycl::malloc_device<float>(host.size(), q);
    q.memcpy(src, host.data(), host.size() * sizeof(float)).wait();
    afi::fwht_device(q, src, dst, n, rows, scale, impl);
    std::vector<float> got(host.size());
    q.memcpy(got.data(), dst, host.size() * sizeof(float)).wait();
    sycl::free(src, q);
    sycl::free(dst, q);

    double max_abs = 0.0;
    double max_rel = 0.0;
    for (size_t i = 0; i < host.size(); ++i) {
        const double e = std::abs(static_cast<double>(got[i]) - expect[i]);
        max_abs = std::max(max_abs, e);
        const double denom = std::max(1e-6, std::abs(static_cast<double>(expect[i])));
        max_rel = std::max(max_rel, e / denom);
    }
    // XMX FP16 atom: rounding on the H.16 leaf, tail stays fp32.
    const double lim = (impl == afi::FwhtImpl::Xmx || impl == afi::FwhtImpl::Auto) ? 1e-2 : 1e-3;
    const bool ok = max_abs < lim;
    std::cout << "impl=" << afi::fwht_impl_name(impl) << " n=" << n
              << " rows=" << rows << " max_abs=" << max_abs
              << " max_rel=" << max_rel << (ok ? "  ok\n" : "  FAIL\n");
    return ok ? 0 : 1;
}

// Half-native check: input is rounded to T before the host oracle so the
// comparison is in the fp16/bf16 domain the kernel lives in.
template <typename T>
int check_n_typed(sycl::queue &q, int n, int64_t rows, uint32_t seed,
                  afi::FwhtImpl impl, const char *label, afi::FwhtElem elem) {
    const float scale = 1.0f;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const size_t elems = static_cast<size_t>(rows) * static_cast<size_t>(n);
    std::vector<float> host(elems);
    for (auto &v : host) v = dist(rng);

    std::vector<T> in(elems);
    std::vector<float> expect_f(elems);
    for (size_t i = 0; i < elems; ++i) {
        in[i] = static_cast<T>(host[i]);
        expect_f[i] = static_cast<float>(in[i]);
    }
    afi::fwht_host(expect_f.data(), n, rows, scale);
    std::vector<T> expect(elems);
    for (size_t i = 0; i < elems; ++i) expect[i] = static_cast<T>(expect_f[i]);

    T *src = sycl::malloc_device<T>(elems, q);
    T *dst = sycl::malloc_device<T>(elems, q);
    q.memcpy(src, in.data(), elems * sizeof(T)).wait();
    afi::fwht_device_typed(q, src, dst, n, rows, scale, elem, impl);
    std::vector<T> got_v(elems);
    q.memcpy(got_v.data(), dst, elems * sizeof(T)).wait();
    sycl::free(src, q);
    sycl::free(dst, q);

    double max_abs = 0.0;
    for (size_t i = 0; i < elems; ++i) {
        const double got = static_cast<double>(static_cast<float>(got_v[i]));
        const double exp = static_cast<double>(static_cast<float>(expect[i]));
        max_abs = std::max(max_abs, std::abs(got - exp));
    }
    const double lim =
        (impl == afi::FwhtImpl::Xmx || impl == afi::FwhtImpl::Auto) ? 1e-2 : 1e-3;
    const bool ok = max_abs < lim;
    std::cout << "impl=" << afi::fwht_impl_name(impl) << " " << label
              << " n=" << n << " rows=" << rows << " max_abs=" << max_abs
              << (ok ? "  ok\n" : "  FAIL\n");
    return ok ? 0 : 1;
}

} // namespace

int main(int argc, char **argv) {
    int dev_index = 0;
    afi::FwhtImpl impl = afi::FwhtImpl::Auto;
    bool both = true;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if ((a == "-d" || a == "--device") && i + 1 < argc) {
            dev_index = std::stoi(argv[++i]);
        } else if ((a == "--impl") && i + 1 < argc) {
            const std::string v = argv[++i];
            both = false;
            if (v == "slm") impl = afi::FwhtImpl::Slm;
            else if (v == "xmx") impl = afi::FwhtImpl::Xmx;
            else if (v == "auto") impl = afi::FwhtImpl::Auto;
            else {
                std::cerr << "fwht-check: --impl slm|xmx|auto\n";
                return 2;
            }
        }
    }
    try {
        auto dev = afi::pick_dgpu(dev_index);
        std::cout << "device: " << dev.get_info<sycl::info::device::name>() << "\n";
        std::cout << "xmx_ok: " << (afi::fwht_xmx_ok(dev) ? "yes" : "no") << "\n";
        sycl::queue q(dev);
        int rc = 0;
        auto run = [&](afi::FwhtImpl i) {
            for (int n : {16, 32, 64, 128, 256}) {
                rc |= check_n(q, n, 32, 0xA770u + static_cast<uint32_t>(n), i);
                rc |= check_n_typed<sycl::half>(q, n, 32,
                                                0xB700u + static_cast<uint32_t>(n),
                                                i, "fp16", afi::FwhtElem::F16);
                rc |= check_n_typed<sycl::ext::oneapi::bfloat16>(
                    q, n, 32, 0xB701u + static_cast<uint32_t>(n), i, "bf16",
                    afi::FwhtElem::BF16);
            }
        };
        if (both) {
            run(afi::FwhtImpl::Slm);
            if (afi::fwht_xmx_ok(dev)) run(afi::FwhtImpl::Xmx);
        } else {
            run(impl);
        }
        return rc;
    } catch (const std::exception &e) {
        std::cerr << "fwht-check: " << e.what() << "\n";
        return 1;
    }
}
