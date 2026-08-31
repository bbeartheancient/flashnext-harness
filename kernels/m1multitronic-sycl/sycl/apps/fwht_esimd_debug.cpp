// afi-fwht-esimd-debug — isolates the GRF FWHT pipeline cut by cut on one
// known input row (N=16): identity round-trip, single butterfly stage,
// full transform vs host expectation.
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include <cstdint>
#include <iostream>
#include <vector>

namespace esimd = sycl::ext::intel::esimd;
namespace xesimd = sycl::ext::intel::experimental::esimd;

namespace {

constexpr int N = 16;

struct CutA {};
struct CutB {};
struct CutC {};
template <int K> struct PrefixName {};

void dump(const char *tag, const std::vector<uint16_t> &v) {
    std::cout << tag << ": ";
    for (auto b : v)
        std::cout << float(sycl::bit_cast<sycl::half>(b)) << " ";
    std::cout << "\n";
}

void host_stage(const std::vector<float> &x, std::vector<float> &y, int h) {
    for (int i = 0; i < N; i += 2 * h)
        for (int j = 0; j < h; ++j) {
            y[i + j] = x[i + j] + x[i + j + h];
            y[i + j + h] = x[i + j] - x[i + j + h];
        }
}

// Apply exactly K leading butterfly stages (h = 1, 2, ..., 2^(K-1)).
template <int H, int REM>
inline std::enable_if_t<(REM <= 0)> apply_stages(esimd::simd<float, N> &) {}

template <int H, int REM>
inline std::enable_if_t<(REM > 0)> apply_stages(esimd::simd<float, N> &f) {
    constexpr int G = N / (2 * H);
    if constexpr (G == 1) {
        auto lo = f.template select<N / 2, 1>(0);
        auto hi = f.template select<N / 2, 1>(N / 2);
        const esimd::simd<float, N / 2> a = lo;
        const esimd::simd<float, N / 2> b = hi;
        lo = a + b;
        hi = a - b;
    } else {
        auto m = f.template bit_cast_view<float, G, 2 * H>();
        auto lo = m.template select<G, 1, H, 1>(0, 0);
        auto hi = m.template select<G, 1, H, 1>(0, H);
        const esimd::simd<float, G * H> a = lo;
        const esimd::simd<float, G * H> b = hi;
        lo = a + b;
        hi = a - b;
    }
    apply_stages<H * 2, REM - 1>(f);
}

template <int K>
void run_cut(sycl::queue &q, uint16_t *din, uint16_t *dout) {
    q.parallel_for<PrefixName<K>>(
        sycl::range<1>(1),
        [=](sycl::id<1>) SYCL_ESIMD_KERNEL {
            auto raw = xesimd::lsc_block_load<uint16_t, N>(din);
            esimd::simd<sycl::half, N> h =
                raw.template bit_cast_view<sycl::half>().read();
            esimd::simd<float, N> f = h;
            apply_stages<1, K>(f);
            esimd::simd<sycl::half, N> ob = f;
            esimd::simd<uint16_t, N> obits =
                ob.template bit_cast_view<uint16_t>().read();
            xesimd::lsc_block_store<uint16_t, N>(dout, obits);
        })
        .wait();
}

// Full butterfly recursion (same shape as sycl/src/fwht_esimd.cpp).
template <int H>
inline std::enable_if_t<(H >= N)> all_stages(esimd::simd<float, N> &) {}

template <int H>
inline std::enable_if_t<(H < N)> all_stages(esimd::simd<float, N> &f) {
    if constexpr (H == N / 2) {
        auto lo = f.template select<H, 1>(0);
        auto hi = f.template select<H, 1>(H);
        const esimd::simd<float, H> a = lo;
        const esimd::simd<float, H> b = hi;
        lo = a + b;
        hi = a - b;
    } else {
        constexpr int G = N / (2 * H);
        auto m = f.template bit_cast_view<float, G, 2 * H>();
        auto lo = m.template select<G, 1, H, 1>(0, 0);
        auto hi = m.template select<G, 1, H, 1>(0, H);
        const esimd::simd<float, G * H> a = lo;
        const esimd::simd<float, G * H> b = hi;
        lo = a + b;
        hi = a - b;
    }
    all_stages<H * 2>(f);
}

} // namespace

int main() {
    try {
        sycl::queue q{sycl::device(sycl::gpu_selector_v)};

        std::vector<uint16_t> hin(N);
        std::vector<float> fin(N);
        for (int i = 0; i < N; ++i) {
            fin[i] = float(i + 1);
            hin[i] = sycl::bit_cast<uint16_t>(sycl::half(fin[i]));
        }

        auto *din = sycl::malloc_device<uint16_t>(N, q);
        auto *dout = sycl::malloc_device<uint16_t>(N, q);
        q.memcpy(din, hin.data(), N * 2).wait();

        // Cut A — identity: u16 load -> half -> float -> half -> u16 bits.
        q.parallel_for<CutA>(sycl::range<1>(1),
                             [=](sycl::id<1>) SYCL_ESIMD_KERNEL {
                                 auto raw =
                                     xesimd::lsc_block_load<uint16_t, N>(din);
                                 esimd::simd<sycl::half, N> h =
                                     raw.template bit_cast_view<sycl::half>()
                                         .read();
                                 esimd::simd<float, N> f = h;
                                 esimd::simd<sycl::half, N> ob = f;
                                 esimd::simd<uint16_t, N> obits =
                                     ob.template bit_cast_view<uint16_t>()
                                         .read();
                                 xesimd::lsc_block_store<uint16_t, N>(dout,
                                                                      obits);
                             })
            .wait();
        std::vector<uint16_t> hout(N);
        q.memcpy(hout.data(), dout, N * 2).wait();
        dump("A identity ", hout);

        // Cut B — single stage h=2.
        q.parallel_for<CutB>(sycl::range<1>(1),
                             [=](sycl::id<1>) SYCL_ESIMD_KERNEL {
                                 auto raw =
                                     xesimd::lsc_block_load<uint16_t, N>(din);
                                 esimd::simd<sycl::half, N> h =
                                     raw.template bit_cast_view<sycl::half>()
                                         .read();
                                 esimd::simd<float, N> f = h;
                                 constexpr int H = 2, G = N / (2 * H);
                                 auto m = f.template bit_cast_view<float, G, 2 * H>();
                                 auto lo = m.template select<G, 1, H, 1>(0, 0);
                                 auto hi = m.template select<G, 1, H, 1>(0, H);
                                 const esimd::simd<float, G * H> a = lo;
                                 const esimd::simd<float, G * H> b = hi;
                                 lo = a + b;
                                 hi = a - b;
                                 esimd::simd<sycl::half, N> ob = f;
                                 esimd::simd<uint16_t, N> obits =
                                     ob.template bit_cast_view<uint16_t>()
                                         .read();
                                 xesimd::lsc_block_store<uint16_t, N>(dout,
                                                                      obits);
                             })
            .wait();
        q.memcpy(hout.data(), dout, N * 2).wait();
        dump("B stage-h2 ", hout);
        {
            std::vector<float> e(N);
            host_stage(fin, e, 2);
            std::cout << "B expected : ";
            for (float x : e) std::cout << x << " ";
            std::cout << "\n";
        }

        // Prefix cuts: K = 1..3 stages.
        auto show = [&](int k, const std::vector<uint16_t> &hk) {
            std::vector<float> e = fin, t(N);
            for (int s2 = 0; s2 < k; ++s2) {
                host_stage(e, t, 1 << s2);
                e = t;
            }
            std::cout << "K" << k << " device : ";
            for (uint16_t b : hk)
                std::cout << float(sycl::bit_cast<sycl::half>(b)) << " ";
            std::cout << "\nK" << k << " expect : ";
            for (float x : e) std::cout << x << " ";
            std::cout << "\n";
        };
        {
            std::vector<uint16_t> hk(N);
            run_cut<1>(q, din, dout);
            q.memcpy(hk.data(), dout, N * 2).wait();
            show(1, hk);
        }
        {
            std::vector<uint16_t> hk(N);
            run_cut<2>(q, din, dout);
            q.memcpy(hk.data(), dout, N * 2).wait();
            show(2, hk);
        }
        {
            std::vector<uint16_t> hk(N);
            run_cut<3>(q, din, dout);
            q.memcpy(hk.data(), dout, N * 2).wait();
            show(3, hk);
        }

        // Cut C — full transform.
        q.parallel_for<CutC>(sycl::range<1>(1),
                             [=](sycl::id<1>) SYCL_ESIMD_KERNEL {
                                 auto raw =
                                     xesimd::lsc_block_load<uint16_t, N>(din);
                                 esimd::simd<sycl::half, N> h =
                                     raw.template bit_cast_view<sycl::half>()
                                         .read();
                                 esimd::simd<float, N> f = h;
                                 all_stages<1>(f);
                                 esimd::simd<sycl::half, N> ob = f;
                                 esimd::simd<uint16_t, N> obits =
                                     ob.template bit_cast_view<uint16_t>()
                                         .read();
                                 xesimd::lsc_block_store<uint16_t, N>(dout,
                                                                      obits);
                             })
            .wait();
        q.memcpy(hout.data(), dout, N * 2).wait();
        dump("C full     ", hout);
        {
            std::vector<float> e = fin, t(N);
            for (int hh = 1; hh < N; hh <<= 1) {
                host_stage(e, t, hh);
                e = t;
            }
            std::cout << "C expected : ";
            for (float x : e) std::cout << x << " ";
            std::cout << "\n";
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "debug: " << e.what() << "\n";
        return 1;
    }
}
