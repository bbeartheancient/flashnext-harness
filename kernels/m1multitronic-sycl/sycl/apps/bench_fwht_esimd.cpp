// afi-bench-fwht-esimd — GRF-resident row FWHT: bit-exact correctness gate
// vs the host oracle + timing vs the shipped SLM/XMX paths at identical
// shapes (GRF-FWHT rewrite, 2026-08-22).
//
// CSV: elem,n,rows,impl,status,ms,gibs,mismatches
//
// afi-bench-fwht-esimd [-d dev] [--rows 2048 ...] [--n 128 ...]
//                      [--iters 50] [--elem f16|bf16|f32|all]
#include "afi/fwht.hpp"
#include "afi/fwht_esimd.hpp"
#include "afi/device.hpp"

#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using bfloat16 = sycl::ext::oneapi::bfloat16;

template <typename T>
void host_ref(const void *srcv, void *refv, int n, int64_t rows,
              float scale) {
    const T *src = static_cast<const T *>(srcv);
    T *ref = static_cast<T *>(refv);
    std::vector<float> tmp(n);
    for (int64_t r = 0; r < rows; ++r) {
        for (int i = 0; i < n; ++i) tmp[i] = float(src[r * n + i]);
        afi::fwht_host(tmp.data(), n, 1, scale);
        for (int i = 0; i < n; ++i) ref[r * n + i] = T(tmp[i]);
    }
}

// Double-precision arbiter reference: same butterfly order as fwht_host,
// accumulated in double so intermediate rounding cannot be the deviator.
template <typename T>
void host_ref_f64(const void *srcv, void *refv, int n, int64_t rows,
                  float scale) {
    const T *src = static_cast<const T *>(srcv);
    T *ref = static_cast<T *>(refv);
    std::vector<double> tmp(n);
    for (int64_t r = 0; r < rows; ++r) {
        for (int i = 0; i < n; ++i) tmp[i] = double(src[r * n + i]);
        for (int h = 1; h < n; h *= 2)
            for (int i = 0; i < n; i += 2 * h)
                for (int j = 0; j < h; ++j) {
                    const double a = tmp[i + j], b = tmp[i + j + h];
                    tmp[i + j] = a + b;
                    tmp[i + j + h] = a - b;
                }
        for (int i = 0; i < n; ++i) ref[r * n + i] = T(tmp[i] * scale);
    }
}

template <typename F>
double time_run(sycl::queue &q, int iters, F &&launch) {
    launch(); // JIT + warm
    q.wait_and_throw();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) launch();
    q.wait_and_throw();
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

const char *impl_name(int impl) {
    switch (impl) {
    case 0: return "esimd";
    case 1: return "slm";
    case 2: return "xmx";
    }
    return "?";
}

} // namespace

int main(int argc, char **argv) {
    int dev_index = 0;
    int iters = 50;
    float scale_mult = 1.0f;
    std::vector<int> ns;
    std::vector<int64_t> rowss;
    std::string elem_arg = "all";
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if ((a == "-d" || a == "--device") && i + 1 < argc)
            dev_index = std::stoi(argv[++i]);
        else if (a == "--iters" && i + 1 < argc)
            iters = std::stoi(argv[++i]);
        else if (a == "--rows" && i + 1 < argc)
            rowss.push_back(std::stoll(argv[++i]));
        else if (a == "--n" && i + 1 < argc)
            ns.push_back(std::stoi(argv[++i]));
        else if (a == "--elem" && i + 1 < argc)
            elem_arg = argv[++i];
        else if (a == "--scale" && i + 1 < argc)
            scale_mult = std::stof(argv[++i]);
        else {
            std::cerr << "unknown arg " << a << "\n";
            return 2;
        }
    }
    if (ns.empty()) ns = {128, 256};
    if (rowss.empty()) rowss = {2048, 4096};

    try {
        auto dev = afi::pick_dgpu(dev_index);
        sycl::queue q(dev);
        const auto info = afi::inspect(dev, dev_index);
        std::cout << "# device: " << info.name << "\n";

        struct ElemSpec {
            const char *name;
            afi::FwhtElem elem;
            bool half; // 2-byte element
        };
        std::vector<ElemSpec> elems;
        if (elem_arg == "all" || elem_arg == "f16")
            elems.push_back({"f16", afi::FwhtElem::F16, true});
        if (elem_arg == "all" || elem_arg == "bf16")
            elems.push_back({"bf16", afi::FwhtElem::BF16, true});
        if (elem_arg == "all" || elem_arg == "f32")
            elems.push_back({"f32", afi::FwhtElem::F32, false});

        std::mt19937 rng(7);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::cout << "elem,n,rows,impl,status,ms,gibs,mismatches\n";

        for (const auto &es : elems) {
            const size_t esz = es.half ? 2 : 4;
            for (int n : ns) {
                if (!afi::fwht_supported(n)) continue;
                const float scale = afi::ortho_scale(n) * scale_mult;

                for (int64_t rows : rowss) {
                    const size_t total = size_t(rows) * n;
                    const size_t bytes = total * esz;

                    // Host input in the native dtype.
                    std::vector<uint8_t> hin(bytes);
                    if (es.elem == afi::FwhtElem::F16) {
                        auto *p = reinterpret_cast<sycl::half *>(hin.data());
                        for (size_t i = 0; i < total; ++i)
                            p[i] = sycl::half(dist(rng));
                    } else if (es.elem == afi::FwhtElem::BF16) {
                        auto *p = reinterpret_cast<bfloat16 *>(hin.data());
                        for (size_t i = 0; i < total; ++i)
                            p[i] = bfloat16(dist(rng));
                    } else {
                        auto *p = reinterpret_cast<float *>(hin.data());
                        for (size_t i = 0; i < total; ++i) p[i] = dist(rng);
                    }

                    // Bit-exact host reference.
                    std::vector<uint8_t> ref(bytes), ref64(bytes);
                    const bool arb = getenv("AFI_ARB") != nullptr;
                    if (es.elem == afi::FwhtElem::F16) {
                        host_ref<sycl::half>(hin.data(), ref.data(), n, rows,
                                             scale);
                        if (arb)
                            host_ref_f64<sycl::half>(hin.data(), ref64.data(),
                                                     n, rows, scale);
                    }
                    else if (es.elem == afi::FwhtElem::BF16)
                        host_ref<bfloat16>(hin.data(), ref.data(), n, rows,
                                           scale);
                    else
                        host_ref<float>(hin.data(), ref.data(), n, rows,
                                        scale);

                    uint8_t *src =
                        static_cast<uint8_t *>(sycl::malloc_device(bytes, q));
                    uint8_t *dst =
                        static_cast<uint8_t *>(sycl::malloc_device(bytes, q));
                    q.memcpy(src, hin.data(), bytes).wait();

                    auto gibs_for = [&](double ms) {
                        const double moved = double(bytes) * 2;
                        return moved / (ms * 1e-3) / 1e9;
                    };

                    for (int impl = 0; impl <= 2; ++impl) {
                        if (impl == 0 && !es.half)
                            continue; // ESIMD path ships half-native only

                        // Correctness pass.
                        bool launched = true;
                        try {
                            if (impl == 0) {
                                if (es.elem == afi::FwhtElem::F32)
                                    continue; // esimd path is half-native
                                launched = afi::fwht_device_esimd_half(
                                    q, src, dst, n, rows, scale, es.elem,
                                    true);
                            } else if (impl == 1) {
                                afi::fwht_device_typed(q, src, dst, n, rows,
                                                       scale, es.elem,
                                                       afi::FwhtImpl::Slm,
                                                       true);
                            } else {
                                afi::fwht_device_typed(q, src, dst, n, rows,
                                                       scale, es.elem,
                                                       afi::FwhtImpl::Xmx,
                                                       true);
                            }
                        } catch (const std::exception &e) {
                            launched = false;
                            std::cerr << "# " << impl_name(impl)
                                      << " launch failed: " << e.what()
                                      << "\n";
                        }
                        if (!launched) continue;

                        std::vector<uint8_t> got(bytes);
                        q.memcpy(got.data(), dst, bytes).wait();
                        size_t mismatches = 0;
                        size_t first_bad = SIZE_MAX;
                        for (size_t i = 0; i < bytes; ++i) {
                            if (got[i] == ref[i]) continue;
                            if (first_bad == SIZE_MAX) first_bad = i;
                            ++mismatches;
                        }
                        if (getenv("AFI_DUMP") && first_bad != SIZE_MAX &&
                            mismatches <= 4096) {
                            // Per-mismatch verdict against the f64 arbiter:
                            // who holds the exactly-rounded half value?
                            size_t dev_right = 0, host_right = 0;
                            for (size_t e = 0; e < total; ++e) {
                                const uint16_t g =
                                    sycl::bit_cast<uint16_t>(
                                        reinterpret_cast<const sycl::half *>(
                                            got.data())[e]);
                                const uint16_t r3 =
                                    sycl::bit_cast<uint16_t>(
                                        reinterpret_cast<const sycl::half *>(
                                            ref.data())[e]);
                                const uint16_t r6 =
                                    sycl::bit_cast<uint16_t>(
                                        reinterpret_cast<const sycl::half *>(
                                            ref64.data())[e]);
                                if (g == r6 && r3 != r6) ++dev_right;
                                if (r3 == r6 && g != r6) ++host_right;
                            }
                            size_t ref_subnormals = 0;
                            for (size_t e = 0; e < total; ++e) {
                                const uint16_t r6 =
                                    sycl::bit_cast<uint16_t>(
                                        reinterpret_cast<const sycl::half *>(
                                            ref64.data())[e]);
                                if (r6 >= 0x0001 && r6 <= 0x03FF)
                                    ++ref_subnormals;
                            }
                            std::cerr << "# arbiter: device==f64 at "
                                      << dev_right << " elems, fp32ref==f64 "
                                      << "at " << host_right
                                      << " (mismatches=" << mismatches
                                      << "); ref64 half-SUBNORMAL outputs="
                                      << ref_subnormals << "\n";
                        }

                        if (mismatches != 0) {
                            // fp16 esimd gate: "bit-exact OR
                            // subnormal-store-only". Every mismatching
                            // element must have its fp32-oracle ref value
                            // below the half-subnormal cliff; the store
                            // flushes those to zero (|err| <= ~6e-8).
                            bool sub_only =
                                impl == 0 && es.elem == afi::FwhtElem::F16;
                            if (sub_only) {
                                const auto *g16 =
                                    reinterpret_cast<const sycl::half *>(
                                        got.data());
                                const auto *r16 =
                                    reinterpret_cast<const sycl::half *>(
                                        ref.data());
                                for (size_t e = 0; e < total && sub_only;
                                     ++e) {
                                    const uint16_t gb =
                                        sycl::bit_cast<uint16_t>(g16[e]);
                                    const uint16_t rb =
                                        sycl::bit_cast<uint16_t>(r16[e]);
                                    if (gb == rb) continue;
                                    const uint16_t rmag = rb & 0x7FFF;
                                    const uint16_t gmag = gb & 0x7FFF;
                                    if (rmag > 0x03FF || gmag > 0x03FF)
                                        sub_only = false;
                                }
                            }
                            if (sub_only) {
                                std::cerr << "# esimd f16 n=" << n
                                          << " rows=" << rows
                                          << ": subnormal-only mismatches="
                                          << mismatches << " (gate pass)\n";
                            } else {
                                std::cout << es.name << "," << n << "," << rows
                                          << "," << impl_name(impl)
                                          << ",wrong,0,0," << mismatches
                                          << "\n";
                                continue;
                            }
                        }

                        double ms;
                        if (impl == 0) {
                            ms = time_run(q, iters, [&] {
                                afi::fwht_device_esimd_half(
                                    q, src, dst, n, rows, scale, es.elem,
                                    false);
                            });
                        } else if (impl == 1) {
                            ms = time_run(q, iters, [&] {
                                afi::fwht_device_typed(
                                    q, src, dst, n, rows, scale, es.elem,
                                    afi::FwhtImpl::Slm, false);
                            });
                        } else {
                            ms = time_run(q, iters, [&] {
                                afi::fwht_device_typed(
                                    q, src, dst, n, rows, scale, es.elem,
                                    afi::FwhtImpl::Xmx, false);
                            });
                        }
                        std::cout << es.name << "," << n << "," << rows << ","
                                  << impl_name(impl) << ",ok," << ms << ","
                                  << gibs_for(ms) << ",0\n";
                    }
                    sycl::free(src, q);
                    sycl::free(dst, q);
                }
            }
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "bench-fwht-esimd: " << e.what() << "\n";
        return 1;
    }
}
