// afi-bench-qkint8-tune — Phase 1 autotuner runner for QkInt8RotKernel
// (docs/fused-dpas-int8-plan.md). Measures one or more QkInt8Config launch
// configs at one grain and prints one CSV line per config for
// python/afi/tuner_qkint8.py. Every benched config is correctness-gated
// against the host oracle on the same inputs (the Layer 2 discipline).
//
//   afi-bench-qkint8-tune [-d dev] [--rq 28] [--rk 2048] [--d 128]
//                         [--elem fp16] [--iters 50]
//                         [--config tm=8,sg=32,wg_k=4,wg_q=1,pad=0,kt=1]...
//                         [--scan]
//
// CSV: d,elem,rq,rk,tm,sg,wg_k,wg_q,pad,kt,status,max_abs_err,ms
// status: ok | illegal | wrong | error — ms is only meaningful for ok.

#include "afi/device.hpp"
#include "afi/qkint8.hpp"

#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

afi::QkInt8Config parse_config(const std::string &s) {
    afi::QkInt8Config c;
    size_t pos = 0;
    while (pos <= s.size()) {
        const size_t comma = s.find(',', pos);
        const std::string kv = s.substr(pos, comma - pos);
        const size_t eq = kv.find('=');
        if (eq == std::string::npos)
            throw std::invalid_argument("bad --config term: " + kv);
        const std::string k = kv.substr(0, eq);
        const int v = std::stoi(kv.substr(eq + 1));
        if (k == "tm") c.tm = v;
        else if (k == "sg") c.sg = v;
        else if (k == "wg_k") c.wg_k = v;
        else if (k == "wg_q") c.wg_q = v;
        else if (k == "pad") c.pad = v;
        else if (k == "kt") c.kt = v;
        else throw std::invalid_argument("bad --config key: " + k);
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    return c;
}

// The tuner's admissible lattice; must match python/afi/tuner_qkint8.py.
std::vector<afi::QkInt8Config> scan_configs(const sycl::device &dev, int d) {
    std::vector<afi::QkInt8Config> out;
    for (int tm : {4, 8}) {
        for (int sg : {16, 32}) {
            for (int wg_k : {1, 2, 4, 8}) {
                for (int wg_q : {1, 2, 4}) {
                    for (int pad : {0, 8}) {
                        for (int kt : {0, 1, 2, 4, 8, 16}) {
                            afi::QkInt8Config c;
                            c.tm = tm; c.sg = sg; c.wg_k = wg_k;
                            c.wg_q = wg_q; c.pad = pad; c.kt = kt;
                            if (afi::qkint8_config_ok(dev, c, d))
                                out.push_back(c);
                        }
                    }
                }
            }
        }
    }
    return out;
}

void make_rows(std::vector<float> &x, int64_t rows, int d, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::uniform_int_distribution<int> pick(0, d - 1);
    for (int64_t r = 0; r < rows; ++r) {
        for (int c = 0; c < d; ++c) x[r * d + c] = g(rng);
        const int nout = 1 + static_cast<int>(rng() % 3);
        for (int i = 0; i < nout; ++i)
            x[r * d + pick(rng)] *= 6.0f + 2.0f * g(rng);
    }
}

template <typename T>
void run_one(sycl::queue &q, int d, int64_t rq, int64_t rk, afi::FwhtElem elem,
             const char *label, const afi::QkInt8Config &cfg, int iters) {
    const auto &dev = q.get_device();
    std::cout << d << "," << label << "," << rq << "," << rk << "," << cfg.tm
              << "," << cfg.sg << "," << cfg.wg_k << "," << cfg.wg_q << ","
              << cfg.pad << "," << cfg.kt << ",";
    if (!afi::qkint8_config_ok(dev, cfg, d)) {
        std::cout << "illegal,,,\n";
        return;
    }

    std::vector<float> hq(rq * d), hk(rk * d);
    make_rows(hq, rq, d, 11);
    make_rows(hk, rk, d, 23);
    std::vector<T> tq(rq * d), tk(rk * d);
    for (size_t i = 0; i < hq.size(); ++i) {
        tq[i] = static_cast<T>(hq[i]);
        hq[i] = static_cast<float>(tq[i]);
    }
    for (size_t i = 0; i < hk.size(); ++i) {
        tk[i] = static_cast<T>(hk[i]);
        hk[i] = static_cast<float>(tk[i]);
    }

    T *dq = sycl::malloc_device<T>(rq * d, q);
    T *dk = sycl::malloc_device<T>(rk * d, q);
    float *dout = sycl::malloc_device<float>(rq * rk, q);
    try {
        q.memcpy(dq, tq.data(), tq.size() * sizeof(T)).wait();
        q.memcpy(dk, tk.data(), tk.size() * sizeof(T)).wait();
        if (!afi::qkint8_device(q, dq, dk, dout, d, rq, rk, elem, cfg)) {
            std::cout << "illegal,,,\n";
            sycl::free(dq, q); sycl::free(dk, q); sycl::free(dout, q);
            return;
        }
        // correctness gate on a bounded window (host oracle is O(rq*rk*d))
        const int64_t wr = std::min<int64_t>(rq, 64);
        const int64_t wc = std::min<int64_t>(rk, 256);
        std::vector<float> got(rq * rk);
        q.memcpy(got.data(), dout, got.size() * sizeof(float)).wait();
        double max_err = 0.0;
        {
            std::vector<float> subq(wr * d), subk(wc * d), ref(wr * wc);
            for (int64_t r = 0; r < wr; ++r)
                std::memcpy(&subq[r * d], &hq[r * d], d * sizeof(float));
            for (int64_t r = 0; r < wc; ++r)
                std::memcpy(&subk[r * d], &hk[r * d], d * sizeof(float));
            afi::qkint8_host(subq.data(), subk.data(), ref.data(), d, wr, wc);
            for (int64_t i = 0; i < wr; ++i)
                for (int64_t j = 0; j < wc; ++j)
                    max_err = std::max(
                        max_err, static_cast<double>(std::fabs(
                                     got[i * rk + j] - ref[i * wc + j])));
        }
        if (max_err > 2e-1) {
            std::cout << "wrong," << max_err << ",,\n";
            sycl::free(dq, q); sycl::free(dk, q); sycl::free(dout, q);
            return;
        }

        afi::qkint8_device(q, dq, dk, dout, d, rq, rk, elem, cfg); // warm
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i)
            afi::qkint8_device(q, dq, dk, dout, d, rq, rk, elem, cfg,
                               /*wait=*/false);
        q.wait();
        auto t1 = std::chrono::steady_clock::now();
        const double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
        std::cout << "ok," << max_err << "," << ms << "\n";
    } catch (const std::exception &e) {
        std::cout << "error,,," << " # " << e.what() << "\n";
    }
    sycl::free(dq, q);
    sycl::free(dk, q);
    sycl::free(dout, q);
}

} // namespace

int main(int argc, char **argv) {
    int dev_index = 0;
    int64_t rq = 28, rk = 2048;
    int iters = 50;
    int d = 128;
    bool scan = false;
    std::string elem = "fp16";
    std::vector<afi::QkInt8Config> configs;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if ((a == "-d" || a == "--device") && i + 1 < argc) {
            dev_index = std::stoi(argv[++i]);
        } else if (a == "--rq" && i + 1 < argc) {
            rq = std::stoll(argv[++i]);
        } else if (a == "--rk" && i + 1 < argc) {
            rk = std::stoll(argv[++i]);
        } else if (a == "--d" && i + 1 < argc) {
            d = std::stoi(argv[++i]);
        } else if (a == "--elem" && i + 1 < argc) {
            elem = argv[++i];
        } else if (a == "--iters" && i + 1 < argc) {
            iters = std::stoi(argv[++i]);
        } else if (a == "--scan") {
            scan = true;
        } else if (a == "--config" && i + 1 < argc) {
            configs.push_back(parse_config(argv[++i]));
        } else if (a == "--csv") {
            // output is always CSV; accepted for tuner compatibility
        } else {
            std::cerr << "bench-qkint8-tune: unknown arg " << a << "\n";
            return 2;
        }
    }
    if (!scan && configs.empty()) configs.push_back(afi::QkInt8Config{});
    try {
        auto dev = afi::pick_dgpu(dev_index);
        sycl::queue q(dev);
        // process-level warmup (JIT/context), pattern of bench_xmx_tune
        {
            afi::QkInt8Config warm;
            warm.kt = 1;
            sycl::half *wq = sycl::malloc_device<sycl::half>(8 * 128, q);
            sycl::half *wk = sycl::malloc_device<sycl::half>(16 * 128, q);
            float *wo = sycl::malloc_device<float>(8 * 16, q);
            q.fill(wq, sycl::half(0.01f), 8 * 128).wait();
            q.fill(wk, sycl::half(0.01f), 16 * 128).wait();
            afi::qkint8_device(q, wq, wk, wo, 128, 8, 16, afi::FwhtElem::F16,
                               warm);
            sycl::free(wq, q); sycl::free(wk, q); sycl::free(wo, q);
        }
        std::cout << "# device: " << dev.get_info<sycl::info::device::name>()
                  << "\n";
        std::cout << "d,elem,rq,rk,tm,sg,wg_k,wg_q,pad,kt,status,max_abs_err,"
                     "ms\n";
        std::vector<afi::QkInt8Config> cs =
            scan ? scan_configs(dev, d) : configs;
        for (const auto &c : cs) {
            if (elem == "fp16")
                run_one<sycl::half>(q, d, rq, rk, afi::FwhtElem::F16, "fp16", c,
                                    iters);
            else
                run_one<sycl::ext::oneapi::bfloat16>(q, d, rq, rk,
                                                     afi::FwhtElem::BF16,
                                                     "bf16", c, iters);
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "bench-qkint8-tune: " << e.what() << "\n";
        return 1;
    }
}
