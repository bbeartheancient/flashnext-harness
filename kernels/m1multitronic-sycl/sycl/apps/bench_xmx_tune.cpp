// afi-bench-xmx-tune — Layer 2 autotuner runner. Measures one or more
// FwhtXmxConfig launch configs for the tuned XMX FWHT kernel and prints one
// CSV line per config for python/afi/tuner.py to consume.
//
//   afi-bench-xmx-tune [-d dev] [-r rows] [--n 128 [--n 256]] [--elem fp16]
//                      [--iters 20] [--scan]
//                      [--config tm=8,sg=16,wg_n=0,wg_m=1,pad=0,pf=0]...
//
// CSV: n,elem,rows,tm,sg,wg_n,wg_m,pad,pf,status,max_abs_err,ms,gib_s
// status: ok | illegal | wrong | error — ms is only meaningful for ok.

#include "afi/device.hpp"
#include "afi/fwht.hpp"

#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr double kTol = 1e-2; // same tolerance as afi-fwht-check XMX path

afi::FwhtXmxConfig parse_config(const std::string &s) {
    afi::FwhtXmxConfig c;
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
        else if (k == "wg_n") c.wg_n = v;
        else if (k == "wg_m") c.wg_m = v;
        else if (k == "pad") c.slm_pad = v;
        else if (k == "pf") c.prefetch = v;
        else throw std::invalid_argument("bad --config key: " + k);
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    return c;
}

// The tuner's admissible lattice: every runtime dim cross-product, wiped by
// fwht_xmx_config_ok. Must match python/afi/tuner.py's domain.
std::vector<afi::FwhtXmxConfig> scan_configs(const sycl::device &dev, int n) {
    std::vector<afi::FwhtXmxConfig> out;
    for (int tm : {1, 2, 4, 8}) {
        for (int sg : {16, 32}) {
            for (int wg_n : {1, 2, 4, 8, 16}) {
                if (wg_n * 16 > n) continue;
                for (int wg_m : {1, 2, 4}) {
                    for (int pad : {0, 8}) {
                        for (int pf : {0, 1}) {
                            afi::FwhtXmxConfig c;
                            c.tm = tm; c.sg = sg; c.wg_n = wg_n;
                            c.wg_m = wg_m; c.slm_pad = pad; c.prefetch = pf;
                            if (afi::fwht_xmx_config_ok(dev, c, n))
                                out.push_back(c);
                        }
                    }
                }
            }
        }
    }
    return out;
}

template <typename T>
void run_one(sycl::queue &q, int n, int64_t rows, afi::FwhtElem elem,
             const char *label, const afi::FwhtXmxConfig &cfg, int iters) {
    const auto &dev = q.get_device();
    std::cout << n << "," << label << "," << rows << "," << cfg.tm << ","
              << cfg.sg << "," << cfg.wg_n << "," << cfg.wg_m << ","
              << cfg.slm_pad << "," << cfg.prefetch << ",";
    if (!afi::fwht_xmx_config_ok(dev, cfg, n)) {
        std::cout << "illegal,,,\n";
        return;
    }

    const size_t elems = static_cast<size_t>(rows) * n;
    T *src = sycl::malloc_device<T>(elems, q);
    T *dst = sycl::malloc_device<T>(elems, q);
    std::vector<T> h_src(elems);
    std::vector<float> h_ref(elems), h_in(elems);
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < elems; ++i) {
        h_src[i] = static_cast<T>(dist(rng)); // round inputs into T's domain
        h_in[i] = static_cast<float>(h_src[i]);
    }
    try {
        q.memcpy(src, h_src.data(), elems * sizeof(T)).wait();
        if (!afi::fwht_device_xmx_tuned(q, src, dst, n, rows,
                                        afi::ortho_scale(n), elem, cfg)) {
            std::cout << "illegal,,,\n";
            sycl::free(src, q); sycl::free(dst, q);
            return;
        }
        std::vector<T> h_out(elems);
        q.memcpy(h_out.data(), dst, elems * sizeof(T)).wait();

        h_ref = h_in;
        afi::fwht_host(h_ref.data(), n, rows, afi::ortho_scale(n));
        double max_err = 0.0;
        for (size_t i = 0; i < elems; ++i) {
            max_err = std::max(max_err, static_cast<double>(std::fabs(
                static_cast<float>(h_out[i]) - h_ref[i])));
        }
        if (max_err > kTol) {
            std::cout << "wrong," << max_err << ",,\n";
            sycl::free(src, q); sycl::free(dst, q);
            return;
        }

        // warm the timed path (JIT/module load already paid by the check)
        afi::fwht_device_xmx_tuned(q, src, dst, n, rows, afi::ortho_scale(n),
                                   elem, cfg);
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) {
            afi::fwht_device_xmx_tuned(q, src, dst, n, rows,
                                       afi::ortho_scale(n), elem, cfg);
        }
        auto t1 = std::chrono::steady_clock::now();
        const double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
        const double gib =
            (2.0 * elems * sizeof(T)) / (1024.0 * 1024.0 * 1024.0);
        std::cout << "ok," << max_err << "," << ms << ","
                  << (gib / (ms / 1000.0)) << "\n";
    } catch (const std::exception &e) {
        std::cout << "error,,," << " # " << e.what() << "\n";
    }
    sycl::free(src, q);
    sycl::free(dst, q);
}

} // namespace

int main(int argc, char **argv) {
    int dev_index = 0;
    int64_t rows = 4096;
    int iters = 20;
    bool scan = false;
    std::vector<int> ns;
    std::string elem = "fp16";
    std::vector<afi::FwhtXmxConfig> configs;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if ((a == "-d" || a == "--device") && i + 1 < argc) {
            dev_index = std::stoi(argv[++i]);
        } else if ((a == "-r" || a == "--rows") && i + 1 < argc) {
            rows = std::stoll(argv[++i]);
        } else if (a == "--n" && i + 1 < argc) {
            ns.push_back(std::stoi(argv[++i]));
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
            std::cerr << "bench-xmx-tune: unknown arg " << a << "\n";
            return 2;
        }
    }
    if (ns.empty()) ns = {128, 256};
    if (elem != "fp16" && elem != "bf16") {
        std::cerr << "bench-xmx-tune: --elem must be fp16|bf16\n";
        return 2;
    }
    try {
        auto dev = afi::pick_dgpu(dev_index);
        sycl::queue q(dev);
        // Process-level warmup: the first config measured in a fresh process
        // pays driver/context setup and reads up to 3x slow. Burn one launch
        // + copy before any measurement.
        {
            sycl::half *ws = sycl::malloc_device<sycl::half>(16 * 16, q);
            sycl::half *wd = sycl::malloc_device<sycl::half>(16 * 16, q);
            q.fill(ws, sycl::half(0.1f), 16 * 16).wait();
            afi::FwhtXmxConfig warm;
            warm.tm = 1;
            afi::fwht_device_xmx_tuned(q, ws, wd, 16, 16, afi::ortho_scale(16),
                                       afi::FwhtElem::F16, warm);
            sycl::free(ws, q);
            sycl::free(wd, q);
        }
        std::cout << "# device: " << dev.get_info<sycl::info::device::name>()
                  << "\n";
        std::cout << "n,elem,rows,tm,sg,wg_n,wg_m,pad,pf,status,max_abs_err,"
                     "ms,gib_s\n";
        for (int n : ns) {
            std::vector<afi::FwhtXmxConfig> cs =
                scan ? scan_configs(dev, n) : configs;
            for (const auto &c : cs) {
                if (elem == "fp16")
                    run_one<sycl::half>(q, n, rows, afi::FwhtElem::F16, "fp16",
                                        c, iters);
                else
                    run_one<sycl::ext::oneapi::bfloat16>(
                        q, n, rows, afi::FwhtElem::BF16, "bf16", c, iters);
            }
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "bench-xmx-tune: " << e.what() << "\n";
        return 1;
    }
}
