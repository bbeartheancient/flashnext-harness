// afi-qkint8-check — Phase 1 correctness gate for QkInt8RotKernel
// (docs/fused-dpas-int8-plan.md). Compares the fused kernel against the
// host oracle (rotate fp32 + per-token INT8 quant + fp32 QK^T + rescale)
// on synthetic heavy-tailed rows at the decode (rq=28) and prefill
// (rq=2048) grains, and reports the INT8 score error vs the unquantized
// fp32 reference (the number the fp16-SDPA envelope is compared against).
//
//   afi-qkint8-check [-d dev] [--d 128 [--d 256]] [--elem fp16]
//                    [--rq 28 --rk 2048]... (defaults run both grains)
//   afi-qkint8-check --real q.bin k.bin out.bin --d 256 --rq 438 --rk 438
//                    [--qoff r] [--koff r] [--config tm=8,...]
// Raw-dump mode for the real-Q/K accuracy gate: q/k are fp32 row-major
// dumps (scripts/validate_qkint8_real.py); out.bin receives fp32 scores.
//
// Gate: max |device - host_oracle| must be ~fp32 rounding (tolerance
// scaled by score magnitude). Prints rms/max per grain.

#include "afi/device.hpp"
#include "afi/qkint8.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

bool read_rows(const std::string &path, int64_t off, int64_t rows, int d,
               std::vector<float> &out) {
    FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    if (std::fseek(f, static_cast<long>(off * d * sizeof(float)), SEEK_SET) !=
        0) {
        std::fclose(f);
        return false;
    }
    out.resize(rows * d);
    const size_t got = std::fread(out.data(), sizeof(float), out.size(), f);
    std::fclose(f);
    return got == out.size();
}

// --real mode: kernel on raw fp32 dumps, fp32 scores to disk.
int run_real(sycl::queue &q, const std::string &qfile, const std::string &kfile,
             const std::string &ofile, int d, int64_t rq, int64_t rk,
             int64_t qoff, int64_t koff, const afi::QkInt8Config &cfg) {
    std::vector<float> hq, hk;
    if (!read_rows(qfile, qoff, rq, d, hq) ||
        !read_rows(kfile, koff, rk, d, hk)) {
        std::cerr << "qkint8-check --real: failed to read input rows\n";
        return 2;
    }
    std::vector<sycl::half> tq(rq * d), tk(rk * d);
    for (size_t i = 0; i < hq.size(); ++i) tq[i] = sycl::half(hq[i]);
    for (size_t i = 0; i < hk.size(); ++i) tk[i] = sycl::half(hk[i]);
    sycl::half *dq = sycl::malloc_device<sycl::half>(rq * d, q);
    sycl::half *dk = sycl::malloc_device<sycl::half>(rk * d, q);
    float *dout = sycl::malloc_device<float>(rq * rk, q);
    q.memcpy(dq, tq.data(), tq.size() * sizeof(sycl::half)).wait();
    q.memcpy(dk, tk.data(), tk.size() * sizeof(sycl::half)).wait();
    if (!afi::qkint8_device(q, dq, dk, dout, d, rq, rk, afi::FwhtElem::F16,
                            cfg)) {
        std::cerr << "qkint8-check --real: illegal config\n";
        return 2;
    }
    std::vector<float> got(rq * rk);
    q.memcpy(got.data(), dout, got.size() * sizeof(float)).wait();
    sycl::free(dq, q);
    sycl::free(dk, q);
    sycl::free(dout, q);
    FILE *f = std::fopen(ofile.c_str(), "wb");
    if (!f) return 2;
    std::fwrite(got.data(), sizeof(float), got.size(), f);
    std::fclose(f);
    std::cout << "# wrote " << ofile << " (" << rq << "x" << rk << " fp32)\n";
    return 0;
}


// Heavy-tailed rows in the shape of real Q/K (docs/xmx-tuner.md close-out:
// real K rows have max|k|~14, kurtosis 10-17): mostly N(0,1) with a few
// channels scaled up per row.
void make_rows(std::vector<float> &x, int64_t rows, int d, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::uniform_int_distribution<int> pick(0, d - 1);
    for (int64_t r = 0; r < rows; ++r) {
        for (int c = 0; c < d; ++c) x[r * d + c] = g(rng);
        const int nout = 1 + static_cast<int>(rng() % 3);
        for (int i = 0; i < nout; ++i)
            x[r * d + pick(rng)] *= 6.0f + 8.0f * g(rng) / 4.0f;
    }
}

template <typename T>
int run_grain(sycl::queue &q, int d, int64_t rq, int64_t rk, afi::FwhtElem elem,
              const char *label) {
    std::vector<float> hq(rq * d), hk(rk * d);
    make_rows(hq, rq, d, 11);
    make_rows(hk, rk, d, 23);

    std::vector<T> tq(rq * d), tk(rk * d);
    for (size_t i = 0; i < hq.size(); ++i) tq[i] = static_cast<T>(hq[i]);
    for (size_t i = 0; i < hk.size(); ++i) tk[i] = static_cast<T>(hk[i]);
    // The device sees T-rounded inputs; the oracle must see the same.
    for (size_t i = 0; i < hq.size(); ++i) hq[i] = static_cast<float>(tq[i]);
    for (size_t i = 0; i < hk.size(); ++i) hk[i] = static_cast<float>(tk[i]);

    T *dq = sycl::malloc_device<T>(rq * d, q);
    T *dk = sycl::malloc_device<T>(rk * d, q);
    float *dout = sycl::malloc_device<float>(rq * rk, q);
    q.memcpy(dq, tq.data(), tq.size() * sizeof(T)).wait();
    q.memcpy(dk, tk.data(), tk.size() * sizeof(T)).wait();

    afi::QkInt8Config cfg; // defaults: tm=8 sg=32 wg_q=1 pad=0 kt=0
    cfg.wg_k = d == 256 ? 2 : 4; // SLM budget at d=256
    if (rq < 1024) cfg.kt = 1; // decode grain: split K across WGs
    if (!afi::qkint8_device(q, dq, dk, dout, d, rq, rk, elem, cfg)) {
        std::cout << label << " d=" << d << " rq=" << rq << " rk=" << rk
                  << " : ILLEGAL config\n";
        sycl::free(dq, q); sycl::free(dk, q); sycl::free(dout, q);
        return 1;
    }
    std::vector<float> got(rq * rk);
    q.memcpy(got.data(), dout, got.size() * sizeof(float)).wait();
    sycl::free(dq, q); sycl::free(dk, q); sycl::free(dout, q);

    std::vector<float> ref(rq * rk), ref32(rq * rk);
    afi::qkint8_host(hq.data(), hk.data(), ref.data(), d, rq, rk);
    afi::qkint8_host_fp32(hq.data(), hk.data(), ref32.data(), d, rq, rk);

    double max_d = 0.0, rms_d = 0.0, rms_q = 0.0, max_q = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double e = std::fabs(got[i] - ref[i]);
        max_d = std::max(max_d, e);
        rms_d += e * e;
        const double eq = std::fabs(got[i] - ref32[i]);
        rms_q += eq * eq;
        max_q = std::max(max_q, eq);
    }
    rms_d = std::sqrt(rms_d / got.size());
    rms_q = std::sqrt(rms_q / got.size());
    // Gate: the device must implement the oracle up to noise small vs the
    // quantizer's own error. The residual is fp16/bf16-leaf rounding of the
    // H.16 DPAS rotate inputs (same leaf the half-native FWHT kernels use),
    // which a bit-exact match cannot express (DPAS sum order != sequential
    // butterflies). A layout/staging bug lands at score magnitude instead.
    const bool ok = rms_d < 0.5 * rms_q && max_d < 0.5 * max_q;
    std::cout << label << " d=" << d << " rq=" << rq << " rk=" << rk
              << " : " << (ok ? "ok" : "FAIL") << " vs_oracle max=" << max_d
              << " rms=" << rms_d << " | int8_vs_fp32 rms=" << rms_q
              << " max=" << max_q << "\n";
    return ok ? 0 : 1;
}

} // namespace

int main(int argc, char **argv) {
    int dev_index = 0;
    std::vector<int> ds;
    std::string elem = "fp16";
    std::string real_q, real_k, real_o;
    int64_t rq = 0, rk = 0, qoff = 0, koff = 0;
    int d_real = 256;
    afi::QkInt8Config real_cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if ((a == "-d" || a == "--device") && i + 1 < argc) {
            dev_index = std::stoi(argv[++i]);
        } else if (a == "--d" && i + 1 < argc) {
            ds.push_back(std::stoi(argv[++i]));
        } else if (a == "--elem" && i + 1 < argc) {
            elem = argv[++i];
        } else if (a == "--real" && i + 2 < argc) {
            real_q = argv[++i];
            real_k = argv[++i];
        } else if (a == "--out" && i + 1 < argc) {
            real_o = argv[++i];
        } else if (a == "--rq" && i + 1 < argc) {
            rq = std::stoll(argv[++i]);
        } else if (a == "--rk" && i + 1 < argc) {
            rk = std::stoll(argv[++i]);
        } else if (a == "--qoff" && i + 1 < argc) {
            qoff = std::stoll(argv[++i]);
        } else if (a == "--koff" && i + 1 < argc) {
            koff = std::stoll(argv[++i]);
        } else if (a == "--config" && i + 1 < argc) {
            // tm=..,sg=..,wg_k=..,wg_q=..,pad=..,kt=..
            size_t pos = 0;
            const std::string s = argv[++i];
            while (pos <= s.size()) {
                const size_t comma = s.find(',', pos);
                const std::string kv = s.substr(pos, comma - pos);
                const size_t eq = kv.find('=');
                const std::string k = kv.substr(0, eq);
                const int v = std::stoi(kv.substr(eq + 1));
                if (k == "tm") real_cfg.tm = v;
                else if (k == "sg") real_cfg.sg = v;
                else if (k == "wg_k") real_cfg.wg_k = v;
                else if (k == "wg_q") real_cfg.wg_q = v;
                else if (k == "pad") real_cfg.pad = v;
                else if (k == "kt") real_cfg.kt = v;
                if (comma == std::string::npos) break;
                pos = comma + 1;
            }
        } else {
            std::cerr << "qkint8-check: unknown arg " << a << "\n";
            return 2;
        }
    }
    try {
        auto dev = afi::pick_dgpu(dev_index);
        sycl::queue q(dev);
        std::cout << "# device: " << dev.get_info<sycl::info::device::name>()
                  << "\n";
        if (!real_q.empty()) {
            const int d = ds.empty() ? d_real : ds[0];
            if (real_o.empty() || rq <= 0 || rk <= 0) {
                std::cerr << "qkint8-check --real: need --out/--rq/--rk\n";
                return 2;
            }
            return run_real(q, real_q, real_k, real_o, d, rq, rk, qoff, koff,
                            real_cfg);
        }
        if (ds.empty()) ds = {128, 256};
        int rc = 0;
        for (int d : ds) {
            if (elem == "fp16") {
                rc |= run_grain<sycl::half>(q, d, 28, 2048, afi::FwhtElem::F16,
                                            "decode");
                rc |= run_grain<sycl::half>(q, d, 2048, 2048,
                                            afi::FwhtElem::F16, "prefill");
            } else {
                rc |= run_grain<sycl::ext::oneapi::bfloat16>(
                    q, d, 28, 2048, afi::FwhtElem::BF16, "decode");
                rc |= run_grain<sycl::ext::oneapi::bfloat16>(
                    q, d, 2048, 2048, afi::FwhtElem::BF16, "prefill");
            }
        }
        return rc;
    } catch (const std::exception &e) {
        std::cerr << "qkint8-check: " << e.what() << "\n";
        return 1;
    }
}
