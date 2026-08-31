#pragma once

// QkInt8RotKernel — fused FWHT-rotate + per-token INT8 quant + s32 QK^T +
// rank-1 rescale (docs/fused-dpas-int8-plan.md, Phase 1).
//
// score(i,j) = s32_DPAS(round(Q_rot[i]*127/absmax_i),
//                       round(K_rot[j]*127/absmax_j)) / (q_inv_i * k_inv_j)
//
// Rotation is orthonormal fp32-compute (H.16 DPAS atom + fp32 tail
// butterflies, same network as FwhtXmxTunedKernel); quant is symmetric
// per-token INT8 with the inverse scale folded nowhere but the final fp32
// rescale. d (head dim) is a template parameter — fixed-N rule.

#include "afi/fwht.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>

namespace afi {

// Tunable launch config. tm/sg are kernel-binary (template); wg_k/wg_q/pad/kt
// are runtime launch params.
struct QkInt8Config {
    int tm = 8;   // DPAS m: Q rows per tile, {4, 8}
    int sg = 32;  // sub-group size, {16, 32}
    int wg_k = 4; // subgroups per WG = 16-key tiles staged in parallel, {1,2,4,8}
    int wg_q = 1; // Q row-tiles (tm rows) per WG, {1,2,4}
    int pad = 0;  // SLM leading-dim pad on the fp32/leaf rotate tiles, {0,8}
    int kt = 0;   // 16-key tiles per WG along K (0 = whole row); >1 splits
                  // the K range across WGs for decode-width parallelism at
                  // the price of re-rotating/re-quantizing the Q block.
};

// True when cfg is launchable for head dim d on dev (domains, sub-group
// sizes, DPAS ldm 16B alignment, SLM budget). Does not JIT.
bool qkint8_config_ok(const sycl::device &dev, const QkInt8Config &cfg,
                      int d);

// Fused rotate+quant+QK^T. q is rq×d, k is rk×d (element type per elem,
// fp16 or bf16), out is fp32 rq×rk. The rotation is the unnormalized
// Sylvester network (H@x): the 1/sqrt(d) factor cancels exactly under
// per-token symmetric quant, so the leaf staging stays input-exact.
// Returns false without launching when the config is illegal.
bool qkint8_device(sycl::queue &q, const void *q_src, const void *k_src,
                   float *out, int d, int64_t rq, int64_t rk, FwhtElem elem,
                   const QkInt8Config &cfg, bool wait = true);

// Host oracle: fp32 rotate (same butterflies as python/afi/fwht.py,
// 1/sqrt(d)) + per-token absmax INT8 quant + fp32 QK^T with the rank-1
// rescale. Input rows are fp32 (converted on the host first). This is the
// exact mathematical output of the fused path; the device must match it
// within fp32 rounding.
void qkint8_host(const float *q, const float *k, float *out, int d,
                 int64_t rq, int64_t rk);

// Host fp32 reference without quant (rotate + fp32 QK^T) — the accuracy
// envelope the INT8 path is compared against.
void qkint8_host_fp32(const float *q, const float *k, float *out, int d,
                      int64_t rq, int64_t rk);

} // namespace afi
