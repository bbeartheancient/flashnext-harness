#pragma once

#include "afi/fwht.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>

namespace afi {

// GRF-resident row FWHT for half-precision buffers: one hardware thread
// transforms one row; the row lives in simd<float, N> registers and every
// butterfly stage is an in-register region op. No SLM staging, no group
// barriers, no DPAS.
//
// bf16: bit-exact vs the host oracle (fp32 butterflies, single end-scale)
// at every supported width. fp16: identical arithmetic, but the final
// fp32->fp16 store flushes outputs below the half-subnormal cliff (~6e-5)
// to zero (max abs error ~6e-8, numerically harmless) — fp16 results are
// "bit-exact OR subnormal-store-only" per docs/rtu-xess-assessment.md.
//
// Requires 4-byte-aligned src and dst (lsc_block constraint). Returns false
// without launching when n is unsupported or pointers are misaligned.

bool fwht_esimd_ok(const void *src, const void *dst, int n);

bool fwht_device_esimd_bf16(sycl::queue &q, const void *src, void *dst,
                            int n, int64_t n_rows, float scale,
                            bool wait = true);

// elem must be FwhtElem::BF16 or FwhtElem::F16; returns false otherwise.
bool fwht_device_esimd_half(sycl::queue &q, const void *src, void *dst,
                            int n, int64_t n_rows, float scale, FwhtElem elem,
                            bool wait = true);

} // namespace afi
