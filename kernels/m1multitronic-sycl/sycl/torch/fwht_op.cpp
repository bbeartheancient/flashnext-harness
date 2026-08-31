// torch.ops.afi.fwht — SYCL FWHT on the current XPU stream.
// Compile with intel/llvm v7 (joint_matrix). Do not splice into GPTQ GEMM.

#include <sycl/sycl.hpp>

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/library.h>

#include "afi/fwht.hpp"

#include <optional>
#include <string>

namespace {

afi::FwhtImpl parse_impl(std::string_view s) {
    if (s.empty() || s == "auto") return afi::FwhtImpl::Auto;
    if (s == "slm") return afi::FwhtImpl::Slm;
    if (s == "xmx") return afi::FwhtImpl::Xmx;
    if (s == "esimd") return afi::FwhtImpl::Esimd;
    TORCH_CHECK(false, "afi.fwht impl must be auto|slm|xmx|esimd, got ", s);
}

float resolve_scale(const std::optional<double> &scale, int n) {
    if (scale.has_value()) return static_cast<float>(*scale);
    return afi::ortho_scale(n);
}

void check_last_dim(const at::Tensor &x) {
    TORCH_CHECK(x.dim() >= 1, "afi.fwht: expected at least 1 dim");
    const auto n = x.size(-1);
    TORCH_CHECK(afi::fwht_supported(static_cast<int>(n)),
                "afi.fwht: last dim must be 16/32/64/128/256, got ", n);
}

at::Tensor fwht_cpu(const at::Tensor &x, std::optional<double> scale,
                    std::string_view impl) {
    (void)impl;
    check_last_dim(x);
    auto src = x.contiguous().to(at::kFloat);
    const int n = static_cast<int>(src.size(-1));
    const int64_t rows = src.numel() / n;
    auto out = src.clone();
    afi::fwht_host(out.data_ptr<float>(), n, rows, resolve_scale(scale, n));
    return out.to(x.scalar_type()).view(x.sizes());
}

at::Tensor fwht_xpu(const at::Tensor &x, std::optional<double> scale,
                    std::string_view impl) {
    TORCH_CHECK(x.is_xpu(), "afi.fwht: expected XPU tensor");
    check_last_dim(x);
    const auto impl_e = parse_impl(impl);
    auto src = x.contiguous();
    const int n = static_cast<int>(src.size(-1));
    const int64_t rows = src.numel() / n;
    const float sc = resolve_scale(scale, n);
    auto &q = c10::xpu::getCurrentXPUStream(src.device().index()).queue();

    auto work_out = at::empty_like(src);
    if (src.scalar_type() == at::kFloat) {
        afi::fwht_device_typed(q, src.data_ptr<float>(),
                               work_out.data_ptr<float>(), n, rows, sc,
                               afi::FwhtElem::F32, impl_e, /*wait=*/false);
    } else if (src.scalar_type() == at::kHalf) {
        // Half-native in/out — no fp32 cast-in / cast-out pass.
        afi::fwht_device_typed(q, src.data_ptr<at::Half>(),
                               work_out.data_ptr<at::Half>(), n, rows, sc,
                               afi::FwhtElem::F16, impl_e, /*wait=*/false);
    } else if (src.scalar_type() == at::kBFloat16) {
        afi::fwht_device_typed(q, src.data_ptr<at::BFloat16>(),
                               work_out.data_ptr<at::BFloat16>(), n, rows, sc,
                               afi::FwhtElem::BF16, impl_e, /*wait=*/false);
    } else {
        TORCH_CHECK(false, "afi.fwht: dtypes are fp32/fp16/bf16, got ",
                    src.scalar_type());
    }
    return work_out.view(x.sizes());
}

at::Tensor fwht_meta(const at::Tensor &x, std::optional<double> scale,
                     std::string_view impl) {
    (void)scale;
    (void)impl;
    check_last_dim(x);
    return at::empty_like(x);
}

bool xmx_ok_xpu() {
    auto &q = c10::xpu::getCurrentXPUStream().queue();
    return afi::fwht_xmx_ok(q.get_device());
}

} // namespace

TORCH_LIBRARY(afi, m) {
    m.def("fwht(Tensor x, float? scale=None, str impl=\"auto\") -> Tensor");
    m.def("xmx_ok() -> bool", &xmx_ok_xpu);
}

TORCH_LIBRARY_IMPL(afi, CPU, m) {
    m.impl("fwht", &fwht_cpu);
}

TORCH_LIBRARY_IMPL(afi, XPU, m) {
    m.impl("fwht", &fwht_xpu);
}

TORCH_LIBRARY_IMPL(afi, Meta, m) {
    m.impl("fwht", &fwht_meta);
}
