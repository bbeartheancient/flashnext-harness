#include "afi/device.hpp"

#include <sycl/ext/oneapi/matrix/matrix.hpp>

#include <sstream>
#include <stdexcept>

namespace afi {
namespace {

const char *matrix_type_name(sycl::ext::oneapi::experimental::matrix::matrix_type t) {
    using sycl::ext::oneapi::experimental::matrix::matrix_type;
    switch (t) {
    case matrix_type::bf16: return "bf16";
    case matrix_type::fp16: return "fp16";
    case matrix_type::tf32: return "tf32";
    case matrix_type::fp32: return "fp32";
    case matrix_type::fp64: return "fp64";
    case matrix_type::sint8: return "sint8";
    case matrix_type::sint16: return "sint16";
    case matrix_type::sint32: return "sint32";
    case matrix_type::sint64: return "sint64";
    case matrix_type::uint8: return "uint8";
    case matrix_type::uint16: return "uint16";
    case matrix_type::uint32: return "uint32";
    case matrix_type::uint64: return "uint64";
    }
    return "?";
}

std::string json_escape(const std::string &s) {
    std::string o;
    o.reserve(s.size());
    for (char c : s) {
        if (c == '"' || c == '\\') {
            o.push_back('\\');
            o.push_back(c);
        } else if (c == '\n') {
            o += "\\n";
        } else {
            o.push_back(c);
        }
    }
    return o;
}

std::string backend_name(sycl::backend b) {
    switch (b) {
    case sycl::backend::ext_oneapi_level_zero: return "level_zero";
    case sycl::backend::opencl: return "opencl";
    default: return "other";
    }
}

void add_aspect(DeviceInfo &info, const sycl::device &dev, sycl::aspect a,
                const char *name) {
    if (dev.has(a)) info.aspects.emplace_back(name);
}

} // namespace

bool looks_like_igpu(const std::string &name) {
    if (name.find("Arc") != std::string::npos) return false;
    return name.find("UHD") != std::string::npos ||
           name.find("Iris") != std::string::npos;
}

std::vector<sycl::device> dgpu_devices(bool include_igpu) {
    std::vector<sycl::device> out;
    for (const auto &d : sycl::device::get_devices(sycl::info::device_type::gpu)) {
        if (d.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        const auto name = d.get_info<sycl::info::device::name>();
        if (!include_igpu && looks_like_igpu(name)) continue;
        out.push_back(d);
    }
    return out;
}

sycl::device pick_dgpu(int index) {
    auto devs = dgpu_devices(false);
    if (devs.empty()) {
        throw std::runtime_error("no Level Zero dGPU (is ONEAPI_DEVICE_SELECTOR set?)");
    }
    if (index < 0 || static_cast<size_t>(index) >= devs.size()) {
        throw std::runtime_error("dGPU index out of range");
    }
    return devs[static_cast<size_t>(index)];
}

DeviceInfo inspect(const sycl::device &dev, int index) {
    DeviceInfo info;
    info.index = index;
    info.name = dev.get_info<sycl::info::device::name>();
    info.vendor = dev.get_info<sycl::info::device::vendor>();
    info.version = dev.get_info<sycl::info::device::version>();
    info.driver = dev.get_info<sycl::info::device::driver_version>();
    info.backend = backend_name(dev.get_backend());
    info.global_mem = dev.get_info<sycl::info::device::global_mem_size>();
    info.local_mem = dev.get_info<sycl::info::device::local_mem_size>();
    info.max_wg = dev.get_info<sycl::info::device::max_work_group_size>();
    info.max_cu = dev.get_info<sycl::info::device::max_compute_units>();
    info.max_clock_mhz = dev.get_info<sycl::info::device::max_clock_frequency>();
    info.is_igpu = looks_like_igpu(info.name);
    try {
        info.subgroup_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
    } catch (...) {
    }

    add_aspect(info, dev, sycl::aspect::fp16, "fp16");
    add_aspect(info, dev, sycl::aspect::fp64, "fp64");
    add_aspect(info, dev, sycl::aspect::atomic64, "atomic64");
    add_aspect(info, dev, sycl::aspect::usm_device_allocations, "usm_device");
    add_aspect(info, dev, sycl::aspect::usm_shared_allocations, "usm_shared");
    add_aspect(info, dev, sycl::aspect::ext_intel_matrix, "ext_intel_matrix");
    info.ext_intel_matrix = dev.has(sycl::aspect::ext_intel_matrix);

    if (info.ext_intel_matrix) {
        try {
            using namespace sycl::ext::oneapi::experimental;
            auto combos = dev.get_info<info::device::matrix_combinations>();
            info.matrix_combos.reserve(combos.size());
            for (const auto &c : combos) {
                MatrixCombo mc;
                mc.m = c.msize;
                mc.n = c.nsize;
                mc.k = c.ksize;
                mc.max_m = c.max_msize;
                mc.max_n = c.max_nsize;
                mc.max_k = c.max_ksize;
                mc.atype = matrix_type_name(c.atype);
                mc.btype = matrix_type_name(c.btype);
                mc.ctype = matrix_type_name(c.ctype);
                mc.dtype = matrix_type_name(c.dtype);
                info.matrix_combos.push_back(mc);
            }
            info.matrix_query_note = "ok";
        } catch (const std::exception &e) {
            info.matrix_query_note = e.what();
        }
    } else {
        info.matrix_query_note = "aspect ext_intel_matrix not present";
    }
    return info;
}

std::string to_json(const std::vector<DeviceInfo> &devs) {
    std::ostringstream o;
    o << "{\n  \"devices\": [\n";
    for (size_t i = 0; i < devs.size(); ++i) {
        const auto &d = devs[i];
        o << "    {\n";
        o << "      \"index\": " << d.index << ",\n";
        o << "      \"name\": \"" << json_escape(d.name) << "\",\n";
        o << "      \"vendor\": \"" << json_escape(d.vendor) << "\",\n";
        o << "      \"version\": \"" << json_escape(d.version) << "\",\n";
        o << "      \"driver\": \"" << json_escape(d.driver) << "\",\n";
        o << "      \"backend\": \"" << json_escape(d.backend) << "\",\n";
        o << "      \"is_igpu\": " << (d.is_igpu ? "true" : "false") << ",\n";
        o << "      \"global_mem_bytes\": " << d.global_mem << ",\n";
        o << "      \"local_mem_bytes\": " << d.local_mem << ",\n";
        o << "      \"max_work_group_size\": " << d.max_wg << ",\n";
        o << "      \"max_compute_units\": " << d.max_cu << ",\n";
        o << "      \"max_clock_mhz\": " << d.max_clock_mhz << ",\n";
        o << "      \"ext_intel_matrix\": " << (d.ext_intel_matrix ? "true" : "false")
          << ",\n";
        o << "      \"matrix_query\": \"" << json_escape(d.matrix_query_note) << "\",\n";
        o << "      \"subgroup_sizes\": [";
        for (size_t s = 0; s < d.subgroup_sizes.size(); ++s) {
            if (s) o << ", ";
            o << d.subgroup_sizes[s];
        }
        o << "],\n      \"aspects\": [";
        for (size_t s = 0; s < d.aspects.size(); ++s) {
            if (s) o << ", ";
            o << "\"" << d.aspects[s] << "\"";
        }
        o << "],\n      \"matrix_combinations\": [\n";
        for (size_t c = 0; c < d.matrix_combos.size(); ++c) {
            const auto &m = d.matrix_combos[c];
            o << "        {\"m\": " << m.m << ", \"n\": " << m.n << ", \"k\": " << m.k
              << ", \"max_m\": " << m.max_m << ", \"max_n\": " << m.max_n
              << ", \"max_k\": " << m.max_k << ", \"a\": \"" << m.atype << "\", \"b\": \""
              << m.btype << "\", \"c\": \"" << m.ctype << "\", \"d\": \"" << m.dtype
              << "\"}";
            o << (c + 1 < d.matrix_combos.size() ? ",\n" : "\n");
        }
        o << "      ]\n    }";
        o << (i + 1 < devs.size() ? ",\n" : "\n");
    }
    o << "  ]\n}\n";
    return o.str();
}

} // namespace afi
