#pragma once

#include <sycl/sycl.hpp>

#include <string>
#include <vector>

namespace afi {

struct MatrixCombo {
    size_t m = 0, n = 0, k = 0;
    size_t max_m = 0, max_n = 0, max_k = 0;
    std::string atype, btype, ctype, dtype;
};

struct DeviceInfo {
    int index = -1;
    std::string name;
    std::string vendor;
    std::string version;
    std::string driver;
    std::string backend;
    uint64_t global_mem = 0;
    uint64_t local_mem = 0;
    size_t max_wg = 0;
    uint32_t max_cu = 0;
    uint32_t max_clock_mhz = 0;
    std::vector<size_t> subgroup_sizes;
    std::vector<std::string> aspects;
    bool ext_intel_matrix = false;
    bool is_igpu = false;
    std::vector<MatrixCombo> matrix_combos;
    std::string matrix_query_note;
};

bool looks_like_igpu(const std::string &name);

// Level Zero GPUs only. iGPU (UHD) skipped unless include_igpu.
std::vector<sycl::device> dgpu_devices(bool include_igpu = false);

sycl::device pick_dgpu(int index = 0);

DeviceInfo inspect(const sycl::device &dev, int index);

std::string to_json(const std::vector<DeviceInfo> &devs);

} // namespace afi
