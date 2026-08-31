// libafi_rtu — C API over the B70 RTU primitives for python/afi/rtu.py.
// Heightfield BLAS, generic triangle mesh BLAS (room-IR), LOS occlusion,
// closest-hit. No agent logic here.

#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <mutex>
#include <string>
#include <vector>

#include "afi/rttrace.h"

namespace {

#define AFI_ZE_CHECK_RET(expr)                                                \
  do {                                                                        \
    ze_result_t r_ = (expr);                                                  \
    if (r_ != ZE_RESULT_SUCCESS) return -int(r_);                             \
  } while (0)
#define AFI_ZE_CHECK_NULL(expr)                                               \
  do {                                                                        \
    ze_result_t r_ = (expr);                                                  \
    if (r_ != ZE_RESULT_SUCCESS) return nullptr;                              \
  } while (0)

using FnBuilderCreate = ze_result_t (*)(ze_driver_handle_t,
                                        const ze_rtas_builder_exp_desc_t *,
                                        ze_rtas_builder_exp_handle_t *);
using FnBuilderDestroy = ze_result_t (*)(ze_rtas_builder_exp_handle_t);
using FnBuilderProps = ze_result_t (*)(ze_rtas_builder_exp_handle_t,
                                       ze_rtas_builder_build_op_exp_desc_t *,
                                       ze_rtas_builder_exp_properties_t *);
using FnBuilderBuild = ze_result_t (*)(ze_rtas_builder_exp_handle_t,
                                       const ze_rtas_builder_build_op_exp_desc_t *,
                                       void *, size_t, void *, size_t,
                                       ze_rtas_parallel_operation_exp_handle_t,
                                       void *, ze_rtas_aabb_exp_t *, size_t *);
using FnParCreate = ze_result_t (*)(ze_driver_handle_t,
                                    ze_rtas_parallel_operation_exp_handle_t *);
using FnParDestroy = ze_result_t (*)(ze_rtas_parallel_operation_exp_handle_t);
using FnParJoin = ze_result_t (*)(ze_rtas_parallel_operation_exp_handle_t);

template <typename Fn>
Fn ze_fn(const char *name) {
  static void *loader = [] {
    void *h = dlopen("libze_loader.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (!h) throw std::runtime_error("dlopen libze_loader.so.1 failed");
    return h;
  }();
  void *p = dlsym(loader, name);
  if (!p) throw std::runtime_error(std::string("dlsym ") + name + " failed");
  return reinterpret_cast<Fn>(p);
}

struct RtuState {
  sycl::queue q;
  ze_driver_handle_t hdrv = nullptr;
  ze_device_handle_t hdev = nullptr;
  bool ok = false;

  static RtuState &get() {
    static RtuState s;
    return s;
  }
};

struct RtuBlas {
  char *rtas = nullptr;
  size_t rtas_bytes = 0;
  intel_raytracing_acceleration_structure_t *accel_cell = nullptr;
};

struct RtuOcclKernel {
  const intel_raytracing_acceleration_structure_t *accel;
  const float *rays;
  const float *tmax;
  int32_t *out;
  void operator()(sycl::nd_item<1> it) const {
#ifdef __SYCL_DEVICE_ONLY__
    const size_t li = it.get_global_linear_id();
    const float *r = rays + li * 6;
    intel_ray_desc_t rd;
    rd.origin = intel_float3(r[0], r[1], r[2]);
    rd.direction = intel_float3(r[3], r[4], r[5]);
    rd.tmin = 1e-3f;
    rd.tmax = tmax[li];
    rd.mask = 0xFF;
    rd.flags = intel_ray_flags_accept_first_hit_and_end_search;
    intel_ray_query_t q = intel_ray_query_init(rd, *accel);
    intel_ray_query_start_traversal(q);
    intel_ray_query_sync(q);
    int32_t hit = 0;
    if (intel_is_traversal_done(q) && intel_has_committed_hit(q)) hit = 1;
    intel_ray_query_abandon(q);
    out[li] = hit;
#endif
  }
};

struct RtuHitKernel {
  const intel_raytracing_acceleration_structure_t *accel;
  const float *rays;
  const float *tmax;
  float *out_t;
  int32_t *out_prim;
  void operator()(sycl::nd_item<1> it) const {
#ifdef __SYCL_DEVICE_ONLY__
    const size_t li = it.get_global_linear_id();
    const float *r = rays + li * 6;
    intel_ray_desc_t rd;
    rd.origin = intel_float3(r[0], r[1], r[2]);
    rd.direction = intel_float3(r[3], r[4], r[5]);
    rd.tmin = 1e-3f;
    rd.tmax = tmax[li];
    rd.mask = 0xFF;
    rd.flags = intel_ray_flags_none;
    intel_ray_query_t q = intel_ray_query_init(rd, *accel);
    intel_ray_query_start_traversal(q);
    intel_ray_query_sync(q);
    float t = -1.f;
    int32_t prim = -1;
    if (intel_is_traversal_done(q) && intel_has_committed_hit(q)) {
      t = intel_get_hit_distance(q, intel_hit_type_committed_hit);
      prim = int32_t(intel_get_hit_primitive_id(q, intel_hit_type_committed_hit));
    }
    intel_ray_query_abandon(q);
    out_t[li] = t;
    out_prim[li] = prim;
#endif
  }
};

// heightfield (G+1)^2 -> verts + L0 quad-split triangles
void mesh_from_heights(const float *h, uint32_t G, float span,
                       std::vector<float> &verts, std::vector<uint32_t> &idx) {
  const uint32_t G1 = G + 1;
  verts.resize(size_t(G1) * G1 * 3);
  for (uint32_t j = 0; j <= G; ++j)
    for (uint32_t i = 0; i <= G; ++i) {
      float *v = &verts[(size_t(j) * G1 + i) * 3];
      v[0] = float(i) / float(G) * span;
      v[1] = h[size_t(j) * G1 + i];
      v[2] = float(j) / float(G) * span;
    }
  idx.reserve(size_t(2) * G * G * 3);
  for (uint32_t j = 0; j < G; ++j)
    for (uint32_t i = 0; i < G; ++i) {
      const uint32_t v00 = j * G1 + i, v10 = v00 + 1;
      const uint32_t v01 = v00 + G1, v11 = v01 + 1;
      idx.push_back(v00); idx.push_back(v10); idx.push_back(v01);
      idx.push_back(v11); idx.push_back(v01); idx.push_back(v10);
    }
}

// Generic triangle BLAS. verts: nverts * 3 float32, idx: nidx uint32
// (nidx = 3 * triangle count). Shared by heightfield and room meshes.
void *build_from_host_mesh(const float *verts, uint32_t nverts,
                           const uint32_t *idx, uint32_t nidx,
                           double *build_ms_out) {
  RtuState &s = RtuState::get();
  if (!s.ok || !verts || !idx || nverts < 3 || nidx < 3 || nidx % 3)
    return nullptr;

  float *dverts = sycl::malloc_shared<float>(size_t(nverts) * 3, s.q);
  uint32_t *didx = sycl::malloc_shared<uint32_t>(nidx, s.q);
  std::memcpy(dverts, verts, size_t(nverts) * 3 * sizeof(float));
  std::memcpy(didx, idx, size_t(nidx) * sizeof(uint32_t));

  auto fnCreate = ze_fn<FnBuilderCreate>("zeRTASBuilderCreateExp");
  auto fnDestroy = ze_fn<FnBuilderDestroy>("zeRTASBuilderDestroyExp");
  auto fnProps = ze_fn<FnBuilderProps>("zeRTASBuilderGetBuildPropertiesExp");
  auto fnBuild = ze_fn<FnBuilderBuild>("zeRTASBuilderBuildExp");
  auto fnParCreate = ze_fn<FnParCreate>("zeRTASParallelOperationCreateExp");
  auto fnParDestroy = ze_fn<FnParDestroy>("zeRTASParallelOperationDestroyExp");
  auto fnParJoin = ze_fn<FnParJoin>("zeRTASParallelOperationJoinExp");

  ze_rtas_device_exp_properties_t rtas_dev{};
  rtas_dev.stype = ZE_STRUCTURE_TYPE_RTAS_DEVICE_EXP_PROPERTIES;
  ze_device_properties_t dev_props{};
  dev_props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
  dev_props.pNext = &rtas_dev;
  if (zeDeviceGetProperties(s.hdev, &dev_props) != ZE_RESULT_SUCCESS) {
    sycl::free(dverts, s.q);
    sycl::free(didx, s.q);
    return nullptr;
  }

  ze_rtas_builder_exp_desc_t bdesc{};
  bdesc.stype = ZE_STRUCTURE_TYPE_RTAS_BUILDER_EXP_DESC;
  bdesc.builderVersion = ZE_RTAS_BUILDER_EXP_VERSION_CURRENT;
  ze_rtas_builder_exp_handle_t hbuilder = nullptr;
  if (fnCreate(s.hdrv, &bdesc, &hbuilder) != ZE_RESULT_SUCCESS) {
    sycl::free(dverts, s.q);
    sycl::free(didx, s.q);
    return nullptr;
  }

  ze_rtas_builder_triangles_geometry_info_exp_t geom{};
  geom.geometryType = ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_TRIANGLES;
  geom.geometryMask = 0xFF;
  geom.triangleFormat =
      ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_TRIANGLE_INDICES_UINT32;
  geom.vertexFormat = ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_FLOAT3;
  geom.triangleCount = nidx / 3;
  geom.vertexCount = nverts;
  geom.triangleStride = 3 * sizeof(uint32_t);
  geom.vertexStride = 3 * sizeof(float);
  geom.pTriangleBuffer = didx;
  geom.pVertexBuffer = dverts;

  const ze_rtas_builder_geometry_info_exp_t *geoms[] = {
      reinterpret_cast<const ze_rtas_builder_geometry_info_exp_t *>(&geom)};
  ze_rtas_parallel_operation_exp_handle_t hpar = nullptr;
  if (fnParCreate(s.hdrv, &hpar) != ZE_RESULT_SUCCESS) {
    fnDestroy(hbuilder);
    sycl::free(dverts, s.q);
    sycl::free(didx, s.q);
    return nullptr;
  }

  ze_rtas_builder_build_op_exp_desc_t op{};
  op.stype = ZE_STRUCTURE_TYPE_RTAS_BUILDER_BUILD_OP_EXP_DESC;
  op.rtasFormat = rtas_dev.rtasFormat;
  op.buildQuality = ZE_RTAS_BUILDER_BUILD_QUALITY_HINT_EXP_MEDIUM;
  op.buildFlags = ZE_RTAS_BUILDER_BUILD_OP_EXP_FLAG_NO_DUPLICATE_ANYHIT_INVOCATION;
  op.ppGeometries = geoms;
  op.numGeometries = 1;

  ze_rtas_builder_exp_properties_t sizes{};
  sizes.stype = ZE_STRUCTURE_TYPE_RTAS_BUILDER_EXP_PROPERTIES;
  if (fnProps(hbuilder, &op, &sizes) != ZE_RESULT_SUCCESS) {
    fnParDestroy(hpar);
    fnDestroy(hbuilder);
    sycl::free(dverts, s.q);
    sycl::free(didx, s.q);
    return nullptr;
  }

  void *scratch = sycl::malloc_shared<char>(sizes.scratchBufferSizeBytes, s.q);
  size_t rtas_bytes = sizes.rtasBufferSizeBytesExpected;
  char *rtas = nullptr;
  const double t0 = std::chrono::duration<double>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count();
  ze_result_t br = ZE_RESULT_ERROR_UNKNOWN;
  for (;;) {
    rtas = static_cast<char *>(sycl::malloc_shared<char>(rtas_bytes, s.q));
    br = fnBuild(hbuilder, &op, scratch, sizes.scratchBufferSizeBytes, rtas,
                 rtas_bytes, hpar, nullptr, nullptr, nullptr);
    if (br == ZE_RESULT_EXP_RTAS_BUILD_RETRY) {
      rtas_bytes = (rtas_bytes * 3 / 2 + 127) & ~size_t(127);
      sycl::free(rtas, s.q);
      continue;
    }
    break;
  }
  if (br == ZE_RESULT_EXP_RTAS_BUILD_DEFERRED) {
    if (fnParJoin(hpar) != ZE_RESULT_SUCCESS) {
      fnParDestroy(hpar);
      fnDestroy(hbuilder);
      sycl::free(scratch, s.q);
      sycl::free(dverts, s.q);
      sycl::free(didx, s.q);
      sycl::free(rtas, s.q);
      return nullptr;
    }
    br = ZE_RESULT_SUCCESS;
  }
  const double t1 = std::chrono::duration<double>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count();
  fnParDestroy(hpar);
  fnDestroy(hbuilder);
  sycl::free(scratch, s.q);
  sycl::free(dverts, s.q);
  sycl::free(didx, s.q);
  if (br != ZE_RESULT_SUCCESS) return nullptr;
  if (build_ms_out) *build_ms_out = (t1 - t0) * 1e3;

  auto *blas = new RtuBlas();
  blas->rtas = rtas;
  blas->rtas_bytes = rtas_bytes;
  blas->accel_cell =
      sycl::malloc_shared<intel_raytracing_acceleration_structure_t>(1, s.q);
  void *bits = rtas;
  std::memcpy(blas->accel_cell, &bits, sizeof(void *));
  return blas;
}

}  // namespace

extern "C" {

int rtu_init(int gpu_index) {
  static std::mutex mtx;
  std::lock_guard<std::mutex> lock(mtx);
  RtuState &s = RtuState::get();
  if (s.ok) return 0;
  try {
    AFI_ZE_CHECK_RET(zeInit(0));
    std::vector<sycl::device> gpus;
    for (const auto &p : sycl::platform::get_platforms())
      if (p.get_backend() == sycl::backend::ext_oneapi_level_zero)
        for (const auto &d : p.get_devices())
          if (d.is_gpu()) gpus.push_back(d);
    if (gpu_index < 0 || size_t(gpu_index) >= gpus.size()) return -2;
    s.q = sycl::queue{gpus[size_t(gpu_index)]};
    s.hdrv = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
        gpus[size_t(gpu_index)].get_platform());
    s.hdev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
        gpus[size_t(gpu_index)]);
    // force JIT/context realization before first RTAS use (M1 landmine)
    auto *sink = sycl::malloc_device<int>(1, s.q);
    s.q.single_task([=]() { *sink = 42; }).wait();
    sycl::free(sink, s.q);
    // resolve builder symbols once (proves the dlsym path early)
    ze_fn<FnBuilderCreate>("zeRTASBuilderCreateExp");
    s.ok = true;
    return 0;
  } catch (...) {
    return -1000;
  }
}

int rtu_device_name(char *buf, int buflen) {
  RtuState &s = RtuState::get();
  if (!s.ok || !buf || buflen <= 0) return -1;
  std::string name = s.q.get_device().get_info<sycl::info::device::name>();
  std::snprintf(buf, size_t(buflen), "%s", name.c_str());
  return 0;
}

// heights: (G+1)^2 row-major float32; returns opaque BLAS handle or NULL.
void *rtu_blas_build(const float *heights, int G, float span,
                     double *build_ms_out) {
  RtuState &s = RtuState::get();
  if (!s.ok || !heights || G < 1) return nullptr;
  try {
    std::vector<float> verts;
    std::vector<uint32_t> idx;
    mesh_from_heights(heights, uint32_t(G), span, verts, idx);
    return build_from_host_mesh(verts.data(), uint32_t(verts.size() / 3),
                                idx.data(), uint32_t(idx.size()),
                                build_ms_out);
  } catch (...) {
    return nullptr;
  }
}

// Generic triangle mesh: verts nverts*3 float32, idx ntri*3 uint32.
void *rtu_mesh_build(const float *verts, int nverts, const uint32_t *idx,
                     int ntri, double *build_ms_out) {
  RtuState &s = RtuState::get();
  if (!s.ok || !verts || !idx || nverts < 3 || ntri < 1) return nullptr;
  try {
    return build_from_host_mesh(verts, uint32_t(nverts), idx,
                                uint32_t(ntri) * 3, build_ms_out);
  } catch (...) {
    return nullptr;
  }
}

void rtu_blas_free(void *handle) {
  if (!handle) return;
  auto *blas = static_cast<RtuBlas *>(handle);
  RtuState &s = RtuState::get();
  if (s.ok) {
    sycl::free(blas->rtas, s.q);
    sycl::free(blas->accel_cell, s.q);
  }
  delete blas;
}

size_t rtu_blas_bytes(void *handle) {
  if (!handle) return 0;
  return static_cast<RtuBlas *>(handle)->rtas_bytes;
}

// rays6: 6*n floats (origin+unit dir); tmax: n floats (segment cap).
// out_hit: n int32 (0/1). Returns 0 or negative error.
int rtu_occlusion(void *handle, const float *rays6, const float *tmax, int n,
                  int32_t *out_hit) {
  RtuState &s = RtuState::get();
  auto *blas = static_cast<RtuBlas *>(handle);
  if (!s.ok || !blas || !rays6 || !tmax || !out_hit || n <= 0) return -1;
  try {
    float *drays = sycl::malloc_device<float>(size_t(n) * 6, s.q);
    float *dtmax = sycl::malloc_device<float>(size_t(n), s.q);
    int32_t *dout = sycl::malloc_device<int32_t>(size_t(n), s.q);
    s.q.memcpy(drays, rays6, size_t(n) * 6 * sizeof(float)).wait();
    s.q.memcpy(dtmax, tmax, size_t(n) * sizeof(float)).wait();
    const uint32_t wg = 64;
    const size_t nw = (size_t(n) + wg - 1) / wg * wg;
    s.q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<class RtuApiOccl>(
          sycl::nd_range<1>(sycl::range<1>(nw), sycl::range<1>(wg)),
          RtuOcclKernel{blas->accel_cell, drays, dtmax, dout});
    }).wait();
    s.q.memcpy(out_hit, dout, size_t(n) * sizeof(int32_t)).wait();
    sycl::free(drays, s.q);
    sycl::free(dtmax, s.q);
    sycl::free(dout, s.q);
    return 0;
  } catch (...) {
    return -1000;
  }
}

// out_t: n floats (-1 on miss); out_prim: n int32 (-1 on miss).
int rtu_horizon(void *handle, const float *rays6, const float *tmax, int n,
                float *out_t, int32_t *out_prim) {
  RtuState &s = RtuState::get();
  auto *blas = static_cast<RtuBlas *>(handle);
  if (!s.ok || !blas || !rays6 || !tmax || !out_t || !out_prim || n <= 0)
    return -1;
  try {
    float *drays = sycl::malloc_device<float>(size_t(n) * 6, s.q);
    float *dtmax = sycl::malloc_device<float>(size_t(n), s.q);
    float *dt = sycl::malloc_device<float>(size_t(n), s.q);
    int32_t *dprim = sycl::malloc_device<int32_t>(size_t(n), s.q);
    s.q.memcpy(drays, rays6, size_t(n) * 6 * sizeof(float)).wait();
    s.q.memcpy(dtmax, tmax, size_t(n) * sizeof(float)).wait();
    const uint32_t wg = 64;
    const size_t nw = (size_t(n) + wg - 1) / wg * wg;
    s.q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<class RtuApiHorizon>(
          sycl::nd_range<1>(sycl::range<1>(nw), sycl::range<1>(wg)),
          RtuHitKernel{blas->accel_cell, drays, dtmax, dt, dprim});
    }).wait();
    s.q.memcpy(out_t, dt, size_t(n) * sizeof(float)).wait();
    s.q.memcpy(out_prim, dprim, size_t(n) * sizeof(int32_t)).wait();
    sycl::free(drays, s.q);
    sycl::free(dtmax, s.q);
    sycl::free(dt, s.q);
    sycl::free(dprim, s.q);
    return 0;
  } catch (...) {
    return -1000;
  }
}

}  // extern "C"
