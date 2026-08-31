// afi-rtu-probe: B70 ray-tracing-unit availability, correctness, and
// first numbers for the hoa64 RTU backend track (docs/hoa64-rtu-backend.md).
//
// Host: L0 RTAS builder (ZE_extension_rtas, EXT v1.0) builds a triangle-mesh
//       BLAS. Device: Embree-style intel_ray_query_* builtins (afi/rttrace.h)
//       lowered by IGC to RTU traversal messages. Correctness oracle is a
//       host Moeller-Trumbore sweep. No SYCL ray-query extension involved.
//
// Usage: afi-rtu-probe [gpu_index] [n_tri] [n_correctness_rays] [n_timing_rays]

#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "afi/rttrace.h"

namespace {

#define AFI_ZE_CHECK(expr)                                                    \
  do {                                                                        \
    ze_result_t r_ = (expr);                                                  \
    if (r_ != ZE_RESULT_SUCCESS) {                                            \
      std::fprintf(stderr, "ze error 0x%x at %s:%d (%s)\n", r_, __FILE__,     \
                   __LINE__, #expr);                                          \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

// --- L0 RTAS builder EXP functions: dlsym'd straight off the loader --------
// (Embree's ZeWrapper pattern; zeDriverGetExtensionFunctionAddress does not
// resolve these names on the current loader<->driver pairing.)

#include <dlfcn.h>

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
    if (!h) {
      std::fprintf(stderr, "dlopen libze_loader.so.1: %s\n", dlerror());
      std::exit(1);
    }
    return h;
  }();
  void *p = dlsym(loader, name);
  if (!p) {
    std::fprintf(stderr, "dlsym %s failed\n", name);
    std::exit(1);
  }
  return reinterpret_cast<Fn>(p);
}

// --- scene / rays -----------------------------------------------------------

struct Scene {
  uint32_t grid = 16;               // G x G cells -> G*G*2 triangles
  float cell = 1.0f;
  std::vector<float> verts;         // 3 * V
  std::vector<uint32_t> idx;        // 3 * T
  uint32_t tris() const { return grid * grid * 2; }
  uint32_t nverts() const { return (grid + 1) * (grid + 1); }

  void build(float bump) {
    const uint32_t G = grid;
    verts.reserve(size_t(nverts()) * 3);
    for (uint32_t j = 0; j <= G; ++j)
      for (uint32_t i = 0; i <= G; ++i) {
        float h = 0.0f;
        if (bump > 0.0f && i >= 4 && i <= 12 && j >= 4 && j <= 12) {
          const float dx = float(i) - 8.0f, dy = float(j) - 8.0f;
          h = bump * std::exp(-(dx * dx + dy * dy) / 8.0f);
        }
        verts.push_back(float(i) * cell);
        verts.push_back(h);
        verts.push_back(float(j) * cell);
      }
    idx.reserve(size_t(tris()) * 3);
    for (uint32_t j = 0; j < G; ++j)
      for (uint32_t i = 0; i < G; ++i) {
        const uint32_t v00 = j * (G + 1) + i, v10 = v00 + 1;
        const uint32_t v01 = v00 + (G + 1), v11 = v01 + 1;
        idx.push_back(v00); idx.push_back(v11); idx.push_back(v10);
        idx.push_back(v00); idx.push_back(v01); idx.push_back(v11);
      }
  }
};

// Moeller-Trumbore nearest hit; returns true on hit.
bool host_nearest(const Scene &sc, const float o[3], const float d[3],
                  float *t_hit, uint32_t *prim) {
  const float eps = 1e-6f;
  bool found = false;
  float best = 1e30f;
  uint32_t best_p = 0;
  for (uint32_t t = 0; t < sc.tris(); ++t) {
    const float *v0 = &sc.verts[size_t(sc.idx[3 * t + 0]) * 3];
    const float *v1 = &sc.verts[size_t(sc.idx[3 * t + 1]) * 3];
    const float *v2 = &sc.verts[size_t(sc.idx[3 * t + 2]) * 3];
    float e1[3] = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
    float e2[3] = {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};
    float p[3] = {d[1] * e2[2] - d[2] * e2[1], d[2] * e2[0] - d[0] * e2[2],
                  d[0] * e2[1] - d[1] * e2[0]};
    float det = e1[0] * p[0] + e1[1] * p[1] + e1[2] * p[2];
    if (det > -eps && det < eps) continue;
    const float inv = 1.0f / det;
    float s[3] = {o[0] - v0[0], o[1] - v0[1], o[2] - v0[2]};
    float u = (s[0] * p[0] + s[1] * p[1] + s[2] * p[2]) * inv;
    if (u < -eps || u > 1.0f + eps) continue;
    float q[3] = {s[1] * e1[2] - s[2] * e1[1], s[2] * e1[0] - s[0] * e1[2],
                  s[0] * e1[1] - s[1] * e1[0]};
    float v = (d[0] * q[0] + d[1] * q[1] + d[2] * q[2]) * inv;
    if (v < -eps || u + v > 1.0f + eps) continue;
    float tt = (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]) * inv;
    if (tt > eps && tt < best) {
      best = tt;
      best_p = t;
      found = true;
    }
  }
  if (found) {
    *t_hit = best;
    *prim = best_p;
  }
  return found;
}

struct RayGen {
  std::mt19937 rng{1234};
  std::uniform_real_distribution<float> u01{0.0f, 1.0f};
  const Scene *sc;

  // 3/4 of rays aim at a random point on the terrain (hits); 1/4 aim past
  // the terrain edge or upward (misses). Origin hovers above the center.
  void ray(float *o, float *d, bool *expect_hit) {
    const float span = sc->grid * sc->cell;
    o[0] = span * 0.5f;
    o[1] = 6.0f;
    o[2] = span * 0.5f;
    float tx, ty, tz;
    if (u01(rng) < 0.75f) {
      tx = u01(rng) * span;
      tz = u01(rng) * span;
      ty = 0.0f;
      *expect_hit = true;
    } else {
      const float off = span * (1.1f + 0.6f * u01(rng));
      if (u01(rng) < 0.5f) {
        tx = -off + u01(rng) * 0.2f * span;
        tz = u01(rng) * span;
      } else {
        tx = u01(rng) * span;
        tz = -off;
      }
      ty = 0.0f;
      *expect_hit = false;
    }
    d[0] = tx - o[0];
    d[1] = ty - o[1];
    d[2] = tz - o[2];
    const float len =
        std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    d[0] /= len; d[1] /= len; d[2] /= len;
  }
};

struct HitOut {
  float t;
  uint32_t prim;
  uint32_t hit;
};

struct RtuProbeKernel {
  const intel_raytracing_acceleration_structure_t *accel;
  const float *rays;      // 6 floats per ray: origin + unit direction
  HitOut *out;

  void operator()(sycl::nd_item<1> it) const {
#ifdef __SYCL_DEVICE_ONLY__
    const size_t li = it.get_global_linear_id();
    const float *r = rays + li * 6;
    intel_ray_desc_t rd;
    rd.origin = intel_float3(r[0], r[1], r[2]);
    rd.direction = intel_float3(r[3], r[4], r[5]);
    rd.tmin = 1e-3f;
    rd.tmax = 1e30f;
    rd.mask = 0xFF;
    rd.flags = intel_ray_flags_none;

    intel_ray_query_t q = intel_ray_query_init(rd, *accel);
    intel_ray_query_start_traversal(q);
    intel_ray_query_sync(q);
    HitOut h{-1.0f, 0xFFFFFFFFu, 0u};
    if (intel_is_traversal_done(q) && intel_has_committed_hit(q)) {
      h.t = intel_get_hit_distance(q, intel_hit_type_committed_hit);
      h.prim = intel_get_hit_primitive_id(q, intel_hit_type_committed_hit);
      h.hit = 1u;
    }
    intel_ray_query_abandon(q);
    out[li] = h;
#endif
  }
};

double now_ms() {
  using clk = std::chrono::steady_clock;
  return std::chrono::duration<double, std::milli>(
             clk::now().time_since_epoch())
      .count();
}

}  // namespace

int main(int argc, char **argv) {
  const int gpu_idx = argc > 1 ? std::atoi(argv[1]) : 1;
  const uint32_t n_rays_correct = argc > 2 ? std::atoi(argv[2]) : 4096;
  const uint32_t n_rays_time = argc > 3 ? std::atol(argv[3]) : 1u << 20;

  AFI_ZE_CHECK(zeInit(0));

  // pick the L0 GPU device
  std::vector<sycl::device> gpus;
  for (const auto &p : sycl::platform::get_platforms())
    if (p.get_backend() == sycl::backend::ext_oneapi_level_zero)
      for (const auto &d : p.get_devices())
        if (d.is_gpu()) gpus.push_back(d);
  if (gpus.empty()) {
    std::fprintf(stderr, "no level_zero:gpu devices\n");
    return 1;
  }
  if (gpu_idx < 0 || size_t(gpu_idx) >= gpus.size()) {
    std::fprintf(stderr, "gpu index %d out of range (%zu L0 GPUs)\n", gpu_idx,
                 gpus.size());
    return 1;
  }
  const sycl::device &dev = gpus[size_t(gpu_idx)];
  sycl::queue q{dev};
  std::printf("device[%d]: %s\n", gpu_idx, dev.get_info<sycl::info::device::name>().c_str());

  // Force real context/module creation on the device before touching the
  // RTAS extension: the driver pairs its raytracing support lib lazily and
  // the tutorial evidence says this happens only once JIT machinery is live.
  {
    auto *sink = sycl::malloc_device<int>(1, q);
    q.single_task([=]() { *sink = 42; }).wait();
    int host_check = 0;
    q.memcpy(&host_check, sink, sizeof(int)).wait();
    sycl::free(sink, q);
    std::printf("warmup kernel: %d\n", host_check);
  }

  ze_driver_handle_t hdrv =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(dev.get_platform());
  ze_device_handle_t hdev =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(dev);

  // extension presence
  uint32_t next = 0;
  AFI_ZE_CHECK(zeDriverGetExtensionProperties(hdrv, &next, nullptr));
  std::vector<ze_driver_extension_properties_t> exts(next);
  AFI_ZE_CHECK(zeDriverGetExtensionProperties(hdrv, &next, exts.data()));
  bool has_rt = false, has_rtas = false;
  for (const auto &e : exts) {
    if (std::strncmp(e.name, ZE_RAYTRACING_EXT_NAME, ZE_MAX_EXTENSION_NAME) == 0) has_rt = true;
    if (std::strncmp(e.name, ZE_RTAS_EXT_NAME, ZE_MAX_EXTENSION_NAME) == 0) has_rtas = true;
  }
  std::printf("extensions: raytracing=%d rtas=%d (%u total)\n", has_rt,
              has_rtas, next);
  if (!has_rt || !has_rtas) return 1;

  // device RT properties
  ze_device_raytracing_ext_properties_t rt_props{};
  rt_props.stype = ZE_STRUCTURE_TYPE_DEVICE_RAYTRACING_EXT_PROPERTIES;
  ze_device_module_properties_t mod_props{};
  mod_props.stype = ZE_STRUCTURE_TYPE_DEVICE_MODULE_PROPERTIES;
  mod_props.pNext = &rt_props;
  AFI_ZE_CHECK(zeDeviceGetModuleProperties(hdev, &mod_props));
  std::printf("rt flags=0x%x\n", rt_props.flags);

  // scene
  Scene sc;
  sc.build(1.5f);
  RayGen gen;
  gen.sc = &sc;

  float *verts = sycl::malloc_shared<float>(sc.verts.size(), q);
  uint32_t *idx = sycl::malloc_shared<uint32_t>(sc.idx.size(), q);
  std::memcpy(verts, sc.verts.data(), sc.verts.size() * sizeof(float));
  std::memcpy(idx, sc.idx.data(), sc.idx.size() * sizeof(uint32_t));

  // --- build the BLAS --------------------------------------------------------
  auto fnCreate = ze_fn<FnBuilderCreate>("zeRTASBuilderCreateExp");
  auto fnDestroy = ze_fn<FnBuilderDestroy>("zeRTASBuilderDestroyExp");
  auto fnProps = ze_fn<FnBuilderProps>("zeRTASBuilderGetBuildPropertiesExp");
  auto fnBuild = ze_fn<FnBuilderBuild>("zeRTASBuilderBuildExp");
  auto fnParCreate = ze_fn<FnParCreate>("zeRTASParallelOperationCreateExp");
  auto fnParDestroy = ze_fn<FnParDestroy>("zeRTASParallelOperationDestroyExp");
  auto fnParJoin = ze_fn<FnParJoin>("zeRTASParallelOperationJoinExp");

  ze_rtas_builder_exp_desc_t bdesc{};
  bdesc.stype = ZE_STRUCTURE_TYPE_RTAS_BUILDER_EXP_DESC;
  bdesc.builderVersion = ZE_RTAS_BUILDER_EXP_VERSION_CURRENT;
  ze_rtas_builder_exp_handle_t hbuilder = nullptr;
  AFI_ZE_CHECK(fnCreate(hdrv, &bdesc, &hbuilder));

  ze_rtas_builder_triangles_geometry_info_exp_t geom{};
  geom.geometryType = ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_TRIANGLES;
  geom.geometryFlags = 0;
  geom.geometryMask = 0xFF;
  geom.triangleFormat = ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_TRIANGLE_INDICES_UINT32;
  geom.vertexFormat = ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_FLOAT3;
  geom.triangleCount = sc.tris();
  geom.vertexCount = sc.nverts();
  geom.triangleStride = 3 * sizeof(uint32_t);
  geom.vertexStride = 3 * sizeof(float);
  geom.pTriangleBuffer = idx;
  geom.pVertexBuffer = verts;

  const ze_rtas_builder_geometry_info_exp_t *geoms[] = {
      reinterpret_cast<const ze_rtas_builder_geometry_info_exp_t *>(&geom)};

  ze_rtas_parallel_operation_exp_handle_t hpar = nullptr;
  AFI_ZE_CHECK(fnParCreate(hdrv, &hpar));

  ze_rtas_device_exp_properties_t rtas_dev{};
  rtas_dev.stype = ZE_STRUCTURE_TYPE_RTAS_DEVICE_EXP_PROPERTIES;
  ze_device_properties_t dev_props{};
  dev_props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
  dev_props.pNext = &rtas_dev;
  AFI_ZE_CHECK(zeDeviceGetProperties(hdev, &dev_props));
  std::printf("rtas device: flags=0x%x rtasFormat(exp)=%d\n", rtas_dev.flags,
              (int)rtas_dev.rtasFormat);

  ze_rtas_builder_build_op_exp_desc_t op{};
  op.stype = ZE_STRUCTURE_TYPE_RTAS_BUILDER_BUILD_OP_EXP_DESC;
  op.rtasFormat = rtas_dev.rtasFormat;
  op.buildQuality = ZE_RTAS_BUILDER_BUILD_QUALITY_HINT_EXP_MEDIUM;
  op.buildFlags = ZE_RTAS_BUILDER_BUILD_OP_EXP_FLAG_NO_DUPLICATE_ANYHIT_INVOCATION;
  op.ppGeometries = geoms;
  op.numGeometries = 1;

  ze_rtas_builder_exp_properties_t sizes{};
  sizes.stype = ZE_STRUCTURE_TYPE_RTAS_BUILDER_EXP_PROPERTIES;
  AFI_ZE_CHECK(fnProps(hbuilder, &op, &sizes));
  std::printf("blas size: expected %zu bytes, max %zu, scratch %zu\n",
              sizes.rtasBufferSizeBytesExpected,
              sizes.rtasBufferSizeBytesMaxRequired,
              sizes.scratchBufferSizeBytes);

  void *scratch = sycl::malloc_shared<char>(sizes.scratchBufferSizeBytes, q);
  size_t rtas_bytes = sizes.rtasBufferSizeBytesExpected;
  void *rtas = sycl::malloc_shared<char>(rtas_bytes, q);

  const double t0 = now_ms();
  ze_result_t br = ZE_RESULT_ERROR_UNKNOWN;
  size_t rtas_used = 0;
  for (;;) {
    br = fnBuild(hbuilder, &op, scratch, sizes.scratchBufferSizeBytes, rtas,
                 rtas_bytes, hpar, nullptr, nullptr, &rtas_used);
    if (br == ZE_RESULT_EXP_RTAS_BUILD_RETRY) {
      rtas_bytes = rtas_used ? rtas_used
                             : (rtas_bytes * 3 / 2 + 127) & ~size_t(127);
      sycl::free(rtas, q);
      rtas = sycl::malloc_shared<char>(rtas_bytes, q);
      continue;
    }
    break;
  }
  if (br == ZE_RESULT_EXP_RTAS_BUILD_DEFERRED) {
    AFI_ZE_CHECK(fnParJoin(hpar));
    br = ZE_RESULT_SUCCESS;
  }
  AFI_ZE_CHECK(br);
  const double build_ms = now_ms() - t0;
  AFI_ZE_CHECK(fnParDestroy(hpar));
  AFI_ZE_CHECK(fnDestroy(hbuilder));
  std::printf("blas built: %zu bytes (used %zu) in %.2f ms\n", rtas_bytes,
              rtas_used, build_ms);

  auto accel_cell =
      sycl::malloc_shared<intel_raytracing_acceleration_structure_t>(1, q);
  void *accel_bits = rtas;  // memcpy sidesteps the address-space-qualified type
  std::memcpy(accel_cell, &accel_bits, sizeof(void *));

  // --- correctness sweep ------------------------------------------------------
  RayGen cgen;
  cgen.rng.seed(5678);
  cgen.sc = &sc;
  std::vector<float> rays6(size_t(n_rays_correct) * 6);
  std::vector<uint8_t> expect_hit(n_rays_correct);
  std::vector<float> expect_t(n_rays_correct);
  std::vector<uint32_t> expect_prim(n_rays_correct);
  for (uint32_t i = 0; i < n_rays_correct; ++i) {
    float o[3], d[3];
    bool eh;
    cgen.ray(o, d, &eh);
    std::memcpy(&rays6[size_t(i) * 6 + 0], o, 3 * sizeof(float));
    std::memcpy(&rays6[size_t(i) * 6 + 3], d, 3 * sizeof(float));
    expect_hit[i] = eh;
    float t;
    uint32_t p;
    expect_t[i] = host_nearest(sc, o, d, &t, &p) ? t : -1.0f;
    expect_prim[i] = p;
  }

  float *dev_rays = sycl::malloc_device<float>(rays6.size(), q);
  HitOut *dev_out = sycl::malloc_device<HitOut>(n_rays_correct, q);
  q.memcpy(dev_rays, rays6.data(), rays6.size() * sizeof(float)).wait();

  const uint32_t wg_c = 64;
  const uint32_t nw_c = ((n_rays_correct + wg_c - 1) / wg_c) * wg_c;
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class RtuProbeKernelCorrect>(
        sycl::nd_range<1>(sycl::range<1>(nw_c), sycl::range<1>(wg_c)),
        RtuProbeKernel{accel_cell, dev_rays, dev_out});
  }).wait();
  std::vector<HitOut> got(n_rays_correct);
  q.memcpy(got.data(), dev_out, n_rays_correct * sizeof(HitOut)).wait();

  uint32_t bad = 0, miss_disagree = 0;
  double max_terr = 0.0;
  for (uint32_t i = 0; i < n_rays_correct; ++i) {
    const bool gh = got[i].hit != 0;
    if (gh != bool(expect_hit[i])) {
      ++miss_disagree;
      ++bad;
      continue;
    }
    if (!gh) continue;
    const double e = std::abs(double(got[i].t) - double(expect_t[i]));
    const double rel = e / std::max(1e-4, double(expect_t[i]));
    max_terr = std::max(max_terr, rel);
    if (rel > 1e-3) ++bad;
  }
  std::printf("correctness: %u/%u rays, miss-disagreements %u, max rel |dt| %.3g -> %s\n",
              n_rays_correct - bad, n_rays_correct, miss_disagree, max_terr,
              bad ? "FAIL" : "PASS");

  // --- timing sweep ------------------------------------------------------------
  std::vector<float> trays6(size_t(n_rays_time) * 6);
  RayGen tgen;
  tgen.rng.seed(9999);
  tgen.sc = &sc;
  bool dummy;
  for (uint32_t i = 0; i < n_rays_time; ++i) {
    float o[3], d[3];
    tgen.ray(o, d, &dummy);
    std::memcpy(&trays6[size_t(i) * 6 + 0], o, 3 * sizeof(float));
    std::memcpy(&trays6[size_t(i) * 6 + 3], d, 3 * sizeof(float));
  }
  float *tdev_rays = sycl::malloc_device<float>(trays6.size(), q);
  HitOut *tdev_out = sycl::malloc_device<HitOut>(n_rays_time, q);
  q.memcpy(tdev_rays, trays6.data(), trays6.size() * sizeof(float)).wait();

  const uint32_t wg = 64;
  const uint32_t nw = ((n_rays_time + wg - 1) / wg) * wg;
  auto trace_once = [&]() {
    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<class RtuProbeKernelNd>(
          sycl::nd_range<1>(sycl::range<1>(nw), sycl::range<1>(wg)),
          RtuProbeKernel{accel_cell, tdev_rays, tdev_out});
    });
  };
  trace_once();
  q.wait();
  const double t1 = now_ms();
  constexpr int kIters = 5;
  for (int it = 0; it < kIters; ++it) {
    trace_once();
    q.wait();
  }
  const double trace_ms = (now_ms() - t1) / kIters;
  const double mrays = double(nw) / (trace_ms * 1e-3) / 1e6;
  std::printf("trace: %u rays in %.3f ms -> %.1f Mray/s (nd_range wg=%u, sync)\n",
              nw, trace_ms, mrays, wg);

  // pipelined submits (no per-iter wait): measures submit-side cost too
  const double t2 = now_ms();
  for (int it = 0; it < kIters; ++it) {
    trace_once();
  }
  q.wait();
  const double trace_nd_ms = (now_ms() - t2) / kIters;
  const double mrays_nd = double(nw) / (trace_nd_ms * 1e-3) / 1e6;
  std::printf("trace: %u rays in %.3f ms -> %.1f Mray/s (nd_range wg=%u, pipelined)\n",
              nw, trace_nd_ms, mrays_nd, wg);

  // host scalar baseline (single thread, same ray set)
  {
    const double h0 = now_ms();
    uint32_t hits = 0;
    const uint32_t hrays = std::min<uint32_t>(n_rays_time, 100000);
    for (uint32_t i = 0; i < hrays; ++i) {
      const float *r = &trays6[size_t(i) * 6];
      float t;
      uint32_t p;
      hits += host_nearest(sc, r, r + 3, &t, &p);
    }
    const double hms = now_ms() - h0;
    std::printf("host: %u rays in %.1f ms -> %.2f Mray/s (1 thread MT x %u tris, %u hits)\n",
                hrays, hms, double(hrays) / (hms * 1e-3) / 1e6, sc.tris(), hits);
  }

  sycl::free(verts, q);
  sycl::free(idx, q);
  sycl::free(scratch, q);
  sycl::free(rtas, q);
  sycl::free(accel_cell, q);
  sycl::free(dev_rays, q);
  sycl::free(dev_out, q);
  sycl::free(tdev_rays, q);
  sycl::free(tdev_out, q);
  return bad ? 1 : 0;
}
