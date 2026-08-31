// afi-rtu-terrain: M2 site_survey race — B70 RTU vs CPU Embree on identical
// heightfield terrain, real query shapes (TX->RX LOS profiles + azimuth
// horizon fans). docs/hoa64-rtu-backend.md, Experiment-2R.
//
// Geometry: G x G heightfield cells -> 2*G*G triangles, IDENTICAL index
// buffer in both arms (L0 quad split v0v1v3 / v2v3v1 applied up front, so
// both engines trace byte-identical geometry).
//
// Usage: afi-rtu-terrain [gpu=1] [G=256] [n_prof=8192] [n_fan=360] [--dump]

#include <sycl/sycl.hpp>  // MUST precede embree4: RTC_SYCL_API expands to
                          // SYCL_EXTERNAL on the device pass
#include <embree4/rtcore.h>
#include <level_zero/ze_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <random>
#include <thread>
#include <vector>

#include "afi/rttrace.h"

namespace {

#define AFI_ZE_CHECK(expr)                                                    \
  do {                                                                        \
    ze_result_t r_ = (expr);                                                  \
    if (r_ != ZE_RESULT_SUCCESS) {                                            \
      std::fprintf(stderr, "ze error 0x%x at %s:%d\n", r_, __LINE__);         \
      std::exit(1);                                                           \
    }                                                                         \
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

double now_ms() {
  using clk = std::chrono::steady_clock;
  return std::chrono::duration<double, std::milli>(
             clk::now().time_since_epoch())
      .count();
}

// --- deterministic fbm heightfield ------------------------------------------

inline float hash01(uint32_t x, uint32_t y, uint32_t seed) {
  uint32_t h = x * 374761393u + y * 668265263u + seed * 1442695041u;
  h = (h ^ (h >> 13)) * 1274126177u;
  return ((h ^ (h >> 16)) & 0xFFFFFF) / float(0xFFFFFF);
}

inline float vnoise(float x, float y, uint32_t seed) {
  const float xi = std::floor(x), yi = std::floor(y);
  const float xf = x - xi, yf = y - yi;
  const float u = xf * xf * (3 - 2 * xf), v = yf * yf * (3 - 2 * yf);
  const uint32_t ix = uint32_t(xi), iy = uint32_t(yi);
  const float a = hash01(ix, iy, seed), b = hash01(ix + 1, iy, seed);
  const float c = hash01(ix, iy + 1, seed), d = hash01(ix + 1, iy + 1, seed);
  return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
}

struct Terrain {
  uint32_t G;          // cells per side
  float span = 4096.f; // meters
  float hmax = 60.f;   // max relief
  std::vector<float> h;  // (G+1)^2

  explicit Terrain(uint32_t G_) : G(G_), h(size_t(G + 1) * (G + 1)) {
    for (uint32_t j = 0; j <= G; ++j)
      for (uint32_t i = 0; i <= G; ++i) {
        const float s = float(G);
        float x = float(i) / s * 24.f, y = float(j) / s * 24.f;
        float amp = 1.f, sum = 0.f, norm = 0.f;
        for (int o = 0; o < 5; ++o) {
          sum += amp * vnoise(x, y, 7u + o * 131u);
          norm += amp;
          amp *= 0.5f;
          x *= 2.03f;
          y *= 1.97f;
        }
        h[size_t(j) * (G + 1) + i] = hmax * (sum / norm);
      }
    // ridge for interesting occlusion
    for (uint32_t j = 0; j <= G; ++j)
      for (uint32_t i = 0; i <= G; ++i) {
        const float dx = float(i) - 0.62f * float(G);
        const float dy = float(j) - 0.38f * float(G);
        const float r2 = (dx * dx + dy * dy) / (0.09f * float(G) * float(G));
        h[size_t(j) * (G + 1) + i] += hmax * 1.6f * std::exp(-r2);
      }
  }

  float at(uint32_t i, uint32_t j) const {
    return h[size_t(j) * (G + 1) + i];
  }
  // bilinear ground height at world (x, z)
  float ground(float x, float z) const {
    const float s = float(G) / span;
    float fi = std::min(std::max(x * s, 0.f), float(G) - 1e-4f);
    float fj = std::min(std::max(z * s, 0.f), float(G) - 1e-4f);
    const uint32_t i = uint32_t(fi), j = uint32_t(fj);
    const float u = fi - i, v = fj - j;
    const float h00 = at(i, j), h10 = at(i + 1, j);
    const float h01 = at(i, j + 1), h11 = at(i + 1, j + 1);
    return h00 * (1 - u) * (1 - v) + h10 * u * (1 - v) +
           h01 * (1 - u) * v + h11 * u * v;
  }
};

// --- scene assembly (identical triangles for both arms) ----------------------

struct TriScene {
  uint32_t G;
  std::vector<float> verts;         // 3*(G+1)^2
  std::vector<uint32_t> idx;        // 6*G*G  (L0 quad split, applied)
  std::vector<uint16_t> pad_verts;  // Embree arm: 16-byte stride float3
  size_t ntri() const { return size_t(2) * G * G; }

  explicit TriScene(const Terrain &t) : G(t.G) {
    const uint32_t G1 = G + 1;
    verts.resize(size_t(G1) * G1 * 3);
    for (uint32_t j = 0; j <= G; ++j)
      for (uint32_t i = 0; i <= G; ++i) {
        float *v = &verts[(size_t(j) * G1 + i) * 3];
        v[0] = float(i) / float(G) * t.span;
        v[1] = t.at(i, j);
        v[2] = float(j) / float(G) * t.span;
      }
    idx.reserve(ntri() * 3);
    for (uint32_t j = 0; j < G; ++j)
      for (uint32_t i = 0; i < G; ++i) {
        const uint32_t v00 = j * G1 + i, v10 = v00 + 1;
        const uint32_t v01 = v00 + G1, v11 = v01 + 1;
        // L0 quad convention: tri1 (v0,v1,v3) = (v00,v10,v01),
        //                    tri2 (v2,v3,v1) = (v11,v01,v10)
        idx.push_back(v00); idx.push_back(v10); idx.push_back(v01);
        idx.push_back(v11); idx.push_back(v01); idx.push_back(v10);
      }
    // padded vertex copy for Embree (16-byte stride)
    pad_verts.resize(size_t(G1) * G1 * 4);
    for (size_t k = 0; k < size_t(G1) * G1; ++k) {
      pad_verts[k * 4 + 0] = verts[k * 3 + 0];
      pad_verts[k * 4 + 1] = verts[k * 3 + 1];
      pad_verts[k * 4 + 2] = verts[k * 3 + 2];
      pad_verts[k * 4 + 3] = 0;
    }
  }
};

// --- rays ---------------------------------------------------------------------

struct Rays {
  std::vector<float> data;  // 6 per ray: origin xyz + unit dir xyz
  std::vector<float> tmax;  // per-ray segment length (1e30 for unbounded)
  size_t n() const { return data.size() / 6; }
  float *org(size_t i) { return &data[i * 6]; }
  float *dir(size_t i) { return &data[i * 6 + 3]; }
};

// LOS profile rays: TX/RX at random grid points, m above local ground.
void gen_profiles(const Terrain &t, size_t n, uint32_t seed, Rays &out) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> u01(0.f, 1.f);
  out.data.resize(n * 6);
  out.tmax.assign(n, 0.f);
  for (size_t i = 0; i < n; ++i) {
    const float tx = u01(rng) * t.span, tz = u01(rng) * t.span;
    const float rx = u01(rng) * t.span, rz = u01(rng) * t.span;
    const float th = t.ground(tx, tz) + 5.f + 35.f * u01(rng);
    const float rh = t.ground(rx, rz) + 2.f + 15.f * u01(rng);
    float d[3] = {rx - tx, rh - th, rz - tz};
    const float len = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    float *o = out.org(i);
    o[0] = tx; o[1] = th; o[2] = tz;
    float *dd = out.dir(i);
    dd[0] = d[0] / len; dd[1] = d[1] / len; dd[2] = d[2] / len;
    out.tmax[i] = len;
  }
}

// Azimuth horizon fan: one TX above center ground, rays slightly downward.
void gen_fan(const Terrain &t, size_t n, uint32_t seed, Rays &out) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> jitter(0.f, 0.15f);
  out.data.resize(n * 6);
  out.tmax.assign(n, 1e30f);
  const float cx = t.span * 0.5f, cz = t.span * 0.5f;
  const float h0 = t.ground(cx, cz) + 25.f;
  for (size_t i = 0; i < n; ++i) {
    const float a = float(i) / float(n) * 2.f * float(M_PI);
    const float elev = -0.035f - jitter(rng) * 0.1f;  // -2 .. -11.6 deg
    float *o = out.org(i);
    o[0] = cx; o[1] = h0; o[2] = cz;
    float *d = out.dir(i);
    d[0] = std::cos(a);
    d[1] = elev;
    d[2] = std::sin(a);
    const float len = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    d[0] /= len; d[1] /= len; d[2] /= len;
  }
}

struct HitOut {
  float t;
  uint32_t prim;
  uint32_t hit;
};

// --- RTU kernels ---------------------------------------------------------------

struct RtuOcclKernel {  // profile rays: boolean LOS, early-out traversal
  const intel_raytracing_acceleration_structure_t *accel;
  const float *rays;
  const float *tmax;
  HitOut *out;
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
    HitOut h{-1.f, 0xFFFFFFFFu, 0u};
    if (intel_is_traversal_done(q) && intel_has_committed_hit(q)) {
      h.t = intel_get_hit_distance(q, intel_hit_type_committed_hit);
      h.hit = 1u;
    }
    intel_ray_query_abandon(q);
    out[li] = h;
#endif
  }
};

struct RtuHitKernel {  // fan rays: nearest-hit distance (horizon profile)
  const intel_raytracing_acceleration_structure_t *accel;
  const float *rays;
  const float *tmax;
  HitOut *out;
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
    HitOut h{-1.f, 0xFFFFFFFFu, 0u};
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

// --- RTU arm --------------------------------------------------------------------

struct RtuArm {
  sycl::queue q;
  ze_rtas_builder_exp_handle_t hbuilder = nullptr;
  FnBuilderDestroy fnDestroy{};
  FnParDestroy fnParDestroy{};
  void *scratch = nullptr;
  size_t scratch_bytes = 0;
  void *rtas = nullptr;
  size_t rtas_bytes = 0;
  intel_raytracing_acceleration_structure_t *accel_cell = nullptr;
  double build_ms = 0;

  RtuArm(sycl::queue q_, ze_device_handle_t hdev, ze_driver_handle_t hdrv,
         const TriScene &sc, void *verts, void *idx)
      : q(q_) {
    auto fnCreate = ze_fn<FnBuilderCreate>("zeRTASBuilderCreateExp");
    fnDestroy = ze_fn<FnBuilderDestroy>("zeRTASBuilderDestroyExp");
    auto fnProps = ze_fn<FnBuilderProps>("zeRTASBuilderGetBuildPropertiesExp");
    auto fnBuild = ze_fn<FnBuilderBuild>("zeRTASBuilderBuildExp");
    auto fnParCreate = ze_fn<FnParCreate>("zeRTASParallelOperationCreateExp");
    fnParDestroy = ze_fn<FnParDestroy>("zeRTASParallelOperationDestroyExp");
    auto fnParJoin = ze_fn<FnParJoin>("zeRTASParallelOperationJoinExp");

    ze_rtas_device_exp_properties_t rtas_dev{};
    rtas_dev.stype = ZE_STRUCTURE_TYPE_RTAS_DEVICE_EXP_PROPERTIES;
    ze_device_properties_t dev_props{};
    dev_props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    dev_props.pNext = &rtas_dev;
    AFI_ZE_CHECK(zeDeviceGetProperties(hdev, &dev_props));

    ze_rtas_builder_exp_desc_t bdesc{};
    bdesc.stype = ZE_STRUCTURE_TYPE_RTAS_BUILDER_EXP_DESC;
    bdesc.builderVersion = ZE_RTAS_BUILDER_EXP_VERSION_CURRENT;
    AFI_ZE_CHECK(fnCreate(hdrv, &bdesc, &hbuilder));

    ze_rtas_builder_triangles_geometry_info_exp_t geom{};
    geom.geometryType = ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_TRIANGLES;
    geom.geometryMask = 0xFF;
    geom.triangleFormat = ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_TRIANGLE_INDICES_UINT32;
    geom.vertexFormat = ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_FLOAT3;
    geom.triangleCount = uint32_t(sc.ntri());
    geom.vertexCount = uint32_t(sc.verts.size() / 3);
    geom.triangleStride = 3 * sizeof(uint32_t);
    geom.vertexStride = 3 * sizeof(float);
    geom.pTriangleBuffer = idx;
    geom.pVertexBuffer = verts;

    const ze_rtas_builder_geometry_info_exp_t *geoms[] = {
        reinterpret_cast<const ze_rtas_builder_geometry_info_exp_t *>(&geom)};
    ze_rtas_parallel_operation_exp_handle_t hpar = nullptr;
    AFI_ZE_CHECK(fnParCreate(hdrv, &hpar));

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
    scratch_bytes = sizes.scratchBufferSizeBytes;
    scratch = sycl::malloc_shared<char>(scratch_bytes, q);
    rtas_bytes = sizes.rtasBufferSizeBytesExpected;

    const double t0 = now_ms();
    for (;;) {
      rtas = sycl::malloc_shared<char>(rtas_bytes, q);
      ze_result_t br = fnBuild(hbuilder, &op, scratch, scratch_bytes, rtas,
                               rtas_bytes, hpar, nullptr, nullptr, nullptr);
      if (br == ZE_RESULT_EXP_RTAS_BUILD_RETRY) {
        rtas_bytes = rtas_bytes * 3 / 2 + 127 & ~size_t(127);
        sycl::free(rtas, q);
        continue;
      }
      if (br == ZE_RESULT_EXP_RTAS_BUILD_DEFERRED) AFI_ZE_CHECK(fnParJoin(hpar));
      else AFI_ZE_CHECK(br);
      break;
    }
    build_ms = now_ms() - t0;
    AFI_ZE_CHECK(fnParDestroy(hpar));
    AFI_ZE_CHECK(fnDestroy(hbuilder));

    accel_cell = sycl::malloc_shared<intel_raytracing_acceleration_structure_t>(1, q);
    void *bits = rtas;
    std::memcpy(accel_cell, &bits, sizeof(void *));
  }

  ~RtuArm() {
    if (scratch) sycl::free(scratch, q);
    if (rtas) sycl::free(rtas, q);
    if (accel_cell) sycl::free(accel_cell, q);
  }

  void trace(const float *dev_rays, const float *dev_tmax, size_t n,
             HitOut *dev_out, bool occl) {
    const uint32_t wg = 64;
    const size_t nw = (n + wg - 1) / wg * wg;
    q.submit([&](sycl::handler &cgh) {
      if (occl)
        cgh.parallel_for<class RtuTerrainOccl>(
            sycl::nd_range<1>(sycl::range<1>(nw), sycl::range<1>(wg)),
            RtuOcclKernel{accel_cell, dev_rays, dev_tmax, dev_out});
      else
        cgh.parallel_for<class RtuTerrainHit>(
            sycl::nd_range<1>(sycl::range<1>(nw), sycl::range<1>(wg)),
            RtuHitKernel{accel_cell, dev_rays, dev_tmax, dev_out});
    }).wait();
  }
};

// --- Embree arm -------------------------------------------------------------------

void embree_error(void *, RTCError code, const char *str) {
  std::fprintf(stderr, "embree error %d: %s\n", int(code), str);
}

struct EmbreeArm {
  RTCDevice dev = nullptr;
  RTCScene scene = nullptr;
  double build_ms = 0;

  EmbreeArm(const TriScene &sc) {
    dev = rtcNewDevice(nullptr);
    rtcSetDeviceErrorFunction(dev, embree_error, nullptr);
    scene = rtcNewScene(dev);
    RTCGeometry geom = rtcNewGeometry(dev, RTC_GEOMETRY_TYPE_TRIANGLE);
    rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0,
                            RTC_FORMAT_FLOAT3, 3 * sizeof(float),
                            sc.verts.size() / 3);
    rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
                            3 * sizeof(uint32_t), sc.ntri());
    // host-pass copy into the Embree-owned buffers
#ifndef __SYCL_DEVICE_ONLY__
    {
      float *vb = (float *)rtcGetGeometryBufferData(geom, RTC_BUFFER_TYPE_VERTEX, 0);
      uint32_t *ib = (uint32_t *)rtcGetGeometryBufferData(geom, RTC_BUFFER_TYPE_INDEX, 0);
      std::memcpy(vb, sc.verts.data(), sc.verts.size() * sizeof(float));
      std::memcpy(ib, sc.idx.data(), sc.idx.size() * sizeof(uint32_t));
    }
#endif
    std::printf("embree buffers: verts=%zu tris=%zu\n", sc.verts.size() / 3,
                sc.ntri());
    rtcCommitGeometry(geom);
    rtcAttachGeometry(scene, geom);
    rtcReleaseGeometry(geom);
    const double t0 = now_ms();
    rtcCommitScene(scene);
    build_ms = now_ms() - t0;

#ifndef __SYCL_DEVICE_ONLY__
    // sanity: straight-down ray at terrain center must hit
    RTCRayHit rh{};
    rh.ray.org_x = sc.verts[(size_t(sc.G / 2) * (sc.G + 1) + sc.G / 2) * 3 + 0];
    rh.ray.org_y = 500.f;
    rh.ray.org_z = sc.verts[(size_t(sc.G / 2) * (sc.G + 1) + sc.G / 2) * 3 + 2];
    rh.ray.dir_x = 0.f; rh.ray.dir_y = -1.f; rh.ray.dir_z = 0.f;
    rh.ray.tnear = 1e-3f;
    rh.ray.tfar = 1e30f;
    rh.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    RTCIntersectArguments iargs;
    rtcInitIntersectArguments(&iargs);
    rtcIntersect1(scene, &rh, &iargs);
    std::printf("embree sanity: center ray %s at t=%.2f\n",
                rh.hit.geomID != RTC_INVALID_GEOMETRY_ID ? "HIT" : "MISS",
                rh.hit.geomID != RTC_INVALID_GEOMETRY_ID ? rh.ray.tfar : -1.f);
#endif
  }
  ~EmbreeArm() {
    if (scene) rtcReleaseScene(scene);
    if (dev) rtcReleaseDevice(dev);
  }

  static void trace_chunk(RTCScene scene, const float *rays,
                          const float *tmax, size_t lo, size_t hi,
                          HitOut *out, bool occl) {
#ifndef __SYCL_DEVICE_ONLY__  // rtcIntersect1/rtcOccluded1 are host-only in
                              // Embree 4.4 (device uses Traversable variants)
    for (size_t i = lo; i < hi; ++i) {
      const float *r = rays + i * 6;
      HitOut h{-1.f, 0xFFFFFFFFu, 0u};
      if (occl) {
        RTCRay ray{};
        ray.org_x = r[0]; ray.org_y = r[1]; ray.org_z = r[2];
        ray.dir_x = r[3]; ray.dir_y = r[4]; ray.dir_z = r[5];
        ray.tnear = 1e-3f;
        ray.tfar = tmax[i];
        ray.mask = 0xFF;
        RTCOccludedArguments oargs;
        rtcInitOccludedArguments(&oargs);
        rtcOccluded1(scene, &ray, &oargs);
        h.hit = ray.tfar < 0 ? 1u : 0u;
      } else {
        RTCRayHit rh{};
        rh.ray.org_x = r[0]; rh.ray.org_y = r[1]; rh.ray.org_z = r[2];
        rh.ray.dir_x = r[3]; rh.ray.dir_y = r[4]; rh.ray.dir_z = r[5];
        rh.ray.tnear = 1e-3f;
        rh.ray.tfar = tmax[i];
        rh.ray.mask = 0xFF;
        rh.hit.geomID = RTC_INVALID_GEOMETRY_ID;
        RTCIntersectArguments iargs;
        rtcInitIntersectArguments(&iargs);
        rtcIntersect1(scene, &rh, &iargs);
        if (rh.hit.geomID != RTC_INVALID_GEOMETRY_ID)
          h = HitOut{rh.ray.tfar, uint32_t(rh.hit.primID), 1u};
      }
      out[i] = h;
    }
#endif
  }

  void trace(const float *rays, const float *tmax, size_t n, HitOut *out,
             bool occl, unsigned threads) {
    const size_t chunk = (n + threads - 1) / threads;
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < threads && t * chunk < n; ++t) {
      const size_t lo = t * chunk, hi = std::min(n, lo + chunk);
      pool.emplace_back(trace_chunk, scene, rays, tmax, lo, hi, out, occl);
    }
    for (auto &th : pool) th.join();
  }
};

// --- compare -----------------------------------------------------------------------

struct Compare {
  size_t bool_disagree = 0;
  size_t common_hits = 0;
  double max_rel_dt = 0;
  size_t checked = 0;
};

Compare compare(const std::vector<HitOut> &a, const std::vector<HitOut> &b,
                bool distance_matters) {
  Compare c;
  c.checked = a.size();
  for (size_t i = 0; i < a.size(); ++i) {
    const bool ha = a[i].hit != 0, hb = b[i].hit != 0;
    if (ha != hb) { ++c.bool_disagree; continue; }
    if (!ha) continue;
    ++c.common_hits;
    if (distance_matters) {
      const double e = std::abs(double(a[i].t) - double(b[i].t));
      c.max_rel_dt = std::max(c.max_rel_dt,
                              e / std::max(1.0, double(b[i].t)));
    }
  }
  return c;
}

}  // namespace

int main(int argc, char **argv) {
  const int gpu_idx = argc > 1 ? std::atoi(argv[1]) : 1;
  const uint32_t G = argc > 2 ? std::atoi(argv[2]) : 256;
  const size_t n_prof = argc > 3 ? std::atol(argv[3]) : 8192;
  const size_t n_fan = argc > 4 ? std::atol(argv[4]) : 360;
  const bool dump = argc > 5 && std::strcmp(argv[5], "--dump") == 0;
  const unsigned threads = std::max(1u, std::thread::hardware_concurrency());

  AFI_ZE_CHECK(zeInit(0));
  std::vector<sycl::device> gpus;
  for (const auto &p : sycl::platform::get_platforms())
    if (p.get_backend() == sycl::backend::ext_oneapi_level_zero)
      for (const auto &d : p.get_devices())
        if (d.is_gpu()) gpus.push_back(d);
  if (gpus.empty() || gpu_idx < 0 || size_t(gpu_idx) >= gpus.size()) {
    std::fprintf(stderr, "no L0 gpu at index %d\n", gpu_idx);
    return 1;
  }
  sycl::queue q{gpus[size_t(gpu_idx)]};
  ze_driver_handle_t hdrv = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
      gpus[size_t(gpu_idx)].get_platform());
  ze_device_handle_t hdev =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(gpus[size_t(gpu_idx)]);
  {
    auto *sink = sycl::malloc_device<int>(1, q);
    q.single_task([=]() { *sink = 42; }).wait();
    sycl::free(sink, q);
  }
  std::printf("== afi-rtu-terrain G=%u (%zu tris) prof=%zu fan=%zu threads=%u\n",
              G, size_t(2) * G * G, n_prof, n_fan, threads);

  Terrain terrain(G);
  TriScene sc(terrain);

  // device geometry (shared USM: builder reads it, host dumps it)
  float *verts = sycl::malloc_shared<float>(sc.verts.size(), q);
  uint32_t *idx = sycl::malloc_shared<uint32_t>(sc.idx.size(), q);
  std::memcpy(verts, sc.verts.data(), sc.verts.size() * sizeof(float));
  std::memcpy(idx, sc.idx.data(), sc.idx.size() * sizeof(uint32_t));

  RtuArm rtu(q, hdev, hdrv, sc, verts, idx);
  std::printf("build: rtu %.1f ms (%zu bytes)\n", rtu.build_ms, rtu.rtas_bytes);

  EmbreeArm embree(sc);
  std::printf("build: embree %.1f ms\n", embree.build_ms);

  Rays prof, fan;
  gen_profiles(terrain, n_prof, 11, prof);
  gen_fan(terrain, n_fan, 22, fan);

  // device ray buffers + outputs
  auto up_rays = [&](const Rays &r) {
    float *d = sycl::malloc_device<float>(r.data.size(), q);
    q.memcpy(d, r.data.data(), r.data.size() * sizeof(float)).wait();
    return d;
  };
  float *prof_rays = up_rays(prof);
  float *fan_rays = up_rays(fan);
  float *prof_tmax = sycl::malloc_device<float>(n_prof, q);
  float *fan_tmax = sycl::malloc_device<float>(n_fan, q);
  q.memcpy(prof_tmax, prof.tmax.data(), n_prof * sizeof(float)).wait();
  q.memcpy(fan_tmax, fan.tmax.data(), n_fan * sizeof(float)).wait();
  HitOut *rtu_prof_out = sycl::malloc_device<HitOut>(n_prof, q);
  HitOut *rtu_fan_out = sycl::malloc_device<HitOut>(n_fan, q);

  std::vector<HitOut> rtu_prof(n_prof), rtu_fan(n_fan), emb_prof(n_prof),
      emb_fan(n_fan);

  // correctness first (small counts), then timing
  rtu.trace(prof_rays, prof_tmax, n_prof, rtu_prof_out, /*occl=*/true);
  q.memcpy(rtu_prof.data(), rtu_prof_out, n_prof * sizeof(HitOut)).wait();
  embree.trace(prof.data.data(), prof.tmax.data(), n_prof, emb_prof.data(),
               true, threads);
  Compare cp = compare(rtu_prof, emb_prof, false);

  rtu.trace(fan_rays, fan_tmax, n_fan, rtu_fan_out, false);
  q.memcpy(rtu_fan.data(), rtu_fan_out, n_fan * sizeof(HitOut)).wait();
  embree.trace(fan.data.data(), fan.tmax.data(), n_fan, emb_fan.data(),
               false, threads);
  Compare cf = compare(rtu_fan, emb_fan, true);

  std::printf("cross-check profile (occl): %zu/%zu bool disagreements, common hits %zu\n",
              cp.bool_disagree, cp.checked, cp.common_hits);
  std::printf("cross-check fan (closest):  %zu/%zu bool disagreements, max rel |dt| %.3g\n",
              cf.bool_disagree, cf.checked, cf.max_rel_dt);

  // timing: 5 iterations each
  auto bench = [&](const char *name, auto &&fn) {
    fn();  // warm
    const double t0 = now_ms();
    constexpr int kIters = 5;
    for (int i = 0; i < kIters; ++i) fn();
    const double ms = (now_ms() - t0) / kIters;
    return ms;
  };

  const double rtu_prof_ms = bench("rtu prof", [&] {
    rtu.trace(prof_rays, prof_tmax, n_prof, rtu_prof_out, true);
  });
  const double emb_prof_ms = bench("emb prof", [&] {
    embree.trace(prof.data.data(), prof.tmax.data(), n_prof, emb_prof.data(),
               true, threads);
  });
  const double rtu_fan_ms = bench("rtu fan", [&] {
    rtu.trace(fan_rays, fan_tmax, n_fan, rtu_fan_out, false);
  });
  const double emb_fan_ms = bench("emb fan", [&] {
    embree.trace(fan.data.data(), fan.tmax.data(), n_fan, emb_fan.data(),
               false, threads);
  });

  std::printf("profile LOS:  rtu %.3f ms (%.1f Mray/s)  embree %.3f ms (%.1f Mray/s)  speedup %.1fx\n",
              rtu_prof_ms, double(n_prof) / rtu_prof_ms / 1e3,
              emb_prof_ms, double(n_prof) / emb_prof_ms / 1e3,
              emb_prof_ms / rtu_prof_ms);
  std::printf("horizon fan:  rtu %.3f ms (%.1f Mray/s)  embree %.3f ms (%.1f Mray/s)  speedup %.1fx\n",
              rtu_fan_ms, double(n_fan) / rtu_fan_ms / 1e3,
              emb_fan_ms, double(n_fan) / emb_fan_ms / 1e3,
              emb_fan_ms / rtu_fan_ms);

  if (dump) {
    char path[256];
    std::snprintf(path, sizeof path, "/tmp/opencode/rtu_terrain_G%u.bin", G);
    FILE *f = std::fopen(path, "wb");
    if (f) {
      const uint32_t G1 = G + 1;
      std::fwrite(&G, 4, 1, f);
      std::fwrite(&terrain.span, 4, 1, f);
      std::fwrite(terrain.h.data(), 4, G1 * G1, f);
      std::fwrite(prof.data.data(), 4, prof.data.size(), f);
      std::fwrite(prof.tmax.data(), 4, n_prof, f);
      std::fwrite(rtu_prof.data(), sizeof(HitOut), n_prof, f);
      std::fwrite(fan.data.data(), 4, fan.data.size(), f);
      std::fwrite(fan.tmax.data(), 4, n_fan, f);
      std::fwrite(rtu_fan.data(), sizeof(HitOut), n_fan, f);
      std::fclose(f);
      std::printf("dumped %s\n", path);
    }
  }

  sycl::free(verts, q);
  sycl::free(idx, q);
  sycl::free(prof_rays, q);
  sycl::free(fan_rays, q);
  sycl::free(prof_tmax, q);
  sycl::free(fan_tmax, q);
  sycl::free(rtu_prof_out, q);
  sycl::free(rtu_fan_out, q);
  return (cp.bool_disagree + cf.bool_disagree) > 0 ? 1 : 0;
}
