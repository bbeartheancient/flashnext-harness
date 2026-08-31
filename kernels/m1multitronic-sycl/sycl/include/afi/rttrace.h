// Device-side RT query ABI for B70 RTUs via IGC name-lowered builtins.
// Derived from RenderKit/Embree kernels/sycl/rthwif_rttrace.h
// (Apache-2.0, (c) Intel Corporation). The struct layouts and symbol
// names are the contract: IGC's RTBuilder matches these exact
// `intel_ray_query_*` / `intel_get_*` extern "C" names in SPIR-V and
// lowers them to hardware traversal messages. No SYCL ray-query
// extension header is required or involved.

#pragma once

#include <cstdint>

enum intel_ray_flags_t {
  intel_ray_flags_none = 0x00,
  intel_ray_flags_force_opaque = 0x01,
  intel_ray_flags_force_non_opaque = 0x02,
  intel_ray_flags_accept_first_hit_and_end_search = 0x04,
  intel_ray_flags_skip_closest_hit_shader = 0x08,
  intel_ray_flags_cull_back_facing_triangles = 0x10,
  intel_ray_flags_cull_front_facing_triangles = 0x20,
  intel_ray_flags_cull_opaque = 0x40,
  intel_ray_flags_cull_non_opaque = 0x80,
  intel_ray_flags_skip_triangles = 0x100,
  intel_ray_flags_skip_procedural_primitives = 0x200,
};

enum intel_hit_type_t {
  intel_hit_type_committed_hit = 0,
  intel_hit_type_potential_hit = 1,
};

typedef __attribute__((opencl_private)) struct intel_ray_query_opaque_t *intel_ray_query_t;
typedef __attribute__((opencl_global))
    struct intel_raytracing_acceleration_structure_opaque_t *intel_raytracing_acceleration_structure_t;

struct intel_float2 {
  float x, y;
  intel_float2() {}
  intel_float2(float x_, float y_) : x(x_), y(y_) {}
};

struct intel_float3 {
  float x, y, z;
  intel_float3() {}
  intel_float3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};

struct intel_ray_desc_t {
  intel_float3 origin;
  intel_float3 direction;
  float tmin;
  float tmax;
  unsigned int mask;
  intel_ray_flags_t flags;
};

#ifndef __SYCL_DEVICE_ONLY__
// Device-only ABI: the builtins below are undefined on the host by design
// (IGC resolves them during SPIR-V lowering). Host code that calls them
// fails at link time, which is the correct signal.
#endif

#ifdef __SYCL_DEVICE_ONLY__

#ifndef SYCL_EXTERNAL
#define SYCL_EXTERNAL __attribute__((sycl_external))
#endif

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"

// initializes a ray query
SYCL_EXTERNAL extern "C" intel_ray_query_t
intel_ray_query_init(intel_ray_desc_t ray,
                     intel_raytracing_acceleration_structure_t accel);

// start traversal of a ray query
SYCL_EXTERNAL extern "C" void intel_ray_query_start_traversal(intel_ray_query_t query);

// synchronize ray query execution; must precede the accessors below
SYCL_EXTERNAL extern "C" void intel_ray_query_sync(intel_ray_query_t query);

// signal the query will not be used further (implicit sync)
SYCL_EXTERNAL extern "C" void intel_ray_query_abandon(intel_ray_query_t query);

// traversal terminated without needing shader processing
SYCL_EXTERNAL extern "C" bool intel_is_traversal_done(intel_ray_query_t query);

// a committed hit exists (after traversal done)
SYCL_EXTERNAL extern "C" bool intel_has_committed_hit(intel_ray_query_t query);

SYCL_EXTERNAL extern "C" float intel_get_hit_distance(intel_ray_query_t query,
                                                      intel_hit_type_t hit_type);
SYCL_EXTERNAL extern "C" intel_float2 intel_get_hit_barycentrics(intel_ray_query_t query,
                                                                 intel_hit_type_t hit_type);
SYCL_EXTERNAL extern "C" unsigned int intel_get_hit_geometry_id(intel_ray_query_t query,
                                                                intel_hit_type_t hit_type);
SYCL_EXTERNAL extern "C" unsigned int intel_get_hit_primitive_id(intel_ray_query_t query,
                                                                 intel_hit_type_t hit_type);

#pragma clang diagnostic pop

#endif  // __SYCL_DEVICE_ONLY__
