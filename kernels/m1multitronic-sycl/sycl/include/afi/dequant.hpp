#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>

namespace afi {

constexpr int kAtom = 8;
constexpr int kNAtoms = 4;

// 8 bytes. Matches the host FluxBlob packing used by afi-dequant-check.
struct FluxTile {
    uint8_t tile_id;  // 0..3
    int8_t row_sign;  // ±1
    int8_t col_sign;  // ±1
    uint8_t pad;
    float scale;
};
static_assert(sizeof(FluxTile) == 8, "FluxTile must stay 8 bytes");

// Host catalog: Sylvester H8 + three seam flips. int8 ±1.
void h8_atoms_host(int8_t out[kNAtoms][kAtom][kAtom]);

// Decode n_blocks tiles into a dense (br*8, bc*8) row-major float matrix.
// atoms is 4*8*8 int8 on device. tiles is n_blocks FluxTile on device.
void dequant_device(sycl::queue &q, const int8_t *atoms, const FluxTile *tiles,
                    float *out, int br, int bc);

} // namespace afi
