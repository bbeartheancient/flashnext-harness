#include "afi/dequant.hpp"

#include <stdexcept>

namespace afi {
namespace {

class DequantKernel {
public:
    DequantKernel(const int8_t *atoms, const FluxTile *tiles, float *out, int br, int bc,
                  sycl::local_accessor<int8_t, 1> slm)
        : atoms_(atoms), tiles_(tiles), out_(out), br_(br), bc_(bc), slm_(slm) {}

    void operator()(sycl::nd_item<1> it) const {
        // One work-group per 8×8 tile. 64 work-items, one per cell.
        const int block = static_cast<int>(it.get_group(0));
        const int lid = static_cast<int>(it.get_local_id(0));
        const int n_blocks = br_ * bc_;
        if (block >= n_blocks) return;

        // Catalog is 4×64 int8. Each of 64 lanes copies one cell per atom.
        for (int r = 0; r < kNAtoms; ++r) {
            slm_[r * 64 + lid] = atoms_[r * 64 + lid];
        }
        sycl::group_barrier(it.get_group());

        const FluxTile t = tiles_[block];
        const int cell_r = lid / kAtom;
        const int cell_c = lid % kAtom;
        const int8_t a = slm_[static_cast<int>(t.tile_id) * 64 + lid];
        const float v = static_cast<float>(a) * static_cast<float>(t.row_sign) *
                        static_cast<float>(t.col_sign) * t.scale;

        const int br = block / bc_;
        const int bc = block % bc_;
        const int out_r = br * kAtom + cell_r;
        const int out_c = bc * kAtom + cell_c;
        out_[out_r * (bc_ * kAtom) + out_c] = v;
    }

private:
    const int8_t *atoms_;
    const FluxTile *tiles_;
    float *out_;
    int br_;
    int bc_;
    sycl::local_accessor<int8_t, 1> slm_;
};

} // namespace

void h8_atoms_host(int8_t out[kNAtoms][kAtom][kAtom]) {
    int8_t H[kAtom][kAtom]{};
    H[0][0] = 1;
    for (int n = 1; n < kAtom; n *= 2) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                const int8_t v = H[i][j];
                H[i][j + n] = v;
                H[i + n][j] = v;
                H[i + n][j + n] = static_cast<int8_t>(-v);
            }
        }
    }
    for (int a = 0; a < kNAtoms; ++a)
        for (int i = 0; i < kAtom; ++i)
            for (int j = 0; j < kAtom; ++j) out[a][i][j] = H[i][j];
    for (int i = 0; i < kAtom; ++i)
        out[1][i][kAtom - 1] = static_cast<int8_t>(-out[1][i][kAtom - 1]);
    for (int j = 0; j < kAtom; ++j)
        out[2][kAtom - 1][j] = static_cast<int8_t>(-out[2][kAtom - 1][j]);
    for (int i = 0; i < kAtom; ++i)
        out[3][i][kAtom - 1] = static_cast<int8_t>(-out[3][i][kAtom - 1]);
    for (int j = 0; j < kAtom; ++j)
        out[3][kAtom - 1][j] = static_cast<int8_t>(-out[3][kAtom - 1][j]);
}

void dequant_device(sycl::queue &q, const int8_t *atoms, const FluxTile *tiles, float *out,
                    int br, int bc) {
    if (br <= 0 || bc <= 0) throw std::invalid_argument("dequant: empty grid");
    const size_t n_blocks = static_cast<size_t>(br) * static_cast<size_t>(bc);
    q.submit([&](sycl::handler &h) {
        sycl::local_accessor<int8_t, 1> slm(sycl::range<1>(kNAtoms * kAtom * kAtom), h);
        h.parallel_for(sycl::nd_range<1>({n_blocks * 64}, {64}),
                       DequantKernel(atoms, tiles, out, br, bc, slm));
    });
    q.wait_and_throw();
}

} // namespace afi
