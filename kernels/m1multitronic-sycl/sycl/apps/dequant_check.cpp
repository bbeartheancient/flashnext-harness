#include "afi/dequant.hpp"
#include "afi/device.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    int dev_index = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if ((a == "-d" || a == "--device") && i + 1 < argc) {
            dev_index = std::stoi(argv[++i]);
        }
    }
    try {
        auto dev = afi::pick_dgpu(dev_index);
        std::cout << "device: " << dev.get_info<sycl::info::device::name>() << "\n";
        sycl::queue q(dev);

        constexpr int br = 4, bc = 4;
        constexpr int M = br * afi::kAtom;
        constexpr int N = bc * afi::kAtom;
        int8_t atoms[afi::kNAtoms][afi::kAtom][afi::kAtom];
        afi::h8_atoms_host(atoms);

        std::mt19937 rng(0xB70);
        std::uniform_int_distribution<int> tid(0, afi::kNAtoms - 1);
        std::uniform_real_distribution<float> sc(0.25f, 2.0f);
        std::vector<afi::FluxTile> tiles(br * bc);
        std::vector<float> expect(M * N);
        for (int bi = 0; bi < br; ++bi) {
            for (int bj = 0; bj < bc; ++bj) {
                auto &t = tiles[bi * bc + bj];
                t.tile_id = static_cast<uint8_t>(tid(rng));
                t.row_sign = 1;
                t.col_sign = (bj & 1) ? -1 : 1;
                t.pad = 0;
                t.scale = sc(rng);
                for (int i = 0; i < afi::kAtom; ++i) {
                    for (int j = 0; j < afi::kAtom; ++j) {
                        const float v = static_cast<float>(atoms[t.tile_id][i][j]) *
                                        static_cast<float>(t.row_sign) *
                                        static_cast<float>(t.col_sign) * t.scale;
                        expect[(bi * afi::kAtom + i) * N + (bj * afi::kAtom + j)] = v;
                    }
                }
            }
        }

        int8_t *d_atoms = sycl::malloc_device<int8_t>(afi::kNAtoms * 64, q);
        auto *d_tiles = sycl::malloc_device<afi::FluxTile>(tiles.size(), q);
        float *d_out = sycl::malloc_device<float>(expect.size(), q);
        q.memcpy(d_atoms, atoms, sizeof(atoms)).wait();
        q.memcpy(d_tiles, tiles.data(), tiles.size() * sizeof(afi::FluxTile)).wait();
        afi::dequant_device(q, d_atoms, d_tiles, d_out, br, bc);
        std::vector<float> got(expect.size());
        q.memcpy(got.data(), d_out, expect.size() * sizeof(float)).wait();
        sycl::free(d_atoms, q);
        sycl::free(d_tiles, q);
        sycl::free(d_out, q);

        double max_abs = 0.0;
        for (size_t i = 0; i < expect.size(); ++i) {
            max_abs = std::max(max_abs, std::abs(static_cast<double>(got[i] - expect[i])));
        }
        const bool ok = max_abs < 1e-6;
        std::cout << "dequant " << M << "x" << N << " max_abs=" << max_abs
                  << (ok ? "  ok\n" : "  FAIL\n");
        return ok ? 0 : 1;
    } catch (const std::exception &e) {
        std::cerr << "dequant-check: " << e.what() << "\n";
        return 1;
    }
}
