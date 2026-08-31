#include "afi/device.hpp"

#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char **argv) {
    std::string out_path;
    bool include_igpu = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if ((a == "-o" || a == "--output") && i + 1 < argc) {
            out_path = argv[++i];
        } else if (a == "--igpu") {
            include_igpu = true;
        } else if (a == "-h" || a == "--help") {
            std::cout << "afi-inventory [-o inventory/a770.json] [--igpu]\n";
            return 0;
        }
    }

    auto devs = afi::dgpu_devices(include_igpu);
    if (devs.empty()) {
        std::cerr << "no Level Zero GPUs. source scripts/env.sh\n";
        return 1;
    }
    std::vector<afi::DeviceInfo> infos;
    infos.reserve(devs.size());
    for (size_t i = 0; i < devs.size(); ++i) {
        infos.push_back(afi::inspect(devs[i], static_cast<int>(i)));
    }
    const std::string json = afi::to_json(infos);
    std::cout << json;
    if (!out_path.empty()) {
        std::ofstream f(out_path);
        if (!f) {
            std::cerr << "cannot write " << out_path << "\n";
            return 1;
        }
        f << json;
        std::cerr << "wrote " << out_path << "\n";
    }
    return 0;
}
