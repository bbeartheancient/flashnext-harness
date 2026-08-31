cmake_minimum_required(VERSION 3.25)

project(arrayfire-intel
    VERSION 0.1.0
    DESCRIPTION "SYCL FWHT + flux-tile kernels for Intel XPU"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_CXX_COMPILER MATCHES "icpx|icx|clang\\+\\+|dpclang")
    message(WARNING
        "CMAKE_CXX_COMPILER is '${CMAKE_CXX_COMPILER}'. "
        "B70 kernels: scripts/with-intel-llvm.sh cmake --preset llvm7. "
        "Alchemist pin: scripts/with-oneapi.sh cmake --preset default")
endif()

add_compile_options(-fsycl -Wall -Wextra -Wpedantic)
add_link_options(-fsycl)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(afi_sycl
    sycl/src/device.cpp
    sycl/src/fwht.cpp
    sycl/src/fwht_esimd.cpp
    sycl/src/qkint8.cpp
    sycl/src/dequant.cpp)
target_include_directories(afi_sycl PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/sycl/include)

# torch.ops.afi.fwht — llvm7 + the vLLM venv's libtorch. Off unless
# scripts/build-torch-fwht.sh passes -DAFI_TORCH=ON.
# Do not find_package(Torch): Caffe2's FindSYCLToolkit pulls oneAPI
# 2025.3 headers and mixes them with llvm7's stl_wrappers.
option(AFI_TORCH "Build torch.ops.afi.fwht XPU extension" OFF)
if(AFI_TORCH)
    if(NOT DEFINED Python_EXECUTABLE AND DEFINED AFI_PYTHON)
        set(Python_EXECUTABLE "${AFI_PYTHON}")
    endif()
    find_package(Python COMPONENTS Interpreter REQUIRED)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c
            "import torch, pathlib; p = pathlib.Path(torch.__file__).parent; print(p); print(int(torch.compiled_with_cxx11_abi()))"
        OUTPUT_VARIABLE _afi_torch_meta
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY)
    string(REPLACE "\n" ";" _afi_torch_meta "${_afi_torch_meta}")
    list(GET _afi_torch_meta 0 TORCH_INSTALL_PREFIX)
    list(GET _afi_torch_meta 1 _afi_torch_abi)
    add_library(afi_xpu SHARED
        sycl/torch/fwht_op.cpp
        sycl/src/fwht.cpp
        sycl/src/fwht_esimd.cpp)
    target_include_directories(afi_xpu PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/sycl/include)
    target_include_directories(afi_xpu SYSTEM PRIVATE
        ${TORCH_INSTALL_PREFIX}/include
        ${TORCH_INSTALL_PREFIX}/include/torch/csrc/api/include)
    target_compile_options(afi_xpu PRIVATE
        -Wno-unused-parameter
        -Wno-pedantic
        -Wno-unused-const-variable)
    target_compile_definitions(afi_xpu PRIVATE
        _GLIBCXX_USE_CXX11_ABI=${_afi_torch_abi})
    target_link_directories(afi_xpu PRIVATE ${TORCH_INSTALL_PREFIX}/lib)
    target_link_libraries(afi_xpu PRIVATE
        torch c10 torch_cpu torch_xpu c10_xpu)
    set_target_properties(afi_xpu PROPERTIES
        PREFIX ""
        SUFFIX ".so"
        BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
        INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib")
    if(DEFINED AFI_SYCL_RPATH)
        set_property(TARGET afi_xpu APPEND PROPERTY BUILD_RPATH "${AFI_SYCL_RPATH}")
        set_property(TARGET afi_xpu APPEND PROPERTY INSTALL_RPATH "${AFI_SYCL_RPATH}")
    endif()
endif()

add_executable(afi-inventory sycl/apps/inventory.cpp)
target_link_libraries(afi-inventory PRIVATE afi_sycl)

add_executable(afi-fwht-check sycl/apps/fwht_check.cpp)
target_link_libraries(afi-fwht-check PRIVATE afi_sycl)

add_executable(afi-dequant-check sycl/apps/dequant_check.cpp)
target_link_libraries(afi-dequant-check PRIVATE afi_sycl)

add_executable(afi-bench-fwht sycl/apps/bench_fwht.cpp)
target_link_libraries(afi-bench-fwht PRIVATE afi_sycl)

add_executable(afi-bench-fwht-esimd sycl/apps/bench_fwht_esimd.cpp)
target_link_libraries(afi-bench-fwht-esimd PRIVATE afi_sycl)

add_executable(afi-fwht-esimd-debug sycl/apps/fwht_esimd_debug.cpp)
target_link_libraries(afi-fwht-esimd-debug PRIVATE afi_sycl)

add_executable(afi-bench-xmx-tune sycl/apps/bench_xmx_tune.cpp)
target_link_libraries(afi-bench-xmx-tune PRIVATE afi_sycl)

add_executable(afi-bench-qkint8 sycl/apps/bench_qkint8.cpp)
target_link_libraries(afi-bench-qkint8 PRIVATE afi_sycl)

add_executable(afi-qkint8-check sycl/apps/qkint8_check.cpp)
target_link_libraries(afi-qkint8-check PRIVATE afi_sycl)

add_executable(afi-bench-qkint8-tune sycl/apps/bench_qkint8_tune.cpp)
target_link_libraries(afi-bench-qkint8-tune PRIVATE afi_sycl)

add_executable(afi-bench-dpas-esimd sycl/apps/bench_dpas_esimd.cpp)
target_link_libraries(afi-bench-dpas-esimd PRIVATE afi_sycl)

add_executable(afi-dpas-vnni-layout sycl/apps/dpas_vnni_layout.cpp)
target_link_libraries(afi-dpas-vnni-layout PRIVATE afi_sycl)

# B70 RTU probe: L0 RTAS builder + IGC-lowered ray queries. Not part of
# afi_sycl (standalone; links the Level Zero loader directly).
find_library(AFI_ZE_LOADER ze_loader REQUIRED)
add_executable(afi-rtu-probe sycl/apps/rtu_probe.cpp)
target_include_directories(afi-rtu-probe PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/sycl/include
    /usr/include/level_zero)
target_link_libraries(afi-rtu-probe PRIVATE ${AFI_ZE_LOADER})

# M2 site_survey race: RTU vs CPU Embree on identical heightfields.
find_library(AFI_EMBREE embree4 REQUIRED)
add_executable(afi-rtu-terrain sycl/apps/rtu_terrain.cpp)
target_include_directories(afi-rtu-terrain PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/sycl/include
    /usr/include/level_zero)
target_link_libraries(afi-rtu-terrain PRIVATE ${AFI_ZE_LOADER} ${AFI_EMBREE})

# libafi_rtu — C API for python/afi/rtu.py (heightfield + generic mesh BLAS).
add_library(afi_rtu SHARED sycl/src/rtu_api.cpp)
target_include_directories(afi_rtu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/sycl/include
    /usr/include/level_zero)
target_link_libraries(afi_rtu PRIVATE ${AFI_ZE_LOADER})
set_target_properties(afi_rtu PROPERTIES OUTPUT_NAME "afi_rtu")

install(TARGETS afi-inventory afi-fwht-check afi-dequant-check afi-bench-fwht
        afi-bench-xmx-tune afi-bench-qkint8 afi-qkint8-check
        afi-bench-qkint8-tune afi-bench-dpas-esimd)
